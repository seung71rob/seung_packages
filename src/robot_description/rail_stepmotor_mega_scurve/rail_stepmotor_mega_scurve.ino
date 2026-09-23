/*
 * Mega 2560 / 16 MHz ONLY. Rail firmware, 500 RPM time-based S-curve.
 * Pins unchanged: D2 PUL (PE4), D3 DIR, D4 suction, active HIGH.
 * CS-D508: 1600 input pulses/rev, screw lead 10 mm/rev.
 * Timer1 is reserved; do not combine with Servo/Timer1/PWM D11,D12.
 * No PULSE_OVERHEAD_US. Timer0 (millis/micros) remains available.
 *
 * Arduino IDE: save in folder rail_stepmotor_mega_scurve, select
 * Arduino Mega or Mega 2560 / ATmega2560, then compile/upload.
 * Serial: 115200 baud, newline. z sets CURRENT position to zero;
 * this is not automatic homing. Positions are commanded pulse counts,
 * NOT encoder measurements. Existing ROS @P/D/L/Z/S/V format retained.
 * Commands: z, p <mm>, m <mm>, c/cw, cc/ccw, s, 0, ?, von/v1,
 * voff/v0, v?, help. Motion range: 0..900 mm.
 * During motion p/m/z are rejected without stopping the current move.
 * s and 0 require a newline, like the other commands. No digit scanning:
 * the '0' in 'p 700' cannot become an emergency stop.
 * s = smooth stop along the CURRENT path; never passes its target.
 * 0 = stop pulse output when parsed (not a hardwired emergency stop).
 * Both keep suction ON if it was ON; voff explicitly releases it.
 * Boot/reset turns suction OFF, and loses the position reference.
 *
 * Normal speed: 0 -> 500 RPM -> 0, 2 seconds per ramp on long moves.
 * A ramp uses v=V*(6u^5-15u^4+10u^3); its integral is used for position.
 * Velocity, acceleration and jerk join continuously in the ideal curve.
 * Short moves reduce peak speed/ramp time while retaining jerk scaling.
 * For 700 mm: 83.333 mm ramp + 533.333 cruise + 83.333 ramp;
 * planned total 10.4 seconds; total average is NOT 500 RPM.
 *
 * Implementation: position sampled every 5 ms in the foreground.
 * Timer1 CTC, prescaler 8 (0.5 us ticks), emits integer pulses within
 * each segment. No floating point or serial calls inside the ISR.
 * Individual pulse spacing is quantized; 500 RPM is the cruise average.
 * PUL high >=5 us, low normally >=~60 us at maximum speed.
 * Interrupt latency adds small edge jitter; this is not hardware PWM.
 * Queue underrun or late timer servicing STOPS rather than catching up.
 *
 * Smooth stop: time-warp the remaining original path, with clock rate
 * r=1-smootherstep(u), u=t/STOP_TIME_S. Integral advances the path clock
 * by STOP_TIME_S/2 total. Starts with r=1,r'=r''=0, ends r=r'=r''=0.
 * This preserves acceleration at a stop request even during a ramp.
 * At most ~40 ms of already queued motion precedes the 2 s stop ramp.
 * At full cruise, stop distance <=~87 mm including that queue.
 * If already accelerating, velocity can briefly keep rising smoothly
 * before falling; it never exceeds the original path speed bound.
 * This is a controlled stop, not the immediate '0' command.
 *
 * Tunables below: MAX_RPM, RAMP_TIME_S, STOP_TIME_S.
 * Leave MAX_RPM<=500 with this timing budget. Increase ramp times to
 * soften motion. Real motor speed/load tracking must be measured.
 * First validate without a held workpiece. This firmware has no encoder
 * feedback, physical limit-switch input, or motor fault input.
 */
#include <Arduino.h>
#include <avr/interrupt.h>
#include <util/atomic.h>
#include <util/delay.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#if !defined(__AVR_ATmega2560__) || F_CPU != 16000000UL
#error "Select Arduino Mega 2560 (ATmega2560), 16 MHz"
#endif

const uint8_t PUL_PIN=2, DIR_PIN=3, SUCTION_PIN=4;
const float PULSES_PER_MM=160.0f, LEAD_MM=10.0f;
const float MAX_RPM=500.0f;
const float RAMP_TIME_S=2.0f, STOP_TIME_S=2.0f;
const long MIN_PULSE=0, MAX_PULSE=144000L; // 900 mm
const uint16_t SEG_TICKS=10000; // 5 ms at 2 MHz
const float SEG_SECONDS=0.005f;
const uint8_t QUEUE_SIZE=8, QUEUE_MASK=QUEUE_SIZE-1;
const uint16_t MIN_INTERVAL_TICKS=140; // 70 us minimum

// Precomputed integer timing: exactly SEG_TICKS per segment.
struct Segment {
  uint16_t baseTicks, remainder, divisions;
  uint8_t pulses; // <=67 at 500 RPM, occasionally 68 from float rounding
  bool last;
};
Segment segments[QUEUE_SIZE];
volatile uint8_t qHead=0, qTail=0;
volatile bool running=false;
volatile uint8_t completion=0; // 1 done, 2 stop, 3 underrun, 4 timing fault
volatile long currentPulse=0;
volatile int8_t direction=1;
volatile long endLimit=0;
volatile uint8_t terminalCode=1;
bool zeroSet=false, suctionOn=false;

// ISR-owned current segment.
Segment active;
uint16_t intervalsLeft=0, errorTicks=0;
bool activePulse=false;

// Foreground planner: all distances relative to the move start.
long moveStart=0, moveDistance=0, plannedPulses=0;
float peak=0, rampTime=0, cruiseTime=0, totalTime=0;
uint32_t plannedSegments=0, stopSegments=0;
float stopBaseTime=0;
bool stopping=false, plannerFinished=false;
uint32_t lastReportMs=0;
long lastReportPulse=0;

long positionSnapshot() {
  long p;
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) { p=currentPulse; }
  return p;
}
void emit(char tag) {
  Serial.print('@'); Serial.print(tag); Serial.print(' ');
  Serial.print(positionSnapshot()); Serial.print(' ');
  Serial.println(zeroSet ? 1 : 0);
}
void emitSuction() { Serial.print(F("@V ")); Serial.println(suctionOn ? 1 : 0); }
void setSuction(bool on) {
  suctionOn=on; digitalWrite(SUCTION_PIN,on ? HIGH : LOW); emitSuction();
}

// Called with interrupts disabled. CTC clears TCNT1 at each compare.
bool scheduleInterval() {
  uint16_t ticks=active.baseTicks;
  errorTicks+=active.remainder;
  if (errorTicks>=active.divisions) { errorTicks-=active.divisions; ++ticks; }
  // OCR must still be ahead of TCNT; never wrap and create a long stall.
  if (ticks<MIN_INTERVAL_TICKS || (uint32_t)TCNT1+20UL>=ticks) {
    TIMSK1=0; TCCR1B=0; running=false; completion=4;
    PORTE &= ~_BV(PE4);
    return false;
  }
  OCR1A=ticks-1;
  return true;
}
bool loadSegment() {
  if (qTail==qHead) {
    TIMSK1=0; TCCR1B=0; running=false; completion=3;
    PORTE &= ~_BV(PE4);
    return false;
  }
  active=segments[qTail];
  qTail=(qTail+1)&QUEUE_MASK;
  intervalsLeft=active.divisions; errorTicks=0;
  activePulse=active.pulses!=0;
  return scheduleInterval();
}
ISR(TIMER1_COMPA_vect) {
  if (!running) return;
  if (activePulse) {
    long next=currentPulse+direction;
    if (next<MIN_PULSE || next>MAX_PULSE ||
        (direction>0 ? next>endLimit : next<endLimit)) {
      TIMSK1=0; TCCR1B=0; running=false; completion=4; return;
    }
    PORTE |= _BV(PE4);
    _delay_us(5);
    PORTE &= ~_BV(PE4);
    currentPulse=next;
  }
  if (--intervalsLeft) { scheduleInterval(); return; }
  if (active.last) {
    TIMSK1=0; TCCR1B=0; running=false; completion=terminalCode;
  } else loadSegment();
}

float integralS(float u) {
  if (u<=0) return 0;
  if (u>=1) return 0.5f;
  float u2=u*u;
  return u2*u2*(2.5f+u*(-3.0f+u));
}
float positionAt(float t) {
  if (t<=0) return 0;
  if (t>=totalTime) return (float)moveDistance;
  float mm;
  if (t<rampTime) mm=peak*rampTime*integralS(t/rampTime);
  else if (t<rampTime+cruiseTime)
    mm=peak*(0.5f*rampTime+t-rampTime);
  else {
    float u=(t-rampTime-cruiseTime)/rampTime;
    mm=peak*(0.5f*rampTime+cruiseTime+rampTime*(u-integralS(u)));
  }
  return mm*PULSES_PER_MM;
}

bool queueHasRoom() { return ((qHead+1)&QUEUE_MASK)!=qTail; }
bool addSegment() {
  if (plannerFinished || !queueHasRoom()) return false;
  float t;
  bool last;
  if (stopping) {
    float elapsed=(float)(++stopSegments)*SEG_SECONDS;
    float u=elapsed/STOP_TIME_S;
    if (u>1) u=1;
    t=stopBaseTime+STOP_TIME_S*(u-integralS(u));
    last=elapsed>=STOP_TIME_S || t>=totalTime;
  } else {
    t=(float)(++plannedSegments)*SEG_SECONDS;
    last=t>=totalTime;
  }
  long desired=lroundf(positionAt(t));
  if (desired<plannedPulses) desired=plannedPulses;
  if (desired>moveDistance) desired=moveDistance;
  long n=desired-plannedPulses;
  if (n>71) { // Cannot honor this segment within the timer budget.
    ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
      TIMSK1=0; TCCR1B=0; running=false; completion=4;
    }
    plannerFinished=true; return false;
  }
  Segment s;
  s.pulses=(uint8_t)n; s.divisions=n ? n : 1;
  s.baseTicks=SEG_TICKS/s.divisions;
  s.remainder=SEG_TICKS%s.divisions; s.last=last;
  segments[qHead]=s;
  // Publish only after the whole structure is written.
  asm volatile("" ::: "memory");
  qHead=(qHead+1)&QUEUE_MASK;
  plannedPulses=desired; plannerFinished=last;
  return true;
}

void startMove(long target) {
  if (running || completion) { Serial.println(F("[BUSY] Move rejected")); return; }
  if (!zeroSet || target<MIN_PULSE || target>MAX_PULSE) { emit('L'); return; }
  moveStart=positionSnapshot(); moveDistance=labs(target-moveStart);
  if (!moveDistance) { emit('D'); return; }
  direction=target>moveStart ? 1 : -1;
  endLimit=target;
  digitalWrite(DIR_PIN,direction>0 ? HIGH : LOW);
  delayMicroseconds(20);
  float distanceMM=(float)moveDistance/PULSES_PER_MM;
  peak=MAX_RPM*LEAD_MM/60.0f; rampTime=RAMP_TIME_S;
  if (distanceMM<peak*rampTime) {
    rampTime=RAMP_TIME_S*powf(distanceMM/(peak*RAMP_TIME_S),1.0f/3.0f);
    peak=distanceMM/rampTime; cruiseTime=0;
  } else cruiseTime=distanceMM/peak-rampTime;
  totalTime=2*rampTime+cruiseTime;
  plannedPulses=0; plannedSegments=0; stopSegments=0;
  stopping=false; plannerFinished=false; terminalCode=1;
  qHead=qTail=0;
  Serial.print(F("[RUN] peak RPM=")); Serial.print(peak*60/LEAD_MM,2);
  Serial.print(F(" ramp s=")); Serial.print(rampTime,3);
  Serial.print(F(" total s=")); Serial.println(totalTime,3);
  while (queueHasRoom() && !plannerFinished) addSegment();
  if (completion) return;
  lastReportPulse=moveStart; lastReportMs=millis();
  emit('P');
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
    TCCR1A=0; TCCR1B=0; TIMSK1=0; TCNT1=0;
    TIFR1=_BV(OCF1A)|_BV(TOV1);
    running=true;
    if (loadSegment()) { TIMSK1=_BV(OCIE1A); TCCR1B=_BV(WGM12)|_BV(CS11); }
  }
}
void immediateStop() {
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
    TIMSK1=0; TCCR1B=0; running=false; completion=0;
    qHead=qTail=0; PORTE &= ~_BV(PE4);
  }
  plannerFinished=true; emit('S'); // suction unchanged
}
void smoothStop() {
  if (!running) { emit('S'); return; }
  if (stopping) return;
  // Already queued end-of-move deceleration is smooth; let it finish.
  terminalCode=2;
  if (plannerFinished) return;
  stopBaseTime=(float)plannedSegments*SEG_SECONDS;
  stopSegments=0; stopping=true;
  Serial.println(F("[STOP] Smooth deceleration requested"));
}

bool parseMM(const char *text, float &out) {
  char *end;
  out=strtod(text,&end);
  if (end==text || !isfinite(out)) return false;
  while (*end && isspace((unsigned char)*end)) ++end;
  return *end==0;
}
void processCommand(char *cmd) {
  while (*cmd && isspace((unsigned char)*cmd)) ++cmd;
  size_t len=strlen(cmd);
  while (len && isspace((unsigned char)cmd[len-1])) cmd[--len]=0;
  for (size_t i=0;i<len;++i) cmd[i]=tolower((unsigned char)cmd[i]);
  if (!len) return;
  if (!strcmp(cmd,"0")) { immediateStop(); return; }
  if (!strcmp(cmd,"s")) { smoothStop(); return; }
  if (!strcmp(cmd,"von") || !strcmp(cmd,"v1")) { setSuction(true); return; }
  if (!strcmp(cmd,"voff") || !strcmp(cmd,"v0")) { setSuction(false); return; }
  if (!strcmp(cmd,"v?")) { emitSuction(); return; }
  if (!strcmp(cmd,"?")) {
    // Finish tags clear RailSerial.moving; idle queries must not emit @P.
    emit(running ? 'P' : 'D'); return;
  }
  if (!strcmp(cmd,"help")) {
    Serial.println(F("z | p mm | m mm | c/cw | cc/ccw | s | 0 | ? | von | voff | v?"));
    return;
  }
  if (running || completion) { Serial.println(F("[BUSY] Command rejected")); return; }
  if (!strcmp(cmd,"z")) {
    ATOMIC_BLOCK(ATOMIC_RESTORESTATE) { currentPulse=0; }
    zeroSet=true; emit('Z'); return;
  }
  if (!strcmp(cmd,"c") || !strcmp(cmd,"cw")) { startMove(MAX_PULSE); return; }
  if (!strcmp(cmd,"cc") || !strcmp(cmd,"ccw")) { startMove(MIN_PULSE); return; }
  if ((cmd[0]=='p' || cmd[0]=='m') && isspace((unsigned char)cmd[1])) {
    float mm;
    if (!parseMM(cmd+2,mm)) { Serial.println(F("[ERROR] Invalid number")); emit('L'); return; }
    if (cmd[0]=='m') mm+=(float)positionSnapshot()/PULSES_PER_MM;
    if (mm<0 || mm>900) { emit('L'); return; }
    startMove(lroundf(mm*PULSES_PER_MM)); return;
  }
  Serial.println(F("[ERROR] Unknown command; send help"));
}

char rxLine[64];
uint8_t rxLen=0;
bool rxOverflow=false;
void pollSerial() {
  // Bounded input work so an input flood cannot starve the planner.
  for (uint8_t i=0;i<16 && Serial.available();++i) {
    char c=Serial.read();
    if (c=='\r') continue;
    if (c=='\n') {
      if (!rxOverflow) { rxLine[rxLen]=0; processCommand(rxLine); }
      else Serial.println(F("[ERROR] Command too long; discarded"));
      rxLen=0; rxOverflow=false;
    } else if (!rxOverflow) {
      if (rxLen<sizeof(rxLine)-1) rxLine[rxLen++]=c;
      else rxOverflow=true;
    }
  }
}
void setup() {
  pinMode(PUL_PIN,OUTPUT); pinMode(DIR_PIN,OUTPUT); pinMode(SUCTION_PIN,OUTPUT);
  digitalWrite(PUL_PIN,LOW); digitalWrite(DIR_PIN,LOW); digitalWrite(SUCTION_PIN,LOW);
  TCCR1A=0; TCCR1B=0; TIMSK1=0;
  Serial.begin(115200);
  Serial.println(F("Mega 2560 rail: 500 RPM, Timer1, S-curve. Set origin with z."));
  Serial.println(F("D2=PUL D3=DIR D4=suction; s=soft stop, 0=immediate; newline required."));
  emit('D'); emitSuction();
}
void loop() {
  if (running && !plannerFinished) while (queueHasRoom() && !plannerFinished) addSegment();
  uint8_t done;
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) { done=completion; completion=0; }
  if (done) {
    plannerFinished=true;
    if (done>=3) {
      Serial.println(done==3 ? F("[FAULT] Planner underrun; pulses stopped") :
                              F("[FAULT] Timer/position limit; pulses stopped"));
      // Require explicit reference confirmation after a timing fault.
      zeroSet=false;
    }
    emit(done==1 ? 'D' : 'S');
  }
  pollSerial();
  if (running && (uint32_t)(millis()-lastReportMs)>=50) {
    long pos=positionSnapshot();
    if (pos!=lastReportPulse && Serial.availableForWrite()>=24) {
      emit('P'); lastReportPulse=pos;
    }
    lastReportMs=millis();
  }
}
