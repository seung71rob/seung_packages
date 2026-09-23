"""
robot_description / rail_serial_5axis.py   (v2)
--------------------------------------------------
아두이노(rail_stepmotor.ino v2) 시리얼 리더 / 명령 송신.

기계 제원
  CS-D508 1600 pulse/rev, 볼스크류 리드 10 mm  ->  160 pulse/mm
  1 pulse = 6.25 um,  1 m = 160,000 pulse
  스트로크 0 ~ 900 mm = 0 ~ 144,000 pulse

아두이노 기계 출력 ('@' 로 시작하는 줄만 파싱)
  @P <pulse> <zeroSet>   위치 보고 (이동 중 주기적)
  @D <pulse> <zeroSet>   이동 완료
  @L <pulse> <zeroSet>   소프트리미트 / 원점 미설정으로 거부
  @Z <pulse> <zeroSet>   원점 설정됨
  @S <pulse> <zeroSet>   정지됨
  @V <0|1>               흡착 MOSFET 상태 (0=OFF, 1=ON)
사람용 출력은 무시한다.

사용
  rail = RailSerial(port='/dev/ttyACM0')
  rail.connect()
  rail.zero()                 # 원점 설정 (이동 전 필수)
  rail.move_abs_mm(500)       # 절대 500 mm
  rail.move_rel_mm(-100)      # 상대 -100 mm
  x = rail.position_m         # 현재 위치 (m)
  rail.smooth_stop()          # s
  rail.estop()                # 0
  rail.suction(True)          # von  흡착 ON
  rail.suction(False)         # voff 흡착 OFF
  rail.suction_on             # 현재 흡착 상태 (읽기)
  rail.close()

pyserial 필요:  pip install pyserial
"""

import threading
import time

PULSES_PER_REV = 1600
LEAD_MM = 10.0
PULSES_PER_MM = PULSES_PER_REV / LEAD_MM        # 160
STEPS_PER_M = int(PULSES_PER_MM * 1000)         # 160000
M_PER_STEP = 1.0 / STEPS_PER_M                  # 6.25e-6 m

# 아두이노 소프트리미트와 URDF joint_rail limit 에 맞출 것
RAIL_MIN_M = 0.0
RAIL_MAX_M = 0.9


class RailSerial:
    """아두이노 레일 컨트롤러와의 시리얼 연결. 읽기는 백그라운드 스레드."""

    def __init__(self, port='/dev/ttyACM1', baud=115200, logger=None):
        self.port_name = port
        self.baud = baud
        self.logger = logger
        self.ser = None

        self._pulse = 0
        self._zero_set = False
        self._moving = False
        self._suction = False
        self._lock = threading.Lock()
        self._thread = None
        self._running = False

        self.last_rx = 0.0
        self.last_event = ''

    # ---------------- 연결 ----------------
    def connect(self):
        import serial                      # pyserial
        self.ser = serial.Serial(self.port_name, self.baud, timeout=0.05)
        time.sleep(2.0)                    # 아두이노 자동 리셋 대기
        self.ser.reset_input_buffer()
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        self.send('?')
        self._log('레일 시리얼 연결: %s @ %d' % (self.port_name, self.baud))

    def close(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self.ser is not None:
            try:
                self.ser.write(b'voff\n')  # 안전: 나가면서 흡착 OFF
                self.ser.flush()
                time.sleep(0.02)
                self.ser.write(b'0\n')     # 안전: 나가면서 즉시 정지
                self.ser.flush()
                time.sleep(0.05)
            except Exception:
                pass
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None
        self._log('레일 시리얼 닫음')

    # ---------------- 읽기 ----------------
    def _reader(self):
        buf = b''
        while self._running:
            try:
                data = self.ser.read(256)
            except Exception:
                break
            if data:
                buf += data
                while b'\n' in buf:
                    line, buf = buf.split(b'\n', 1)
                    self._parse(line.decode('ascii', 'ignore').strip())
            else:
                time.sleep(0.005)

    def _parse(self, line):
        # 기계 출력만 처리. 사람용 텍스트는 무시.
        if not line.startswith('@') or len(line) < 2:
            return
        tag = line[1]
        parts = line[2:].split()
        if not parts:
            return

        if tag == 'V':                      # 흡착 MOSFET 상태
            with self._lock:
                self._suction = (parts[0] == '1')
                self.last_rx = time.time()
            self._log('흡착 %s' % ('ON' if parts[0] == '1' else 'OFF'))
            return

        try:
            pulse = int(parts[0])
        except ValueError:
            return
        zero = (len(parts) > 1 and parts[1] == '1')

        with self._lock:
            self._pulse = pulse
            self._zero_set = zero
            self.last_rx = time.time()
            self.last_event = tag
            if tag in ('D', 'S', 'L', 'Z'):
                self._moving = False
            elif tag == 'P':
                self._moving = True

        if tag == 'L':
            self._log('레일 이동 거부 (소프트리미트 또는 원점 미설정)')
        elif tag == 'Z':
            self._log('레일 원점 설정됨')
        elif tag == 'D':
            self._log('레일 이동 완료: %.3f m' % (pulse * M_PER_STEP))

    def poll(self):
        """호환용. 읽기는 스레드가 하므로 실제로 할 일은 없다."""
        return self.position_m

    # ---------------- 상태 ----------------
    @property
    def steps(self):
        with self._lock:
            return self._pulse

    @property
    def position_m(self):
        with self._lock:
            return self._pulse * M_PER_STEP

    @property
    def position_mm(self):
        with self._lock:
            return self._pulse / PULSES_PER_MM

    @property
    def suction_on(self):
        """흡착 MOSFET 이 켜져 있는지."""
        with self._lock:
            return self._suction

    @property
    def zero_set(self):
        with self._lock:
            return self._zero_set

    @property
    def moving(self):
        with self._lock:
            return self._moving

    @property
    def connected(self):
        return self.ser is not None and self._running

    def stale(self, sec=3.0):
        """sec 초 이상 수신이 없으면 True."""
        with self._lock:
            if self.last_rx == 0.0:
                return True
            return (time.time() - self.last_rx) > sec

    # ---------------- 명령 ----------------
    def send(self, cmd):
        if self.ser is None:
            return False
        try:
            self.ser.write((str(cmd).strip() + '\n').encode('ascii'))
            self.ser.flush()
            return True
        except Exception as e:
            self._log('레일 명령 전송 실패: %s' % e)
            return False

    def zero(self):
        """현재 위치를 0 mm 로 설정. 이동 전 반드시 1회."""
        return self.send('z')

    def move_abs_mm(self, mm):
        """절대 위치(mm)로 이동."""
        return self.send('p %.3f' % float(mm))

    def move_abs_m(self, m):
        return self.move_abs_mm(float(m) * 1000.0)

    def move_rel_mm(self, mm):
        """상대 이동(mm). 음수면 반대방향."""
        return self.send('m %.3f' % float(mm))

    def move_rel_m(self, m):
        return self.move_rel_mm(float(m) * 1000.0)

    def to_end(self):
        """스트로크 끝까지 (= p MAX)."""
        return self.send('c')

    def to_origin(self):
        """원점까지 (= p 0)."""
        return self.send('ccw')

    def smooth_stop(self):
        """S-curve 감속 정지."""
        return self.send('s')

    def estop(self):
        """즉시 정지 (펄스 출력 중단). 관성으로 탈조 가능."""
        return self.send('0')

    def query(self):
        return self.send('?')

    def suction(self, on):
        """흡착 MOSFET on/off (아두이노 von / voff)."""
        return self.send('von' if on else 'voff')

    def suction_query(self):
        return self.send('v?')

    # 구버전 호환
    def cw(self):
        return self.to_end()

    def ccw(self):
        return self.to_origin()

    def stop(self):
        return self.smooth_stop()

    # ---------------- 내부 ----------------
    def _log(self, msg):
        if self.logger is not None:
            try:
                self.logger.info(msg)
                return
            except Exception:
                pass
        print('[RAIL] ' + msg)
