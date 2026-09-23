#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3-jaw centric gripper driver and optional ROS 2 standalone node.

The :class:`CentricGripper` class is imported by centric_sequence_5axis.py.
This module can also be run independently through ROS 2 for gripper tests.

Important: with ``capture_open_on_start:=true`` (the default), the gripper
must be physically fully open before this program starts.  The present motor
position is then stored as the 0% OPEN reference, exactly like the original
centric_gripper_test.py.
"""

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler


# XL430 control table
ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_HARDWARE_ERROR_STATUS = 70
ADDR_PWM_LIMIT = 36        # EEPROM. 위치 모드에서 출력을 제한하는 값 (0~885)
ADDR_GOAL_PWM = 100        # PWM 제어 모드 전용. Extended Position 에서는 무시된다
ADDR_PROFILE_ACCELERATION = 108
ADDR_PROFILE_VELOCITY = 112
ADDR_GOAL_POSITION = 116
ADDR_MOVING = 122
ADDR_PRESENT_PWM = 124
ADDR_PRESENT_LOAD = 126
ADDR_PRESENT_VELOCITY = 128
ADDR_PRESENT_POSITION = 132
ADDR_PRESENT_INPUT_VOLTAGE = 144
ADDR_PRESENT_TEMPERATURE = 146

TORQUE_DISABLE = 0
TORQUE_ENABLE = 1
EXTENDED_POSITION_MODE = 4
PROTOCOL_VERSION = 2.0


@dataclass(frozen=True)
class GraspResult:
    success: bool
    reason: str
    position: Optional[int] = None
    load_percent: Optional[float] = None


class CentricGripper:
    """Dynamixel ID 8 controller for the 18T/22T centric gripper."""

    def __init__(
        self,
        port="/dev/ttyUSB1",
        baud=2_000_000,
        dxl_id=8,
        logger=None,
        close_direction=+1,
        sun_teeth=18,
        planet_teeth=22,
        planet_max_angle_deg=160.0,
        safety_margin_pulse=31,
        normal_profile_velocity=44,
        auto_profile_velocity=22,
        profile_acceleration=5,
        goal_pwm_limit=250,
        pwm_limit=300,                 # 주소 36. 885=100%. 파지 전류 상한
        grasp_target_load_percent=20.0,  # 이 부하에 닿으면 멈추고 유지
        grasp_approach_step=40,          # 접촉 전 목표 증분(펄스)
        grasp_contact_step=20,           # 접촉 후 목표 증분(펄스)
        grasp_contact_load=5.0,          # 이 부하를 넘으면 접촉으로 본다
        grasp_acceleration=3,            # 파지 중 가속도 (전류 스파이크 감소)
        brownout_warn_v=10.5,
        auto_load_threshold_percent=12.0,
        auto_load_count_limit=8,
        auto_sample_sec=0.02,
        hold_extra_pulse=0,
        min_grasp_travel_pulse=100,
        auto_grasp_timeout_sec=8.0,
        emergency_load_percent=35.0,
        temperature_stop_c=60,
        position_tolerance_pulse=15,
        move_timeout_sec=10.0,
    ):
        self.port_name = str(port)
        self.baud = int(baud)
        self.dxl_id = int(dxl_id)
        self.logger = logger
        self.close_direction = 1 if int(close_direction) >= 0 else -1

        motor_max_angle = (
            float(planet_max_angle_deg) * float(planet_teeth) / float(sun_teeth)
        )
        self.max_move_pulse = int(round(motor_max_angle / 360.0 * 4096.0))
        self.safe_move_pulse = self.max_move_pulse - int(safety_margin_pulse)
        if self.safe_move_pulse <= 0:
            raise ValueError("safe_move_pulse must be positive")

        self.normal_profile_velocity = int(normal_profile_velocity)
        self.auto_profile_velocity = int(auto_profile_velocity)
        self.profile_acceleration = int(profile_acceleration)
        self.goal_pwm_limit = int(goal_pwm_limit)
        self.pwm_limit = int(pwm_limit)
        self.grasp_target_load = float(grasp_target_load_percent)
        self.grasp_approach_step = max(1, int(grasp_approach_step))
        self.grasp_contact_step = max(1, int(grasp_contact_step))
        self.grasp_contact_load = float(grasp_contact_load)
        self.grasp_acceleration = int(grasp_acceleration)
        self.brownout_warn_v = float(brownout_warn_v)
        self.auto_load_threshold = float(auto_load_threshold_percent)
        self.auto_load_count_limit = int(auto_load_count_limit)
        self.auto_sample_sec = float(auto_sample_sec)
        self.hold_extra_pulse = int(hold_extra_pulse)
        self.min_grasp_travel = int(min_grasp_travel_pulse)
        self.auto_grasp_timeout = float(auto_grasp_timeout_sec)
        self.emergency_load = float(emergency_load_percent)
        self.temperature_stop_c = int(temperature_stop_c)
        self.position_tolerance = int(position_tolerance_pulse)
        self.move_timeout = float(move_timeout_sec)

        self.port_handler = None
        self.packet_handler = None
        self.open_position = None
        self.safe_close_position = None
        self.connected = False
        self._owns_port = False
        self._lock = threading.RLock()

    # ---------------------------------------------------------- communication
    @staticmethod
    def _signed32(value):
        return value - 0x100000000 if value >= 0x80000000 else value

    @staticmethod
    def _signed16(value):
        return value - 0x10000 if value >= 0x8000 else value

    def _check(self, result, error, label):
        if result != COMM_SUCCESS:
            raise RuntimeError(
                "%s communication error: %s"
                % (label, self.packet_handler.getTxRxResult(result))
            )
        if error:
            raise RuntimeError(
                "%s Dynamixel error: %s"
                % (label, self.packet_handler.getRxPacketError(error))
            )

    def _write1(self, address, value, label):
        with self._lock:
            result, error = self.packet_handler.write1ByteTxRx(
                self.port_handler, self.dxl_id, int(address), int(value)
            )
        self._check(result, error, label)

    def _write2(self, address, value, label):
        with self._lock:
            result, error = self.packet_handler.write2ByteTxRx(
                self.port_handler, self.dxl_id, int(address), int(value) & 0xFFFF
            )
        self._check(result, error, label)

    def _write4(self, address, value, label):
        with self._lock:
            result, error = self.packet_handler.write4ByteTxRx(
                self.port_handler, self.dxl_id, int(address), int(value) & 0xFFFFFFFF
            )
        self._check(result, error, label)

    def _read1(self, address, label):
        with self._lock:
            value, result, error = self.packet_handler.read1ByteTxRx(
                self.port_handler, self.dxl_id, int(address)
            )
        self._check(result, error, label)
        return int(value)

    def _read2(self, address, label, signed=False):
        with self._lock:
            value, result, error = self.packet_handler.read2ByteTxRx(
                self.port_handler, self.dxl_id, int(address)
            )
        self._check(result, error, label)
        return self._signed16(value) if signed else int(value)

    def _read4(self, address, label, signed=False):
        with self._lock:
            value, result, error = self.packet_handler.read4ByteTxRx(
                self.port_handler, self.dxl_id, int(address)
            )
        self._check(result, error, label)
        return self._signed32(value) if signed else int(value)

    # --------------------------------------------------------------- lifecycle
    def connect(self, capture_open_on_start=True, configured_open_position=506):
        with self._lock:
            if self.connected:
                return
            self.port_handler = PortHandler(self.port_name)
            self.packet_handler = PacketHandler(PROTOCOL_VERSION)
            if not self.port_handler.openPort():
                raise RuntimeError("centric gripper port open failed: " + self.port_name)
            self._owns_port = True
            try:
                if not self.port_handler.setBaudRate(self.baud):
                    raise RuntimeError("centric gripper baud-rate setting failed")
                self._configure_motor(
                    capture_open_on_start=capture_open_on_start,
                    configured_open_position=configured_open_position,
                    connection_label=self.port_name,
                )
            except Exception:
                try:
                    self.port_handler.closePort()
                finally:
                    self.port_handler = None
                    self.packet_handler = None
                    self._owns_port = False
                raise

    def connect_shared(
        self,
        port_handler,
        packet_handler,
        bus_lock,
        capture_open_on_start=True,
        configured_open_position=506,
    ):
        """Attach ID 8 to the U2D2 connection already opened by A-arm.

        The caller retains ownership of the physical port.  Every ID 8 Tx/Rx
        uses the same re-entrant lock as IDs 1~7, preventing overlapping SDK
        packets on the shared TTL bus.
        """
        if port_handler is None or packet_handler is None:
            raise ValueError("shared Dynamixel handlers are required")
        if bus_lock is None:
            raise ValueError("shared Dynamixel bus lock is required")
        self._lock = bus_lock
        with self._lock:
            if self.connected:
                return
            self.port_handler = port_handler
            self.packet_handler = packet_handler
            self._owns_port = False
            try:
                self._configure_motor(
                    capture_open_on_start=capture_open_on_start,
                    configured_open_position=configured_open_position,
                    connection_label="shared A-arm U2D2 (%s)" % self.port_name,
                )
            except Exception:
                self.port_handler = None
                self.packet_handler = None
                raise

    def _configure_motor(
        self,
        capture_open_on_start,
        configured_open_position,
        connection_label,
    ):
        model, result, error = self.packet_handler.ping(
            self.port_handler, self.dxl_id
        )
        self._check(result, error, "ping")

        self._write1(ADDR_TORQUE_ENABLE, TORQUE_DISABLE, "initial torque off")
        self._write1(
            ADDR_OPERATING_MODE,
            EXTENDED_POSITION_MODE,
            "extended position mode",
        )
        # ★ PWM Limit(주소 36)은 EEPROM 이라 torque off 인 지금만 써진다.
        #   주소 100(Goal PWM)은 PWM 제어 모드 전용이라 Extended Position
        #   에서는 무시된다. 885(기본)로 두면 집게가 물체에 닿는 순간
        #   최대 토크로 밀어붙여 전류가 튀고, 체인 끝 모터가 전압 강하로
        #   리셋된다(전원이 꺼진 것처럼 보인다).
        self._apply_pwm_limit()

        present = self.read_position()
        self.open_position = (
            present if bool(capture_open_on_start) else int(configured_open_position)
        )
        self.safe_close_position = (
            self.open_position + self.close_direction * self.safe_move_pulse
        )

        # Prevent a jump when torque is enabled.
        self._write4(ADDR_GOAL_POSITION, present, "initial goal position")
        self._write1(ADDR_TORQUE_ENABLE, TORQUE_ENABLE, "torque on")
        self.set_profile_velocity(self.normal_profile_velocity)
        self._write4(
            ADDR_PROFILE_ACCELERATION,
            self.profile_acceleration,
            "profile acceleration",
        )
        self._write2(ADDR_GOAL_PWM, self.goal_pwm_limit, "goal pwm")
        self.connected = True
        self._log(
            "connected: %s, id=%d, model=%d, OPEN=%d, SAFE_CLOSE=%d"
            % (
                connection_label,
                self.dxl_id,
                model,
                self.open_position,
                self.safe_close_position,
            )
        )

    def _apply_pwm_limit(self):
        """PWM Limit(주소 36)을 쓰고 되읽어 확인한다."""
        try:
            cur = self._read2(ADDR_PWM_LIMIT, "pwm limit")
        except Exception:
            cur = None
        target = max(0, min(885, int(self.pwm_limit)))
        if cur == target:
            self._log("PWM limit already %d (%.0f%%)" % (target, target / 885.0 * 100))
            return True
        self._write2(ADDR_PWM_LIMIT, target, "pwm limit")
        try:
            new = self._read2(ADDR_PWM_LIMIT, "pwm limit")
        except Exception:
            new = None
        if new != target:
            self._log(
                "PWM limit write failed (want %d, got %s). "
                "파지 전류가 제한되지 않아 모터가 리셋될 수 있습니다."
                % (target, new),
                "error",
            )
            return False
        self._log("PWM limit %s -> %d (%.0f%%)"
                  % (cur, target, target / 885.0 * 100))
        return True

    def close(self, torque_off=True):
        with self._lock:
            if self.port_handler is None:
                return
            try:
                if torque_off:
                    self._write1(ADDR_TORQUE_ENABLE, TORQUE_DISABLE, "torque off")
            except Exception as exc:
                self._log("shutdown warning: %s" % exc, warning=True)
            finally:
                if self._owns_port:
                    self.port_handler.closePort()
                self.port_handler = None
                self.packet_handler = None
                self.connected = False
                self._owns_port = False

    # ------------------------------------------------------------------ status
    def read_position(self):
        return self._read4(ADDR_PRESENT_POSITION, "present position", signed=True)

    def read_load_percent(self):
        return self._read2(ADDR_PRESENT_LOAD, "present load", signed=True) / 10.0

    def read_temperature_c(self):
        return self._read1(ADDR_PRESENT_TEMPERATURE, "present temperature")

    def read_voltage_v(self):
        """현재 공급 전압(V). 읽기에 실패하거나 규격 밖이면 None."""
        try:
            raw = self._read2(ADDR_PRESENT_INPUT_VOLTAGE, "present voltage")
        except Exception:
            return None
        if raw is None:
            return None
        volt = raw / 10.0
        return volt if 3.0 <= volt <= 30.0 else None

    def read_status(self):
        position = self.read_position()
        load = self.read_load_percent()
        return {
            "position": position,
            "close_percent": self.close_percent(position),
            "load_percent": load,
            "velocity_raw": self._read4(
                ADDR_PRESENT_VELOCITY, "present velocity", signed=True
            ),
            "pwm_percent": self._read2(
                ADDR_PRESENT_PWM, "present pwm", signed=True
            ) / 8.85,
            "voltage_v": self._read2(
                ADDR_PRESENT_INPUT_VOLTAGE, "present voltage"
            ) / 10.0,
            "temperature_c": self.read_temperature_c(),
            "moving": bool(self._read1(ADDR_MOVING, "moving")),
            "hardware_error": self._read1(
                ADDR_HARDWARE_ERROR_STATUS, "hardware error"
            ),
        }

    def close_percent(self, position=None):
        if self.open_position is None:
            return None
        if position is None:
            position = self.read_position()
        travelled = (int(position) - self.open_position) * self.close_direction
        return 100.0 * travelled / float(self.safe_move_pulse)

    # ---------------------------------------------------------------- movement
    def clamp_position(self, position):
        if self.open_position is None or self.safe_close_position is None:
            raise RuntimeError("centric gripper OPEN reference is not configured")
        low = min(self.open_position, self.safe_close_position)
        high = max(self.open_position, self.safe_close_position)
        return max(low, min(int(position), high))

    def set_profile_velocity(self, value):
        self._write4(ADDR_PROFILE_VELOCITY, int(value), "profile velocity")

    def set_goal_position(self, position):
        goal = self.clamp_position(position)
        self._write4(ADDR_GOAL_POSITION, goal, "goal position")
        return goal

    def hold_current_position(self):
        position = self.read_position()
        self.set_goal_position(position)
        return position

    def emergency_stop(self, reason="requested"):
        position = self.hold_current_position()
        self._log(
            "EMERGENCY STOP: %s; holding position %d" % (reason, position),
            warning=True,
        )

    def _wait_for_position(
        self,
        goal,
        timeout=None,
        stop_requested: Optional[Callable[[], bool]] = None,
    ):
        deadline = time.monotonic() + (
            self.move_timeout if timeout is None else float(timeout)
        )
        while time.monotonic() < deadline:
            if stop_requested is not None and stop_requested():
                self.emergency_stop("external stop")
                return False
            position = self.read_position()
            load = self.read_load_percent()
            temperature = self.read_temperature_c()
            if abs(load) >= self.emergency_load:
                self.emergency_stop("overload %.1f%%" % load)
                return False
            if temperature >= self.temperature_stop_c:
                self.emergency_stop("temperature %d C" % temperature)
                return False
            if abs(goal - position) <= self.position_tolerance:
                return True
            time.sleep(0.03)
        self.hold_current_position()
        self._log("position timeout; holding current position", warning=True)
        return False

    def open(self, stop_requested=None):
        self.set_profile_velocity(self.normal_profile_velocity)
        goal = self.set_goal_position(self.open_position)
        return self._wait_for_position(goal, stop_requested=stop_requested)

    def move_ratio(self, close_ratio, stop_requested=None):
        ratio = max(0.0, min(float(close_ratio), 1.0))
        goal = self.open_position + self.close_direction * int(
            round(self.safe_move_pulse * ratio)
        )
        self.set_profile_velocity(self.normal_profile_velocity)
        goal = self.set_goal_position(goal)
        return self._wait_for_position(goal, stop_requested=stop_requested)

    def auto_grasp(
        self, stop_requested: Optional[Callable[[], bool]] = None
    ) -> GraspResult:
        """
        부하가 grasp_target_load 에 닿을 때까지 조금씩 닫고 그 힘을 유지한다.

        ★ 예전에는 목표를 safe_close_position(최대 닫힘)으로 한 번에 던져
          놓고 부하만 관찰했다. 집게가 물체에 닿아도 위치 오차가 그대로
          크게 남으니 드라이버가 최대 PWM 을 밀어붙였고, 스톨 전류가 튀면서
          체인 끝의 모터가 전압 강하로 리셋됐다("조금 움직이다 전원이 꺼짐").
          단독 스크립트(centric_pick_test.py)가 되는 이유는 목표를 조금씩만
          앞세우기 때문이다. 그 방식을 그대로 옮겼다.

        지켜야 하는 것
          1) 목표는 '직전 목표' 기준으로 증분한다. '현재 위치' 기준으로 하면
             부하가 걸린 뒤 목표가 수렴해 힘이 더 안 올라간다.
          2) 목표 부하에 닿아도 목표를 현재 위치로 되돌리지 않는다.
             되돌리면 위치 오차가 0 이 되어 잡는 힘이 사라진다.
          3) 모터가 도는지는 부하가 아니라 위치 변화로 본다. 빈 공간을 닫는
             동안에는 부하가 0% 라서 부하만 보면 정상 동작을 오판한다.
        """
        cap = self.pwm_limit / 885.0 * 100.0
        if self.grasp_target_load >= cap:
            self._log(
                "목표 부하 %.0f%% 가 PWM 상한 %.0f%% 이상입니다. 도달할 수 없습니다."
                % (self.grasp_target_load, cap),
                "error",
            )
            return GraspResult(False, "target above pwm cap")

        start_position = self.read_position()
        self.set_profile_velocity(self.auto_profile_velocity)
        # 파지 중에는 가속도를 낮춘다. 전류 스파이크는 속도가 아니라 가속에서 나온다.
        self._write4(
            ADDR_PROFILE_ACCELERATION, self.grasp_acceleration, "grasp acceleration"
        )

        goal = start_position
        contact_step = self.grasp_contact_step
        hard_limit = self.grasp_target_load * 1.4
        last_position = start_position
        no_move = 0
        watch = 0
        v_min = None
        max_load = 0.0
        started = time.monotonic()

        self._log(
            "AUTO GRASP: target=%.1f%%, approach=%d/contact=%d pulse, "
            "pwm cap=%.0f%%, timeout=%.1fs"
            % (
                self.grasp_target_load,
                self.grasp_approach_step,
                contact_step,
                cap,
                self.auto_grasp_timeout,
            )
        )

        while True:
            if stop_requested is not None and stop_requested():
                position = self.hold_current_position()
                self._restore_acceleration()
                return GraspResult(False, "external stop", position)

            position = self.read_position()
            load = self.read_load_percent()
            temperature = self.read_temperature_c()
            abs_load = abs(load)
            max_load = max(max_load, abs_load)
            follow = abs(goal - position)
            elapsed = time.monotonic() - started

            if temperature >= self.temperature_stop_c:
                self.emergency_stop("temperature %d C" % temperature)
                self._restore_acceleration()
                return GraspResult(False, "overtemperature", position, load)
            if elapsed >= self.auto_grasp_timeout:
                self.hold_current_position()
                self._restore_acceleration()
                self._log("GRASP TIMEOUT (max load %.1f%%)" % max_load, "warning")
                return GraspResult(False, "timeout", position, load)

            # 전압과 토크 상태를 주기적으로 본다.
            # 토크가 저절로 0 이 되었다면 모터가 리셋된 것이다(브라운아웃).
            watch += 1
            if watch >= 5:
                watch = 0
                volt = self.read_voltage_v()
                if volt is not None:
                    v_min = volt if v_min is None else min(v_min, volt)
                    if volt < self.brownout_warn_v:
                        self._log(
                            "전압 %.1fV (부하 %.1f%%). 전압 강하 중입니다."
                            % (volt, abs_load),
                            "warning",
                        )
                te = self._read1(ADDR_TORQUE_ENABLE, "torque enable")
                if te == 0:
                    self._restore_acceleration()
                    self._log(
                        "★ 토크가 저절로 꺼졌습니다. ID%d 가 리셋되었습니다 "
                        "(전압 %s V, 부하 %.1f%%, 위치 %d)."
                        % (self.dxl_id, volt, abs_load, position),
                        "error",
                    )
                    self._log(
                        "  파지 전류로 공급 전압이 순간적으로 떨어졌습니다. "
                        "pwm_limit 을 더 낮추거나 이 모터의 체인 위치/커넥터를 "
                        "확인하세요.",
                        "error",
                    )
                    return GraspResult(False, "motor reset (brownout)", position, load)

            # 접촉 전에는 크게, 접촉 후에는 잘게. 게이트도 스텝에 비례시킨다.
            if abs_load < self.grasp_contact_load:
                step = self.grasp_approach_step
            elif abs_load < self.grasp_target_load * 0.5:
                step = contact_step
            elif abs_load < self.grasp_target_load * 0.8:
                step = max(2, contact_step // 2)
            else:
                step = max(1, contact_step // 4)
            advance_gate = max(step * 4, 80)

            # 너무 세게 물었다 -> 한 스텝 물러나고 증분을 절반으로
            if abs_load >= hard_limit:
                goal = self.clamp_position(goal - self.close_direction * contact_step)
                self.set_goal_position(goal)
                contact_step = max(1, contact_step // 2)
                self._log(
                    "load %.1f%% > limit %.1f%% -> back off, step=%d"
                    % (abs_load, hard_limit, contact_step),
                    "warning",
                )
                time.sleep(self.auto_sample_sec)
                continue

            # 목표 부하 도달 -> 목표를 그대로 두고 증분만 멈춘다(힘 유지)
            if abs_load >= self.grasp_target_load:
                self._restore_acceleration()
                self._log(
                    "GRASP SUCCESS: position=%d, load=%.1f%%, goal=%d 유지%s"
                    % (
                        position,
                        load,
                        goal,
                        "" if v_min is None else " (최저 전압 %.1fV)" % v_min,
                    )
                )
                return GraspResult(True, "contact detected", goal, load)

            # 모터가 도는지는 위치 변화로 판단한다
            moving = abs(position - last_position) >= 2
            last_position = position
            if moving or abs_load >= self.grasp_contact_load:
                no_move = 0
            else:
                no_move += 1
                if no_move >= 60:
                    self.hold_current_position()
                    self._restore_acceleration()
                    self._log(
                        "위치가 %d 에서 움직이지 않고 부하도 %.1f%% 입니다. "
                        "모터가 돌지 않습니다." % (position, abs_load),
                        "error",
                    )
                    return GraspResult(False, "motor not moving", position, load)

            if follow > advance_gate:
                # 위치가 따라오거나 부하가 오르기를 기다린다
                time.sleep(self.auto_sample_sec)
                continue

            close_progress = (position - self.open_position) * self.close_direction
            if close_progress >= self.safe_move_pulse - self.position_tolerance:
                self.hold_current_position()
                self._restore_acceleration()
                return GraspResult(False, "no object", position, load)

            goal = self.clamp_position(goal + self.close_direction * step)
            self.set_goal_position(goal)
            time.sleep(self.auto_sample_sec)

    def _restore_acceleration(self):
        try:
            self._write4(
                ADDR_PROFILE_ACCELERATION,
                self.profile_acceleration,
                "restore acceleration",
            )
        except Exception:
            pass

    def _log(self, text, warning=False):
        if self.logger is not None:
            method = self.logger.warning if warning else self.logger.info
            method(str(text))
        else:
            print("[CENTRIC] " + str(text))


def main(args=None):
    """Standalone ROS 2 wrapper for manual gripper tests."""
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    class CentricGripperNode(Node):
        def __init__(self):
            super().__init__("centric_gripper_5axis")
            d = self.declare_parameter
            d("port", "/dev/ttyUSB1")
            d("baud", 2_000_000)
            d("dxl_id", 8)
            d("capture_open_on_start", True)
            d("configured_open_position", 506)
            d("status_hz", 2.0)

            self.gripper = CentricGripper(
                port=str(self.get_parameter("port").value),
                baud=int(self.get_parameter("baud").value),
                dxl_id=int(self.get_parameter("dxl_id").value),
                logger=self.get_logger(),
            )
            self.gripper.connect(
                capture_open_on_start=bool(
                    self.get_parameter("capture_open_on_start").value
                ),
                configured_open_position=int(
                    self.get_parameter("configured_open_position").value
                ),
            )
            self.get_logger().warning(
                "현재 위치가 OPEN 기준입니다. 실행 전에 완전 개방 상태였는지 확인하세요."
            )
            self.publisher = self.create_publisher(
                String, "/centric_gripper/state", 10
            )
            self.create_subscription(
                String, "/centric_gripper/command", self._on_command, 10
            )
            hz = max(0.2, float(self.get_parameter("status_hz").value))
            self.create_timer(1.0 / hz, self._publish_status)
            self._busy_lock = threading.Lock()
            self._stop_event = threading.Event()

        def _on_command(self, msg):
            parts = msg.data.strip().lower().split()
            if not parts:
                return
            if parts[0] in ("stop", "s"):
                self._stop_event.set()
                self.gripper.emergency_stop("ROS command")
                return
            if not self._busy_lock.acquire(blocking=False):
                self.get_logger().warning("gripper is busy")
                return
            threading.Thread(
                target=self._run_command, args=(parts,), daemon=True
            ).start()

        def _run_command(self, parts):
            try:
                self._stop_event.clear()
                command = parts[0]
                if command in ("open", "o"):
                    self.gripper.open(self._stop_event.is_set)
                elif command in ("grasp", "g"):
                    result = self.gripper.auto_grasp(self._stop_event.is_set)
                    self.get_logger().info("grasp result: %s" % result)
                elif command in ("close", "ratio") and len(parts) == 2:
                    percent = float(parts[1])
                    if percent > 1.0:
                        percent /= 100.0
                    self.gripper.move_ratio(percent, self._stop_event.is_set)
                elif command == "status":
                    self.get_logger().info(str(self.gripper.read_status()))
                else:
                    self.get_logger().warning(
                        "commands: open | grasp | close <0..100> | status | stop"
                    )
            except Exception as exc:
                self.get_logger().error("centric command failed: %s" % exc)
            finally:
                self._busy_lock.release()

        def _publish_status(self):
            try:
                msg = String()
                msg.data = json.dumps(self.gripper.read_status(), ensure_ascii=False)
                self.publisher.publish(msg)
            except Exception as exc:
                self.get_logger().error("centric status read failed: %s" % exc)

        def destroy_node(self):
            self._stop_event.set()
            self.gripper.close(torque_off=True)
            super().destroy_node()

    rclpy.init(args=args)
    node = None
    try:
        node = CentricGripperNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

