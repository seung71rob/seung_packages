#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# robot_b_control_0519.py
# B robot arm controller for webcam-based multi-view inspection
# Motor order: 9 -> 10 -> 11 -> 12 -> 13 -> 14 -> 15 -> 16
# Joint order: q1 yaw, q2 pitch, q3 yaw, q4 roll, q5 yaw, q6 roll

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from dynamixel_sdk import PortHandler, PacketHandler, GroupSyncWrite, GroupSyncRead

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    ROS2_AVAILABLE = True
except Exception:
    rclpy = None
    Node = object
    String = None
    ROS2_AVAILABLE = False


ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_HARDWARE_ERROR_STATUS = 70
ADDR_PROFILE_ACCELERATION = 108
ADDR_PROFILE_VELOCITY = 112
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_CURRENT = 126
ADDR_PRESENT_VELOCITY = 128
ADDR_PRESENT_POSITION = 132
ADDR_PRESENT_INPUT_VOLTAGE = 144
ADDR_PRESENT_TEMPERATURE = 146

LEN_GOAL_POSITION = 4
LEN_PRESENT_POSITION = 4

TORQUE_ENABLE = 1
TORQUE_DISABLE = 0
OPERATING_MODE_EXTENDED_POSITION = 4
PROTOCOL_VERSION = 2.0
DXL_RESOLUTION = 4096


@dataclass
class DxlConfig:
    device_name: str = "/dev/ttyUSB1"
    baudrate: int = 2000000
    ids: List[int] = field(default_factory=lambda: [9, 10, 11, 12, 13, 14, 15, 16])
    position_tolerance_tick: int = 80

    # Motor-specific tolerance.
    # ID 10 is q2 with 21:1 reduction. A small joint error becomes hundreds of motor ticks,
    # so the old common 80 tick tolerance can cause false timeout even when the joint is practically reached.
    position_tolerance_tick_by_id: Dict[int, int] = field(default_factory=lambda: {
        9: 120,
        10: 700,
        11: 120,
        12: 120,
        13: 120,
        14: 160,
        15: 160,
        16: 120,
    })

    move_timeout_sec: float = 25.0
    pose_dwell_sec: float = 0.5
    joint_limits_deg: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "q1": (-360.0, 360.0),
        "q2": (-90.0, 90.0),
        "q3": (-120.0, 120.0),
        "q4": (-360.0, 360.0),
        "q5": (-120.0, 120.0),
        "q6": (-360.0, 360.0),
    })


@dataclass
class RobotCalibration:
    motor_zero_tick: Dict[int, int] = field(default_factory=lambda: {
        9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0
    })

    # Joint -> motor direction signs.
    # If a joint moves in the opposite direction, change the related sign.
    s_q1_m9: float = +1.0
    s_q2_m10: float = +1.0

    # q3 uses two motors. They should rotate in opposite directions.
    s_q3_m11: float = -1.0
    s_q3_m12: float = +1.0

    s_q4_m13: float = -1.0

    # q5 uses two direct motors. They should rotate in opposite directions.
    s_q5_m14: float = -1.0
    s_q5_m15: float = +1.0

    s_q6_m16: float = +1.0

    # Gear ratios: motor angle = joint angle * ratio
    r_q1: float = 40.0 / 14.0
    r_q2: float = 21.0
    r_q3: float = 72.0 / 15.0
    r_q4: float = 30.0 / 15.0
    r_q5: float = 1.0
    r_q6: float = 1.0


@dataclass
class PoseLibrary:
    B_HOME: Dict[str, float] = field(default_factory=lambda: {
        "q1": 90.0, "q2": 0.0, "q3": 0.0, "q4": 0.0, "q5": 0.0, "q6": 0.0
    })
    B_BOTTOM_VIEW: Dict[str, float] = field(default_factory=lambda: {
        "q1": 0.0, "q2": 0.0, "q3": 15.0, "q4": 0.0, "q5": -85.0, "q6": 180.0
    })
    B_FLASH_VIEW: Dict[str, float] = field(default_factory=lambda: {
        "q1": 0.0, "q2": 0.0, "q3": 50.0, "q4": 0.0, "q5": -82.0, "q6": 180.0
    })
    B_BACK_VIEW: Dict[str, float] = field(default_factory=lambda: {
        "q1": 0.0, "q2": 0.0, "q3": 25.0, "q4": 0.0, "q5": -60.0, "q6": 180.0
    })


@dataclass
class InspectionState:
    has_car_part: bool
    has_defect: bool
    defect_count: int
    judge: str
    stamp: float


def deg_to_tick(deg: float) -> int:
    return int(round((deg / 360.0) * DXL_RESOLUTION))


def tick_to_deg(tick: int) -> float:
    return (tick / DXL_RESOLUTION) * 360.0


def uint32_from_int(value: int) -> int:
    return value & 0xFFFFFFFF


def dxl_signed32(value: int) -> int:
    if value & 0x80000000:
        value = -((~value + 1) & 0xFFFFFFFF)
    return value


def dxl_signed16(value: int) -> int:
    if value & 0x8000:
        value = -((~value + 1) & 0xFFFF)
    return value


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class JointMotorMapperB:
    def __init__(self, calib: RobotCalibration):
        self.c = calib

    def joint_deg_to_motor_deg(self, q: Dict[str, float]) -> Dict[int, float]:
        return {
            9: self.c.s_q1_m9 * q["q1"] * self.c.r_q1,
            10: self.c.s_q2_m10 * q["q2"] * self.c.r_q2,
            11: self.c.s_q3_m11 * q["q3"] * self.c.r_q3,
            12: self.c.s_q3_m12 * q["q3"] * self.c.r_q3,
            13: self.c.s_q4_m13 * q["q4"] * self.c.r_q4,
            14: self.c.s_q5_m14 * q["q5"] * self.c.r_q5,
            15: self.c.s_q5_m15 * q["q5"] * self.c.r_q5,
            16: self.c.s_q6_m16 * q["q6"] * self.c.r_q6,
        }

    def joint_deg_to_goal_tick(self, q: Dict[str, float]) -> Dict[int, int]:
        motor_deg = self.joint_deg_to_motor_deg(q)
        return {
            dxl_id: self.c.motor_zero_tick[dxl_id] + deg_to_tick(mdeg)
            for dxl_id, mdeg in motor_deg.items()
        }

    def present_tick_to_joint_deg(self, present_tick: Dict[int, int]) -> Dict[str, float]:
        m9 = tick_to_deg(present_tick[9] - self.c.motor_zero_tick[9])
        m10 = tick_to_deg(present_tick[10] - self.c.motor_zero_tick[10])
        m11 = tick_to_deg(present_tick[11] - self.c.motor_zero_tick[11])
        m12 = tick_to_deg(present_tick[12] - self.c.motor_zero_tick[12])
        m13 = tick_to_deg(present_tick[13] - self.c.motor_zero_tick[13])
        m14 = tick_to_deg(present_tick[14] - self.c.motor_zero_tick[14])
        m15 = tick_to_deg(present_tick[15] - self.c.motor_zero_tick[15])
        m16 = tick_to_deg(present_tick[16] - self.c.motor_zero_tick[16])

        q1 = m9 / (self.c.s_q1_m9 * self.c.r_q1)
        q2 = m10 / (self.c.s_q2_m10 * self.c.r_q2)

        q3_from_11 = m11 / (self.c.s_q3_m11 * self.c.r_q3)
        q3_from_12 = m12 / (self.c.s_q3_m12 * self.c.r_q3)
        q3 = 0.5 * (q3_from_11 + q3_from_12)

        q4 = m13 / (self.c.s_q4_m13 * self.c.r_q4)

        q5_from_14 = m14 / (self.c.s_q5_m14 * self.c.r_q5)
        q5_from_15 = m15 / (self.c.s_q5_m15 * self.c.r_q5)
        q5 = 0.5 * (q5_from_14 + q5_from_15)

        q6 = m16 / (self.c.s_q6_m16 * self.c.r_q6)

        return {
            "q1": q1, "q2": q2, "q3": q3,
            "q4": q4, "q5": q5, "q6": q6
        }


class DynamixelArmB:
    def __init__(self, cfg: DxlConfig, calib: RobotCalibration):
        self.cfg = cfg
        self.calib = calib
        self.mapper = JointMotorMapperB(calib)
        self.port_handler = PortHandler(cfg.device_name)
        self.packet_handler = PacketHandler(PROTOCOL_VERSION)
        self.sync_write_goal = GroupSyncWrite(
            self.port_handler,
            self.packet_handler,
            ADDR_GOAL_POSITION,
            LEN_GOAL_POSITION
        )
        self.sync_read_present = GroupSyncRead(
            self.port_handler,
            self.packet_handler,
            ADDR_PRESENT_POSITION,
            LEN_PRESENT_POSITION
        )
        self._opened = False

    def _check_comm(self, dxl_comm_result, dxl_error, context=""):
        if dxl_comm_result != 0:
            raise RuntimeError(f"[{context}] communication failed: {self.packet_handler.getTxRxResult(dxl_comm_result)}")
        if dxl_error != 0:
            raise RuntimeError(f"[{context}] dxl error: {self.packet_handler.getRxPacketError(dxl_error)}")

    def open(self):
        if not self.port_handler.openPort():
            raise RuntimeError(f"port open failed: {self.cfg.device_name}")
        if not self.port_handler.setBaudRate(self.cfg.baudrate):
            raise RuntimeError(f"baudrate set failed: {self.cfg.baudrate}")
        self._opened = True

        self.sync_read_present.clearParam()
        for dxl_id in self.cfg.ids:
            ok = self.sync_read_present.addParam(dxl_id)
            if not ok:
                raise RuntimeError(f"SyncRead addParam failed: ID {dxl_id}")

        print(f"[OK] opened {self.cfg.device_name}, baudrate={self.cfg.baudrate}")

    def close(self):
        if self._opened:
            self.port_handler.closePort()
            self._opened = False
            print("[OK] port closed")

    def write1(self, dxl_id, addr, value, context="write1"):
        comm, err = self.packet_handler.write1ByteTxRx(self.port_handler, dxl_id, addr, value)
        self._check_comm(comm, err, context)

    def write2(self, dxl_id, addr, value, context="write2"):
        comm, err = self.packet_handler.write2ByteTxRx(self.port_handler, dxl_id, addr, value)
        self._check_comm(comm, err, context)

    def write4(self, dxl_id, addr, value, context="write4"):
        comm, err = self.packet_handler.write4ByteTxRx(self.port_handler, dxl_id, addr, value)
        self._check_comm(comm, err, context)

    def read1(self, dxl_id, addr, context="read1"):
        val, comm, err = self.packet_handler.read1ByteTxRx(self.port_handler, dxl_id, addr)
        self._check_comm(comm, err, context)
        return val

    def read2(self, dxl_id, addr, context="read2"):
        val, comm, err = self.packet_handler.read2ByteTxRx(self.port_handler, dxl_id, addr)
        self._check_comm(comm, err, context)
        return val

    def read_present_current_raw(self, dxl_id):
        val = self.read2(dxl_id, ADDR_PRESENT_CURRENT, f"read_current_id{dxl_id}")
        return dxl_signed16(val)

    def read_present_current_ma(self, dxl_id):
        # XL430 Present Current unit is approximately 2.69 mA per raw unit.
        # If your model differs, use the raw value for comparison.
        return self.read_present_current_raw(dxl_id) * 2.69

    def read4(self, dxl_id, addr, context="read4"):
        val, comm, err = self.packet_handler.read4ByteTxRx(self.port_handler, dxl_id, addr)
        self._check_comm(comm, err, context)
        return dxl_signed32(val)

    def disable_torque_all(self):
        for dxl_id in self.cfg.ids:
            try:
                self.write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE, f"disable_torque_id{dxl_id}")
            except Exception as e:
                print(f"[WARN] torque disable failed ID {dxl_id}: {e}")

    def enable_torque_all(self):
        for dxl_id in self.cfg.ids:
            self.write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE, f"enable_torque_id{dxl_id}")

    def set_profiles(self):
        # Conservative initial speed.
        # q2 uses 21:1 cycloidal reduction, so motor 10 velocity is allowed higher.
        vel_map = {
            9: 60,
            10: 190,
            11: 90,
            12: 90,
            13: 80,
            14: 25,
            15: 25,
            16: 30,
        }
        acc_map = {
            9: 4,
            10: 5,
            11: 4,
            12: 4,
            13: 4,
            14: 3,
            15: 3,
            16: 3,
        }
        for dxl_id in self.cfg.ids:
            self.write4(dxl_id, ADDR_PROFILE_VELOCITY, vel_map[dxl_id], f"velocity_id{dxl_id}")
            self.write4(dxl_id, ADDR_PROFILE_ACCELERATION, acc_map[dxl_id], f"acceleration_id{dxl_id}")
        print("[OK] velocity/acceleration profiles applied")

    def set_extended_position_mode_all(self):
        self.disable_torque_all()
        for dxl_id in self.cfg.ids:
            self.write1(dxl_id, ADDR_OPERATING_MODE, OPERATING_MODE_EXTENDED_POSITION, f"set_mode_id{dxl_id}")
        self.enable_torque_all()
        self.set_profiles()
        print("[OK] Extended Position Mode enabled")

    def ping_all(self):
        print("\n[PING]")
        for dxl_id in self.cfg.ids:
            model_number, comm, err = self.packet_handler.ping(self.port_handler, dxl_id)
            if comm == 0 and err == 0:
                print(f"  ID {dxl_id}: OK, model={model_number}")
            else:
                print(f"  ID {dxl_id}: FAIL, {self.packet_handler.getTxRxResult(comm)}, err={err}")

    def read_present_ticks_individual(self, retry_per_id=3, retry_dt=0.04):
        out = {}
        failed = []
        for dxl_id in self.cfg.ids:
            ok = False
            last_error = None
            for _ in range(retry_per_id):
                try:
                    out[dxl_id] = self.read4(dxl_id, ADDR_PRESENT_POSITION, f"present_id{dxl_id}")
                    ok = True
                    break
                except Exception as e:
                    last_error = e
                    time.sleep(retry_dt)
            if not ok:
                failed.append((dxl_id, last_error))

        if failed:
            msg = ", ".join([f"ID {dxl_id}: {err}" for dxl_id, err in failed])
            raise RuntimeError(f"individual present read failed: {msg}")
        return out

    def read_present_ticks(self, retry=6, retry_dt=0.06, use_fallback=True):
        last_comm = None
        for attempt in range(1, retry + 1):
            comm = self.sync_read_present.txRxPacket()
            last_comm = comm
            if comm == 0:
                out = {}
                ok = True
                for dxl_id in self.cfg.ids:
                    if not self.sync_read_present.isAvailable(dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION):
                        ok = False
                        break
                    raw = self.sync_read_present.getData(dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
                    out[dxl_id] = dxl_signed32(raw)
                if ok:
                    return out

            print(f"[WARN] SyncRead retry {attempt}/{retry}: {self.packet_handler.getTxRxResult(comm)}")
            time.sleep(retry_dt)

        if use_fallback:
            print("[WARN] SyncRead failed -> individual read fallback")
            return self.read_present_ticks_individual()

        raise RuntimeError(f"SyncRead failed: {self.packet_handler.getTxRxResult(last_comm)}")

    def write_goal_ticks(self, goal_tick: Dict[int, int]):
        self.sync_write_goal.clearParam()
        for dxl_id in self.cfg.ids:
            raw = uint32_from_int(goal_tick[dxl_id])
            param = [
                raw & 0xFF,
                (raw >> 8) & 0xFF,
                (raw >> 16) & 0xFF,
                (raw >> 24) & 0xFF
            ]
            ok = self.sync_write_goal.addParam(dxl_id, bytes(param))
            if not ok:
                raise RuntimeError(f"SyncWrite addParam failed: ID {dxl_id}")

        comm = self.sync_write_goal.txPacket()
        if comm != 0:
            raise RuntimeError(f"SyncWrite failed: {self.packet_handler.getTxRxResult(comm)}")

    def print_present_state(self):
        ticks = self.read_present_ticks()
        joints = self.mapper.present_tick_to_joint_deg(ticks)

        print("\n[PRESENT MOTOR TICKS]")
        for dxl_id in self.cfg.ids:
            print(f"  ID {dxl_id}: {ticks[dxl_id]}")

        print("[PRESENT JOINT DEG]")
        for k in ["q1", "q2", "q3", "q4", "q5", "q6"]:
            print(f"  {k}: {joints[k]: .3f}")

    def print_device_status(self):
        print("\n[DEVICE STATUS]")
        for dxl_id in self.cfg.ids:
            try:
                torque = self.read1(dxl_id, ADDR_TORQUE_ENABLE, f"read_torque_id{dxl_id}")
                hwerr = self.read1(dxl_id, ADDR_HARDWARE_ERROR_STATUS, f"read_hwerr_id{dxl_id}")
                volt_raw = self.read2(dxl_id, ADDR_PRESENT_INPUT_VOLTAGE, f"read_voltage_id{dxl_id}")
                current_raw = self.read_present_current_raw(dxl_id)
                current_ma = current_raw * 2.69
                temp = self.read1(dxl_id, ADDR_PRESENT_TEMPERATURE, f"read_temp_id{dxl_id}")
                vel = self.read4(dxl_id, ADDR_PRESENT_VELOCITY, f"read_velocity_id{dxl_id}")
                print(
                    f"  ID {dxl_id}: torque={torque}, hwerr={hwerr}, "
                    f"voltage={volt_raw/10.0:.1f}V, current={current_ma:.1f}mA(raw={current_raw}), "
                    f"temp={temp}C, vel={vel}"
                )
            except Exception as e:
                print(f"  ID {dxl_id}: status read failed: {e}")

    def set_current_as_zero(self):
        ticks = self.read_present_ticks()
        self.calib.motor_zero_tick = dict(ticks)
        print("[INFO] current motor ticks saved as zero:")
        print(self.calib.motor_zero_tick)

    def _validate_joint_limits(self, q: Dict[str, float]):
        for key in ["q1", "q2", "q3", "q4", "q5", "q6"]:
            lo, hi = self.cfg.joint_limits_deg[key]
            if not (lo <= q[key] <= hi):
                raise ValueError(f"{key}={q[key]:.2f} deg is outside limit [{lo}, {hi}]")

    def move_joint_deg(self, q: Dict[str, float], wait=True):
        self._validate_joint_limits(q)
        goal_tick = self.mapper.joint_deg_to_goal_tick(q)
        motor_deg = self.mapper.joint_deg_to_motor_deg(q)

        print("\n[MOVE] joint target(deg):", q)
        print("[MOVE] motor target(deg):", {k: round(v, 3) for k, v in motor_deg.items()})
        print("[MOVE] goal tick:", goal_tick)

        self.write_goal_ticks(goal_tick)

        if not wait:
            return

        start = time.time()
        last_present = None
        while True:
            present = self.read_present_ticks()
            last_present = present

            done = True
            max_err_id = None
            max_err_tick = -1
            for dxl_id in self.cfg.ids:
                err_tick = abs(present[dxl_id] - goal_tick[dxl_id])
                tol_tick = self.cfg.position_tolerance_tick_by_id.get(
                    dxl_id,
                    self.cfg.position_tolerance_tick
                )
                if err_tick > max_err_tick:
                    max_err_tick = err_tick
                    max_err_id = dxl_id
                if err_tick > tol_tick:
                    done = False

            if done:
                print(f"[MOVE] done (max_err: ID {max_err_id} = {max_err_tick} tick)")
                return

            if time.time() - start > self.cfg.move_timeout_sec:
                print("[TIMEOUT] goal:", goal_tick)
                print("[TIMEOUT] present:", last_present)
                self.print_device_status()
                raise TimeoutError("move_joint_deg timeout")

            time.sleep(0.10)


class BInspectionSubscriber(Node):
    def __init__(
        self,
        inspection_topic: str,
        sync_command_topic: str,
        view_result_topic: str,
        b_status_topic: str,
        camera_control_topic: str,
        state_holder: Dict[str, object],
        lock: threading.Lock,
    ):
        super().__init__("b_inspection_listener")
        self._state_holder = state_holder
        self._lock = lock

        self.inspection_subscription = self.create_subscription(
            String, inspection_topic, self.inspection_callback, 10
        )
        self.sync_command_subscription = self.create_subscription(
            String, sync_command_topic, self.sync_command_callback, 10
        )
        self.view_result_subscription = self.create_subscription(
            String, view_result_topic, self.view_result_callback, 10
        )
        self.b_status_pub = self.create_publisher(String, b_status_topic, 10)
        self.camera_control_pub = self.create_publisher(
            String, camera_control_topic, 10
        )

        self.get_logger().info(f"Subscribed to {inspection_topic}")
        self.get_logger().info(f"Subscribed to {sync_command_topic}")
        self.get_logger().info(f"Subscribed to {view_result_topic}")
        self.get_logger().info(f"B status topic: {b_status_topic}")
        self.get_logger().info(f"Camera control topic: {camera_control_topic}")

    def inspection_callback(self, msg):
        try:
            data = json.loads(msg.data)
            state = InspectionState(
                has_car_part=bool(data.get("has_car_part", False)),
                has_defect=bool(data.get("has_defect", False)),
                defect_count=int(data.get("defect_count", 0)),
                judge=str(data.get("judge", "NO_OBJECT")),
                stamp=time.time()
            )
            with self._lock:
                self._state_holder["inspection"] = state
        except Exception as e:
            self.get_logger().warning(f"inspection parse failed: {e} / raw={msg.data}")

    def sync_command_callback(self, msg):
        try:
            data = json.loads(msg.data)
            if not isinstance(data, dict):
                raise ValueError("sync command must be a JSON object")
            data["command"] = str(data.get("command", "")).strip().upper()
            with self._lock:
                self._state_holder["sync_commands"].append(data)
        except Exception as e:
            self.get_logger().warning(
                f"sync command parse failed: {e} / raw={msg.data}"
            )

    def view_result_callback(self, msg):
        try:
            data = json.loads(msg.data)
            if not isinstance(data, dict):
                raise ValueError("view result must be a JSON object")
            with self._lock:
                self._state_holder["view_results"].append(data)
        except Exception as e:
            self.get_logger().warning(
                f"view result parse failed: {e} / raw={msg.data}"
            )

    def drain_outbound(self):
        while True:
            with self._lock:
                if not self._state_holder["outbound"]:
                    return
                channel, payload = self._state_holder["outbound"].popleft()

            msg = String()
            msg.data = json.dumps(payload, ensure_ascii=False)
            if channel == "b_status":
                self.b_status_pub.publish(msg)
            elif channel == "camera_control":
                self.camera_control_pub.publish(msg)


class BInspectionROSBridge:
    def __init__(
        self,
        inspection_topic: str = "/inspection_state_side",
        sync_command_topic: str = "/multi_arm/inspection_command",
        view_result_topic: str = "/baseplate_side/view_result",
        b_status_topic: str = "/multi_arm/b_arm_status",
        camera_control_topic: str = "/baseplate_side/inspection_control",
    ):
        self.inspection_topic = inspection_topic
        self.sync_command_topic = sync_command_topic
        self.view_result_topic = view_result_topic
        self.b_status_topic = b_status_topic
        self.camera_control_topic = camera_control_topic
        self._thread = None
        self._lock = threading.Lock()
        self._state_holder: Dict[str, object] = {
            "inspection": None,
            "sync_commands": deque(),
            "view_results": deque(),
            "outbound": deque(),
        }
        self._stop_evt = threading.Event()
        self._started = False

    def start(self):
        if not ROS2_AVAILABLE:
            print("[WARN] ROS2 Python environment not found. Source ROS2 setup files first.")
            return False
        if self._started:
            print("[OK] ROS2 subscriber already running")
            return True

        self._thread = threading.Thread(target=self._spin_thread, daemon=True)
        self._thread.start()
        self._started = True
        time.sleep(0.3)
        return True

    def _spin_thread(self):
        rclpy.init(args=None)
        node = BInspectionSubscriber(
            self.inspection_topic,
            self.sync_command_topic,
            self.view_result_topic,
            self.b_status_topic,
            self.camera_control_topic,
            self._state_holder,
            self._lock,
        )
        try:
            while rclpy.ok() and not self._stop_evt.is_set():
                rclpy.spin_once(node, timeout_sec=0.2)
                node.drain_outbound()
        finally:
            node.destroy_node()
            rclpy.shutdown()

    def stop(self):
        if not self._started:
            return
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._started = False

    def get_inspection(self) -> Optional[InspectionState]:
        with self._lock:
            return self._state_holder.get("inspection")

    def clear_sync_queues(self):
        with self._lock:
            self._state_holder["sync_commands"].clear()
            self._state_holder["view_results"].clear()

    def wait_sync_command(self, timeout_sec=None):
        start = time.time()
        while self._started:
            with self._lock:
                if self._state_holder["sync_commands"]:
                    return self._state_holder["sync_commands"].popleft()
            if timeout_sec is not None and time.time() - start >= timeout_sec:
                return None
            time.sleep(0.05)
        return None

    def wait_view_result(self, expected_view, expected_cycle_id, timeout_sec=8.0, request_id=''):
        start = time.monotonic()
        while self._started and time.monotonic()-start < timeout_sec:
            with self._lock:
                # Let the outer engine process ABORT without waiting eight seconds.
                if any(c.get('command') == 'ABORT' for c in self._state_holder['sync_commands']):
                    return None
                queue = self._state_holder['view_results']
                for _ in range(len(queue)):
                    result = queue.popleft()
                    if (result.get('view') == expected_view and
                        result.get('cycle_id') == expected_cycle_id and
                        result.get('request_id') == request_id):
                        return result
                    # Stale view results cannot become valid for a future request.
            time.sleep(0.02)
        return None


    def _queue_outbound(self, channel, payload):
        with self._lock:
            self._state_holder["outbound"].append((channel, payload))

    def publish_b_status(self, state, pose="", cycle_id="", **extra):
        payload = {
            "robot": "b_arm",
            "state": state,
            "pose": pose,
            "cycle_id": cycle_id,
            "stamp_sec": time.time(),
        }
        payload.update(extra)
        self._queue_outbound("b_status", payload)
        print(f"[SYNC STATUS] {payload}")

    def start_camera_view_inspection(self, view, cycle_id, request_id=''):
        self._queue_outbound('camera_control', {
            'command':'START', 'view':view, 'cycle_id':cycle_id,
            'request_id':request_id, 'inspection_duration_sec':3.0,
            'required_defect_sec':2.0})



class BArmController:
    def __init__(self, arm: DynamixelArmB, poses: PoseLibrary, cfg: DxlConfig, ros_bridge: BInspectionROSBridge):
        self.arm = arm
        self.poses = poses
        self.cfg = cfg
        self.ros_bridge = ros_bridge

    def move_pose(self, pose_name: str, dwell=None):
        if dwell is None:
            dwell = self.cfg.pose_dwell_sec

        pose = getattr(self.poses, pose_name)
        print(f"\n[POSE] {pose_name}")
        self.arm.move_joint_deg(pose)
        time.sleep(dwell)

    def move_manual(self, q1, q2, q3, q4, q5, q6):
        q = {
            "q1": q1, "q2": q2, "q3": q3,
            "q4": q4, "q5": q5, "q6": q6
        }
        self.arm.move_joint_deg(q)

    def preview_inspection_state(self):
        state = self.ros_bridge.get_inspection()
        if state is None:
            print("[INSPECT] no /b_inspection_state message received yet")
            return None

        age = time.time() - state.stamp
        print("\n[LATEST B INSPECTION]")
        print(f"  has_car_part = {state.has_car_part}")
        print(f"  has_defect   = {state.has_defect}")
        print(f"  defect_count = {state.defect_count}")
        print(f"  judge        = {state.judge}")
        print(f"  age          = {age:.2f} sec")
        return state

    def inspect_current_view(self, view_name: str, hold_sec=3.0, required_defect_sec=2.0,
                             sample_dt=0.05, max_state_age_sec=0.5):
        print(f"\n[INSPECT] {view_name}: observe for {hold_sec:.1f} sec")

        if not self.ros_bridge._started:
            print("[ROS] starting /b_inspection_state subscriber")
            self.ros_bridge.start()
            time.sleep(0.5)

        start = time.time()
        last_t = start
        defect_accum_sec = 0.0
        max_defect_count = 0
        sample_count = 0
        valid_sample_count = 0

        while True:
            now = time.time()
            if now - start >= hold_sec:
                break

            dt = now - last_t
            last_t = now
            sample_count += 1

            state = self.ros_bridge.get_inspection()
            if state is not None and (now - state.stamp) <= max_state_age_sec:
                valid_sample_count += 1
                max_defect_count = max(max_defect_count, state.defect_count)
                if state.has_defect:
                    defect_accum_sec += dt

            time.sleep(sample_dt)

        detected = defect_accum_sec >= required_defect_sec
        result = {
            "view": view_name,
            "detected": detected,
            "judge": "NG" if detected else "OK",
            "defect_accum_sec": round(defect_accum_sec, 3),
            "hold_sec": hold_sec,
            "required_defect_sec": required_defect_sec,
            "max_defect_count": max_defect_count,
            "sample_count": sample_count,
            "valid_sample_count": valid_sample_count,
        }

        print(
            f"[INSPECT RESULT] {view_name}: "
            f"defect_time={defect_accum_sec:.2f}/{hold_sec:.2f}s, "
            f"threshold={required_defect_sec:.2f}s -> {result['judge']}"
        )
        return result

    def inspect_view_through_camera(self, view_name, cycle_id, timeout_sec=8.0, request_id=''):
        self.ros_bridge.start_camera_view_inspection(view_name, cycle_id, request_id)
        self.ros_bridge.publish_b_status('INSPECTION_STARTED', pose=view_name,
                                         cycle_id=cycle_id, request_id=request_id)
        return self.ros_bridge.wait_view_result(view_name, cycle_id, timeout_sec,
                                                request_id=request_id)


    def run_synchronized_two_view_inspection(self):
        """Menu 12: persistent BOTTOM + BACK(8) + FLASH(8) engine.

        Camera START is sent only after A reports arrival and B has reached its
        pose. FINISH and NG ABORT(return_home=True) wait for measured B_HOME.
        Ordinary ABORT holds current position, stops camera, retains suction
        control on A. No automatic recovery motion on a hardware exception.
        """
        if not self.ros_bridge._started and not self.ros_bridge.start(): return
        self.ros_bridge.clear_sync_queues()
        self.move_pose('B_HOME', dwell=0.5)
        current_pose = 'B_HOME'
        active_cycle = ''
        results = {}
        required = {'B_BOTTOM_VIEW'} | {'B_%s_VIEW_%02d'%(v,i)
                    for v in ('BACK','FLASH') for i in range(1,9)}
        print('[SYNC] Ready for repeated 17-view cycles. Ctrl+C to exit.')
        while self.ros_bridge._started:
            data = self.ros_bridge.wait_sync_command(timeout_sec=1.0)
            if data is None:
                if current_pose == 'B_HOME':
                    self.ros_bridge.publish_b_status('WAITING_FOR_GO_BOTTOM', pose=current_pose)
                continue
            cmd = str(data.get('command','')).upper()
            cycle = str(data.get('cycle_id',''))
            rid = str(data.get('request_id',''))
            def status(state, **extra):
                self.ros_bridge.publish_b_status(state, pose=current_pose,
                                                cycle_id=cycle, request_id=rid, **extra)
            def reject(reason): status('COMMAND_REJECTED', reason=reason)
            if not cycle or not rid:
                reject('cycle_id and request_id required'); continue
            if active_cycle and cycle != active_cycle:
                reject('different active cycle'); continue
            try:
                if cmd == 'ABORT':
                    self.ros_bridge._queue_outbound('camera_control', {'command':'STOP'})
                    if data.get('return_home', False):
                        self.move_pose('B_HOME', dwell=0.5)
                        current_pose = 'B_HOME'
                        status('ABORTED_HOME')
                    else:
                        # SyncWrite current ticks holds the measured position.
                        self.arm.write_goal_ticks(self.arm.read_present_ticks())
                        status('SEQUENCE_ABORTED')
                    results.clear(); active_cycle = ''
                    continue
                if cmd == 'GO_HOME':
                    self.move_pose('B_HOME', dwell=0.5)
                    current_pose = 'B_HOME'; results.clear(); active_cycle = ''
                    status('ARRIVED'); continue
                if cmd == 'CHECK_READY':
                    if current_pose != 'B_HOME':
                        reject('B_HOME required before new cycle'); continue
                    if not active_cycle:
                        results.clear(); active_cycle = cycle
                    status('READY'); continue
                if not active_cycle:
                    reject('CHECK_READY required first'); continue
                if not data.get('a_arrived', False):
                    reject('A arm arrival acknowledgment required'); continue
                if cmd in ('GO_BOTTOM','GO_BACK','GO_FLASH'):
                    view = cmd[3:]
                    # NG 가 하나라도 있으면 남은 검사를 하지 않는다.
                    # 이전 단계가 전부 OK 일 때만 다음 자세로 넘어간다.
                    if view == 'BACK' and results.get('B_BOTTOM_VIEW',{}).get('judge') != 'OK':
                        reject('bottom OK required'); continue
                    if view == 'FLASH' and not all(
                            results.get('B_BACK_VIEW_%02d'%i,{}).get('judge') == 'OK'
                            for i in range(1,9)):
                        reject('eight back OK results required'); continue
                    target = 'B_'+view+'_VIEW'
                    status('MOVING', target_pose=target)
                    self.move_pose(target, dwell=0.5)
                    current_pose = target
                    status('ARRIVED')
                elif cmd in ('INSPECT_BOTTOM','INSPECT_BACK','INSPECT_FLASH'):
                    view = cmd[8:]
                    if current_pose != 'B_'+view+'_VIEW':
                        reject('wrong B pose'); continue
                    index = int(data.get('repeat_index',1))
                    if not 1 <= index <= (1 if view == 'BOTTOM' else 8):
                        reject('repeat_index out of range'); continue
                    key = 'B_BOTTOM_VIEW' if view == 'BOTTOM' else 'B_%s_VIEW_%02d'%(view,index)
                    if data.get('view') != key:
                        reject('view does not match command/repeat'); continue
                    if key in results:
                        reject('view already inspected'); continue
                    if view != 'BOTTOM' and any('B_%s_VIEW_%02d'%(view,i) not in results for i in range(1,index)):
                        reject('previous repeat missing'); continue
                    result = self.inspect_view_through_camera(key, cycle, request_id=rid)
                    if result is None:
                        status('INSPECTION_TIMEOUT'); continue
                    results[key] = result
                    if result.get('judge') not in ('OK','NG'):
                        status('INSPECTION_ERROR', reason=result.get('reason','incomplete'), result=result)
                    else:
                        status('INSPECTION_COMPLETE', view=key, result=result)
                        if result.get('judge') == 'NG':
                            # NG 면 여기서 끝이다. 게이트가 다음 GO_* 를 막으므로
                            # 남은 뷰는 검사되지 않는다. 홈 복귀는 A arm 이 보내는
                            # ABORT(return_home) 가 처리한다(중복 이동 방지).
                            print(f'[SYNC] {key} NG -> 남은 검사 중단. A arm 의 ABORT 대기.')
                elif cmd == 'FINISH':
                    # NG 는 발견 즉시 SEQUENCE_ABORTED 로 닫히므로,
                    # 여기까지 오면 17개가 모두 OK 인 경우다.
                    if set(results) != required or any(r.get('judge') != 'OK'
                                                       for r in results.values()):
                        reject('all 17 explicit OK results required'); continue
                    self.move_pose('B_HOME', dwell=0.5)
                    current_pose = 'B_HOME'
                    status('SEQUENCE_COMPLETE', final_judgement='OK', result_count=17)
                    active_cycle = ''; results.clear()
                else:
                    reject('unknown command')
            except Exception as exc:
                try: self.arm.write_goal_ticks(self.arm.read_present_ticks())
                except Exception: pass
                status('MOTION_ERROR', reason=str(exc))



def build_robot():
    cfg = DxlConfig()
    calib = RobotCalibration()
    poses = PoseLibrary()
    ros_bridge = BInspectionROSBridge("/b_inspection_state")
    arm = DynamixelArmB(cfg, calib)
    ctrl = BArmController(arm, poses, cfg, ros_bridge)
    return arm, ctrl, poses, ros_bridge


def print_pose_library(poses: PoseLibrary):
    print("\n[POSE LIBRARY]")
    names = ["B_HOME", "B_BOTTOM_VIEW", "B_BACK_VIEW", "B_FLASH_VIEW"]
    for name in names:
        print(f"{name}: {getattr(poses, name)}")


def print_config(cfg: DxlConfig, calib: RobotCalibration):
    print("\n[DXL CONFIG]")
    print(f"device_name = {cfg.device_name}")
    print(f"baudrate    = {cfg.baudrate}")
    print(f"ids         = {cfg.ids}")

    print("\n[GEAR RATIOS]")
    print(f"q1 ratio = {calib.r_q1:.6f}")
    print(f"q2 ratio = {calib.r_q2:.6f}")
    print(f"q3 ratio = {calib.r_q3:.6f}")
    print(f"q4 ratio = {calib.r_q4:.6f}")
    print(f"q5 ratio = {calib.r_q5:.6f}")
    print(f"q6 ratio = {calib.r_q6:.6f}")

    print("\n[SIGNS]")
    print(f"s_q1_m9 = {calib.s_q1_m9}")
    print(f"s_q2_m10 = {calib.s_q2_m10}")
    print(f"s_q3_m11 = {calib.s_q3_m11}")
    print(f"s_q3_m12 = {calib.s_q3_m12}")
    print(f"s_q4_m13 = {calib.s_q4_m13}")
    print(f"s_q5_m14 = {calib.s_q5_m14}")
    print(f"s_q5_m15 = {calib.s_q5_m15}")
    print(f"s_q6_m16 = {calib.s_q6_m16}")


def print_menu():
    print("\n======================================")
    print(" B ROBOT ARM CONTROL MENU")
    print("======================================")
    print("1. Re-run auto init: mode setup + current pose zero")
    print("2. Ping all motor IDs")
    print("3. Read current motor/joint state")
    print("4. Save current pose as zero")
    print("5. Read device status: voltage/current/temp")
    print("6. Show saved poses")
    print("7. Move to saved pose")
    print("8. Manual joint move: enter q1~q6 on one line")
    print("9. Start ROS inspection/synchronization bridge")
    print("10. Show latest B inspection state")
    print("11. Inspect current camera view for 3 sec")
    print("12. Run synchronized BOTTOM + BACK(8x) + FLASH(8x), repeated cycles")
    print("13. Show config")
    print("Direct command: j q1 q2 q3 q4 q5 q6")
    print("0. Exit")
    print("======================================")



def auto_initialize_on_start(arm: DynamixelArmB):
    print("\n[AUTO INIT] Port open -> Extended Position Mode -> current pose zero")
    print("[AUTO INIT] Make sure the B arm is already placed in the intended safe zero pose.")
    arm.open()
    arm.set_extended_position_mode_all()
    arm.ping_all()
    arm.set_current_as_zero()
    print("[AUTO INIT] Done. You can now use manual joint input immediately.")
    arm.print_device_status()


def parse_joint_values(raw: str):
    """Parse six joint angles separated by spaces or commas."""
    parts = raw.replace(",", " ").split()
    if len(parts) != 6:
        raise ValueError("Enter exactly 6 values in q1 q2 q3 q4 q5 q6 order.")
    return [float(value) for value in parts]


def move_from_joint_line(ctrl: BArmController, raw: str):
    q1, q2, q3, q4, q5, q6 = parse_joint_values(raw)
    ctrl.move_manual(q1, q2, q3, q4, q5, q6)


def print_current_joint_line(arm: DynamixelArmB):
    """Read and print the current q1~q6 angles on one compact line."""
    try:
        ticks = arm.read_present_ticks()
        joints = arm.mapper.present_tick_to_joint_deg(ticks)
        values = " | ".join(
            f"{name}={joints[name]:.1f} deg"
            for name in ["q1", "q2", "q3", "q4", "q5", "q6"]
        )
        print(f"\n[CURRENT JOINTS] {values}")
    except Exception as e:
        print(f"\n[WARN] Could not read current joint angles: {e}")


def main():
    arm, ctrl, poses, ros_bridge = build_robot()

    try:
        try:
            auto_initialize_on_start(arm)
        except Exception as e:
            print(f"[AUTO INIT ERROR] {e}")
            print("Check U2D2 port, power, motor IDs, baudrate, and wiring.")
            return

        while True:
            print_menu()
            sel = input("select: ").strip()

            # Direct joint command, without opening a menu item.
            # Example: j 0 -20 30 0 10 0
            if sel.lower().startswith("j "):
                try:
                    move_from_joint_line(ctrl, sel[2:])
                except ValueError as e:
                    print(f"[ERROR] {e}")
                continue

            if sel == "1":
                if not arm._opened:
                    arm.open()
                arm.set_extended_position_mode_all()
                arm.set_current_as_zero()
                arm.print_device_status()

            elif sel == "2":
                arm.ping_all()

            elif sel == "3":
                arm.print_present_state()

            elif sel == "4":
                confirm = input("Current physical pose will become q=0. Continue? (y/n): ").strip().lower()
                if confirm == "y":
                    arm.set_current_as_zero()
                else:
                    print("[CANCEL]")

            elif sel == "5":
                arm.print_device_status()

            elif sel == "6":
                print_pose_library(poses)

            elif sel == "7":
                pose_name = input("pose name: ").strip().upper()
                valid = ["B_HOME", "B_BOTTOM_VIEW", "B_BACK_VIEW", "B_FLASH_VIEW"]
                if pose_name not in valid:
                    print(f"[ERROR] invalid pose. valid={valid}")
                    continue
                ctrl.move_pose(pose_name)

            elif sel == "8":
                print_current_joint_line(arm)
                raw = input("joints> ").strip()
                try:
                    move_from_joint_line(ctrl, raw)
                except ValueError as e:
                    print(f"[ERROR] {e}")

            elif sel == "9":
                ros_bridge.start()

            elif sel == "10":
                ctrl.preview_inspection_state()

            elif sel == "11":
                view_name = input("view name, ex B_BOTTOM_VIEW/B_BACK_VIEW: ").strip()
                if view_name == "":
                    view_name = "manual"
                ctrl.inspect_current_view(view_name, hold_sec=3.0, required_defect_sec=2.0)

            elif sel == "12":
                ctrl.run_synchronized_two_view_inspection()

            elif sel == "13":
                print_config(arm.cfg, arm.calib)

            elif sel == "0":
                print("[EXIT]")
                break

            else:
                print("[ERROR] invalid input")

    finally:
        ros_bridge.stop()
        arm.close()


if __name__ == "__main__":
    main()
