#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
robot_description / dxl_bridge_centric_5axis.py
--------------------------------------------------
U2D2 + Dynamixel SDK 브리지 (centric 그리퍼).
모터 구성:
  ID1        = BASE     -> q1
  ID2        = CYCLOID  -> q2
  ID3 + ID4  = JOINT1   -> q3  (쌍, 반대)
  ID5 + ID6  = JOINT2   -> q4  (쌍, 반대)
  ID7        = 그리퍼 회전(joint5) -> q5
프로토콜 2.0 / 2,000,000 / Extended Position / 1회전=4096틱.
필요: pip install dynamixel-sdk
"""
import math
import threading
import time
from dynamixel_sdk import PortHandler, PacketHandler, GroupSyncWrite, COMM_SUCCESS

# ---- 통신 보호 ----
#   A팔 ID1~7 과 센트릭 ID8 이 같은 U2D2 를 쓴다. 락으로 순서는 지켜지지만
#   그것만으로는 부족했다.
#     - 직전 트랜잭션의 응답이 버퍼에 남으면 다음 읽기가 그것을 가져간다.
#       (Present Load 자리에 Present Position 값이 들어오던 증상)
#       -> 요청 전마다 clearPort()
#     - 2 Mbps 에서 트랜잭션을 쉼 없이 붙이면 잔여 바이트가 생긴다.
#       -> 트랜잭션 사이 최소 간격
#     - 패킷이 깨지면 SDK 가 IndexError 를 그대로 던진다.
#       -> 예외를 통신 실패로 변환하고 재시도
TXRX_GAP = 0.002
READ_RETRY = 3
READ_RETRY_DELAY = 0.004
COMM_EXCEPTION = -9999

ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_PROFILE_ACCELERATION = 108
ADDR_PROFILE_VELOCITY = 112
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
ADDR_PRESENT_LOAD = 126
ADDR_PRESENT_VOLTAGE = 144
LEN_GOAL_POSITION = 4
OP_EXTENDED_POSITION = 4
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0
DXL_RESOLUTION = 4096

# ★ 관절->모터 매핑 (방향 dir / 감속비 gear 조정)
#   gear = 모터회전 / 관절회전.  잇수비를 분수 그대로 써서 반올림 오차를 없앤다.
#   실측 잇수비:
#     q1 base     16:40  -> 40/16 = 2.5
#     q2 cycloid   1:15  -> 15
#     q3 joint1   16:72  -> 72/16 = 4.5          (구 20:72 = 3.6 에서 변경)
#     q4 joint2   15:50  -> 50/15 = 3.3333...    (구 15:30 = 2.0 에서 변경)
#     q5 gripper   1:1   -> 1                    (직결, 확정)
#   teeth 는 코드가 쓰지 않는 기록용. 조립이 바뀌면 여기부터 확인할 것.
#   dir 을 뒤집으면 그 관절의 회전 방향이 반대가 된다(쌍모터도 dir 하나로 둘 다).
#   pair_sign 은 쌍모터가 서로 반대로 도는 관계이므로 고정.
JOINT_MOTORS = [
    dict(name='q1_base',    ids=[1],    pair_sign=[+1],     dir=+1, gear=40 / 16, teeth=(16, 40)),
    dict(name='q2_cycloid', ids=[2],    pair_sign=[+1],     dir=-1, gear=15 / 1,  teeth=(1, 15)),
    dict(name='q3_joint1',  ids=[3, 4], pair_sign=[+1, -1], dir=-1, gear=72 / 16, teeth=(16, 72)),
    dict(name='q4_joint2',  ids=[5, 6], pair_sign=[+1, -1], dir=-1, gear=50 / 15, teeth=(15, 50)),
    dict(name='q5_gripper', ids=[7],    pair_sign=[+1],     dir=+1, gear=1 / 1,   teeth=(1, 1)),
]
# ★ 속도/가속도 (값 클수록 빠름)
#   관절 각속도[deg/s] = PV * 0.229 * 6 / gear
#     ID1 q1  PV 30  / 2.500 -> 16.5      ID2 q2  PV 120 / 15.000 -> 11.0
#     ID3,4 q3 PV 75 / 4.500 -> 22.9      ID5,6 q4 PV 110 / 3.333 -> 45.3
#   기어비를 올린 q3(3.6->4.5), q4(2.0->3.33) 는 그만큼 PV 도 올려야
#   예전과 비슷한 관절 속도가 나온다. suction/centric 값은 같아야 한다
#   (같은 팔에 그리퍼만 바뀌므로).
#   ★ 올린 뒤에는 반드시 부하율을 확인할 것. ID5,6 이 60% 를 넘으면 낮춘다.
PROFILE_VELOCITY = {1: 30, 2: 120, 3: 75, 4: 75, 5: 110, 6: 110, 7: 30}
# 가속도가 너무 낮으면 짧은 구간에서 최고속도에 도달하지 못해
# 속도를 올린 효과가 나지 않는다. ID5,6 은 2 -> 6 으로 올림.
PROFILE_ACCELERATION = {1: 5, 2: 8, 3: 6, 4: 6, 5: 6, 6: 6, 7: 3}


class DxlBridge:
    def __init__(self, port='/dev/ttyUSB0', baud=2000000, bus_lock=None):
        self.port = port; self.baud = baud
        self.all_ids = [i for j in JOINT_MOTORS for i in j['ids']]
        self.zero_tick = {i: 0 for i in self.all_ids}
        self.ph = None; self.pk = None
        # A팔 ID1~7과 센트릭 ID8이 같은 U2D2를 공유할 때 모든 Tx/Rx를 직렬화한다.
        # 밖에서 같은 락을 넘겨받을 수도 있다(그리퍼와 공유할 때).
        self.bus_lock = bus_lock if bus_lock is not None else threading.RLock()
        self.exc_count = 0
        self.fail_count = 0

    # ---------------- 통신 기본 ----------------
    #   _unlocked 계열은 이미 bus_lock 을 잡은 상태에서 부른다.
    #   (bus_lock 은 RLock 이라 중첩 획득해도 안전하다)
    def _txrx(self, fn, *args):
        """SDK 호출 래퍼. 패킷이 깨지면 SDK 가 예외를 던지므로 잡아서 실패로."""
        try:
            out = fn(*args)
        except Exception:
            self.exc_count += 1
            return (0, COMM_EXCEPTION, 0)
        if isinstance(out, tuple) and len(out) == 3:
            return out
        if isinstance(out, tuple) and len(out) == 2:
            return (0, out[0], out[1])
        return (0, COMM_EXCEPTION, 0)

    def _clear(self):
        try:
            self.ph.clearPort()
        except Exception:
            pass

    def _read_unlocked(self, i, size, addr):
        """레지스터 읽기. 실패하면 재시도. 모두 실패하면 None."""
        fn = {1: self.pk.read1ByteTxRx,
              2: self.pk.read2ByteTxRx,
              4: self.pk.read4ByteTxRx}[size]
        for _n in range(READ_RETRY):
            with self.bus_lock:
                self._clear()
                raw, comm, err = self._txrx(fn, self.ph, i, addr)
            time.sleep(TXRX_GAP)
            if comm == COMM_SUCCESS and err == 0:
                return raw
            time.sleep(READ_RETRY_DELAY)
        self.fail_count += 1
        return None

    def _write_unlocked(self, i, size, addr, value):
        fn = {1: self.pk.write1ByteTxRx,
              2: self.pk.write2ByteTxRx,
              4: self.pk.write4ByteTxRx}[size]
        for _n in range(2):
            with self.bus_lock:
                self._clear()
                _raw, comm, err = self._txrx(fn, self.ph, i, addr, int(value))
            time.sleep(TXRX_GAP)
            if comm == COMM_SUCCESS and err == 0:
                return True
            time.sleep(READ_RETRY_DELAY)
        self.fail_count += 1
        print('[DXL] 쓰기 실패: ID%d addr%d' % (i, addr))
        return False

    def stats(self):
        return 'SDK 예외 %d, 통신 실패 %d' % (self.exc_count, self.fail_count)

    def connect(self):
        self.ph = PortHandler(self.port); self.pk = PacketHandler(2.0)
        if not self.ph.openPort():
            raise RuntimeError('포트 열기 실패: ' + self.port)
        if not self.ph.setBaudRate(self.baud):
            raise RuntimeError('보드레이트 실패')
        with self.bus_lock:
            for i in self.all_ids:
                self._w1_unlocked(i, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
                self._w1_unlocked(i, ADDR_OPERATING_MODE, OP_EXTENDED_POSITION)
                self._w4_unlocked(i, ADDR_PROFILE_VELOCITY, PROFILE_VELOCITY[i])
                self._w4_unlocked(i, ADDR_PROFILE_ACCELERATION, PROFILE_ACCELERATION[i])
                self._w1_unlocked(i, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
        print('[DXL] 연결/설정 완료:', self.all_ids)

    def capture_home(self):
        for i in self.all_ids:
            self.zero_tick[i] = self._present(i)
        print('[DXL] home 기준 저장:', self.zero_tick)

    @staticmethod
    def _r2t(q_rad, gear):
        return (q_rad * gear) / (2 * math.pi) * DXL_RESOLUTION

    def joints_to_goals(self, q):
        goals = {}
        for k, j in enumerate(JOINT_MOTORS):
            if k >= len(q):
                continue
            ticks = self._r2t(q[k], j['gear'])
            for m, mid in enumerate(j['ids']):
                goals[mid] = int(round(self.zero_tick[mid] + j['dir'] * j['pair_sign'][m] * ticks))
        return goals

    def read_joints(self):
        q = []
        for j in JOINT_MOTORS:
            mid = j['ids'][0]
            dt = self._present(mid) - self.zero_tick[mid]
            mr = dt / DXL_RESOLUTION * 2 * math.pi
            q.append(mr / (j['dir'] * j['pair_sign'][0] * j['gear']))
        return q

    def read_loads(self):
        """각 모터의 현재 부하(Present Load) -> {id: 백분율(%)}."""
        loads = {}
        with self.bus_lock:
            for i in self.all_ids:
                raw = self._read_unlocked(i, 2, ADDR_PRESENT_LOAD)
                if raw is None:
                    loads[i] = None
                    continue
                if raw > 32767:
                    raw -= 65536
                load = raw / 10.0
                # 규격은 -100 ~ +100 %. 밖이면 잘못 읽은 값으로 본다.
                loads[i] = None if abs(load) > 100.0 else load
        return loads

    def read_status(self):
        """각 모터의 (부하%, 전압V) -> {id: (load, volt)}. 실패 시 None."""
        st = {}
        with self.bus_lock:
            for i in self.all_ids:
                load = None; volt = None
                raw = self._read_unlocked(i, 2, ADDR_PRESENT_LOAD)
                if raw is not None:
                    if raw > 32767:
                        raw -= 65536
                    load = raw / 10.0
                    if abs(load) > 100.0:
                        load = None
                v = self._read_unlocked(i, 1, ADDR_PRESENT_VOLTAGE)
                if v is not None:
                    volt = v / 10.0
                    if not (3.0 <= volt <= 30.0):
                        volt = None
                st[i] = (load, volt)
        return st

    def command_joints(self, q):
        goals = self.joints_to_goals(q)
        with self.bus_lock:
            self._clear()
            sw = GroupSyncWrite(self.ph, self.pk, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)
            for mid, g in goals.items():
                sw.addParam(mid, bytes([g & 0xFF, (g >> 8) & 0xFF, (g >> 16) & 0xFF, (g >> 24) & 0xFF]))
            try:
                sw.txPacket()
            except Exception:
                self.exc_count += 1
            sw.clearParam()
        time.sleep(TXRX_GAP)

    def emergency_stop(self):
        with self.bus_lock:
            goals = {}
            for i in self.all_ids:
                try:
                    goals[i] = self._present_unlocked(i)
                except Exception:
                    pass          # 한 축을 못 읽어도 나머지는 세운다
            self._clear()
            sw = GroupSyncWrite(self.ph, self.pk, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)
            for i, g in goals.items():
                sw.addParam(i, bytes([g & 0xFF, (g >> 8) & 0xFF, (g >> 16) & 0xFF, (g >> 24) & 0xFF]))
            try:
                sw.txPacket()
            except Exception:
                self.exc_count += 1
            sw.clearParam()
        print('[DXL] EMERGENCY STOP')

    def close(self, torque_off=False):
        if self.ph is None:
            return
        with self.bus_lock:
            if torque_off:
                for i in self.all_ids:
                    self._w1_unlocked(i, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
            self.ph.closePort()
        print('[DXL] 포트 닫음')

    def _present_unlocked(self, i):
        v = self._read_unlocked(i, 4, ADDR_PRESENT_POSITION)
        if v is None:
            raise RuntimeError('present read 실패 id%d (%s)' % (i, self.stats()))
        return v - (1 << 32) if v >= (1 << 31) else v

    def _present(self, i):
        with self.bus_lock:
            return self._present_unlocked(i)

    def _w1_unlocked(self, i, a, v):
        return self._write_unlocked(i, 1, a, v)

    def _w4_unlocked(self, i, a, v):
        return self._write_unlocked(i, 4, a, v & 0xFFFFFFFF)

    def _w1(self, i, a, v):
        with self.bus_lock:
            return self._w1_unlocked(i, a, v)

    def _w4(self, i, a, v):
        with self.bus_lock:
            return self._w4_unlocked(i, a, v)

# ==========================================================
# ROS2 standalone node wrapper
# ==========================================================
# 이 파일의 DxlBridge 클래스는 centric 계열 노드에서 그대로 import 가능하다.
# 아래 노드는 이 모듈 자체를 `ros2 run`으로 단독 실행하고 싶을 때만 사용한다.
#
# Topic
#   subscribe /a_arm/joint_command_deg  std_msgs/Float64MultiArray [q1..q5] deg
#   subscribe /a_arm/command            std_msgs/String
#       "home" / "stop" / "zero" / "status"
#   publish   /a_arm/joint_state_deg    std_msgs/Float64MultiArray [q1..q5] deg
#
# 주의:
#   process_sequence_5axis 노드가 같은 U2D2를 사용 중일 때 이 노드를 동시에 실행하면 안 된다.

def _ros_imports():
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float64MultiArray, String
    return rclpy, Node, Float64MultiArray, String


class DxlBridgeROSNode:
    """실제 Node 상속 클래스는 런타임에 생성한다. 아래 make 함수가 담당한다."""
    pass


def _make_dxl_bridge_node_class():
    rclpy, Node, Float64MultiArray, String = _ros_imports()

    class _DxlBridgeROSNode(Node):
        def __init__(self):
            super().__init__('dxl_bridge_centric_5axis')

            self.declare_parameter('dxl_port', '/dev/ttyUSB0')
            self.declare_parameter('dxl_baud', 2000000)
            self.declare_parameter('auto_capture_home', True)
            self.declare_parameter('state_hz', 2.0)

            port = str(self.get_parameter('dxl_port').value)
            baud = int(self.get_parameter('dxl_baud').value)
            auto_home = bool(self.get_parameter('auto_capture_home').value)
            state_hz = max(0.2, float(self.get_parameter('state_hz').value))

            self.bridge = DxlBridge(port=port, baud=baud)
            self.bridge.connect()

            if auto_home:
                self.bridge.capture_home()
                self.get_logger().warning(
                    'A팔 현재 자세를 q1~q5 = 0 deg 기준으로 저장했습니다.'
                )

            self.cmd_sub = self.create_subscription(
                Float64MultiArray,
                '/a_arm/joint_command_deg',
                self._on_joint_command,
                10,
            )
            self.text_sub = self.create_subscription(
                String,
                '/a_arm/command',
                self._on_text_command,
                10,
            )
            self.state_pub = self.create_publisher(
                Float64MultiArray,
                '/a_arm/joint_state_deg',
                10,
            )
            self.create_timer(1.0 / state_hz, self._publish_state)

            self.get_logger().info(
                'A-arm DXL ROS2 node ready: port=%s, baud=%d' % (port, baud)
            )
            self.get_logger().info(
                'command topic: /a_arm/joint_command_deg [q1 q2 q3 q4 q5] deg'
            )

        def _on_joint_command(self, msg):
            if len(msg.data) != 5:
                self.get_logger().error(
                    '/a_arm/joint_command_deg requires exactly 5 values.'
                )
                return
            try:
                import numpy as _np
                q = _np.radians(_np.asarray(msg.data, dtype=float))
                self.bridge.command_joints(q)
                self.get_logger().info(
                    'A target [deg] = %s' %
                    _np.asarray(msg.data, dtype=float).round(2).tolist()
                )
            except Exception as e:
                self.get_logger().error('A-arm command failed: %s' % e)

        def _on_text_command(self, msg):
            cmd = str(msg.data).strip().lower()
            try:
                if cmd == 'home':
                    import numpy as _np
                    self.bridge.command_joints(_np.zeros(5))
                elif cmd == 'stop':
                    self.bridge.emergency_stop()
                elif cmd == 'zero':
                    self.bridge.capture_home()
                    self.get_logger().warning(
                        '현재 A팔 자세를 새 q=0 기준으로 저장했습니다.'
                    )
                elif cmd == 'status':
                    self.get_logger().info(str(self.bridge.read_status()))
                else:
                    self.get_logger().warning(
                        'unknown /a_arm/command: %s (home/stop/zero/status)' % cmd
                    )
            except Exception as e:
                self.get_logger().error('A-arm text command failed: %s' % e)

        def _publish_state(self):
            try:
                import numpy as _np
                q = self.bridge.read_joints()
                out = Float64MultiArray()
                out.data = [float(v) for v in _np.degrees(q)]
                self.state_pub.publish(out)
            except Exception as e:
                self.get_logger().warning('A-arm state read failed: %s' % e)

        def shutdown_hardware(self):
            try:
                self.bridge.close(torque_off=False)
            except Exception:
                pass

    return _DxlBridgeROSNode


def main(args=None):
    rclpy, _Node, _Float64MultiArray, _String = _ros_imports()
    rclpy.init(args=args)
    node = None
    try:
        NodeClass = _make_dxl_bridge_node_class()
        node = NodeClass()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.shutdown_hardware()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
