"""
robot_description / dxl_bridge_suction_5axis.py
--------------------------------------------------
U2D2 + Dynamixel SDK 브리지 (suction 그리퍼).
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
from dynamixel_sdk import PortHandler, PacketHandler, GroupSyncWrite, COMM_SUCCESS

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
    def __init__(self, port='/dev/ttyUSB0', baud=2000000):
        self.port = port; self.baud = baud
        self.all_ids = [i for j in JOINT_MOTORS for i in j['ids']]
        self.zero_tick = {i: 0 for i in self.all_ids}
        self.ph = None; self.pk = None

    def connect(self):
        self.ph = PortHandler(self.port); self.pk = PacketHandler(2.0)
        if not self.ph.openPort():
            raise RuntimeError('포트 열기 실패: ' + self.port)
        if not self.ph.setBaudRate(self.baud):
            raise RuntimeError('보드레이트 실패')
        for i in self.all_ids:
            self._w1(i, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
            self._w1(i, ADDR_OPERATING_MODE, OP_EXTENDED_POSITION)
            self._w4(i, ADDR_PROFILE_VELOCITY, PROFILE_VELOCITY[i])
            self._w4(i, ADDR_PROFILE_ACCELERATION, PROFILE_ACCELERATION[i])
            self._w1(i, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
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
        """각 모터의 현재 부하(Present Load) -> {id: 백분율(%)}.
           +는 한쪽, -는 반대쪽 방향 부하. 0.1% 단위값을 %로 환산."""
        loads = {}
        for i in self.all_ids:
            raw, comm, _ = self.pk.read2ByteTxRx(self.ph, i, ADDR_PRESENT_LOAD)
            if comm != COMM_SUCCESS:
                loads[i] = None
                continue
            if raw > 32767:
                raw -= 65536
            loads[i] = raw / 10.0    # 0.1% 단위 -> %
        return loads

    def read_status(self):
        """각 모터의 (부하%, 전압V) -> {id: (load, volt)}. 실패 시 None."""
        st = {}
        for i in self.all_ids:
            load = None; volt = None
            raw, comm, _ = self.pk.read2ByteTxRx(self.ph, i, ADDR_PRESENT_LOAD)
            if comm == COMM_SUCCESS:
                if raw > 32767:
                    raw -= 65536
                load = raw / 10.0
            v, comm2, _ = self.pk.read1ByteTxRx(self.ph, i, ADDR_PRESENT_VOLTAGE)
            if comm2 == COMM_SUCCESS:
                volt = v / 10.0
            st[i] = (load, volt)
        return st

    def command_joints(self, q):
        goals = self.joints_to_goals(q)
        sw = GroupSyncWrite(self.ph, self.pk, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)
        for mid, g in goals.items():
            sw.addParam(mid, bytes([g & 0xFF, (g >> 8) & 0xFF, (g >> 16) & 0xFF, (g >> 24) & 0xFF]))
        sw.txPacket(); sw.clearParam()

    def emergency_stop(self):
        sw = GroupSyncWrite(self.ph, self.pk, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)
        for i in self.all_ids:
            g = self._present(i)
            sw.addParam(i, bytes([g & 0xFF, (g >> 8) & 0xFF, (g >> 16) & 0xFF, (g >> 24) & 0xFF]))
        sw.txPacket(); sw.clearParam()
        print('[DXL] EMERGENCY STOP')

    def close(self, torque_off=False):
        if self.ph is None:
            return
        if torque_off:
            for i in self.all_ids:
                self._w1(i, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
        self.ph.closePort(); print('[DXL] 포트 닫음')

    def _present(self, i):
        v, comm, _ = self.pk.read4ByteTxRx(self.ph, i, ADDR_PRESENT_POSITION)
        if comm != COMM_SUCCESS:
            raise RuntimeError('present read 실패 id%d' % i)
        return v - (1 << 32) if v >= (1 << 31) else v

    def _w1(self, i, a, v): self.pk.write1ByteTxRx(self.ph, i, a, v)
    def _w4(self, i, a, v): self.pk.write4ByteTxRx(self.ph, i, a, v & 0xFFFFFFFF)
