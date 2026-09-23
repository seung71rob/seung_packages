"""
robot_description / dxl_bridge_5axis.py
---------------------------------------
U2D2 + Dynamixel SDK 로 실제 로봇팔에 관절 명령을 보내는 브리지.

모터 구성(아두이노 코드 기준):
  ID1        = BASE     -> kinematics q1  (단독)
  ID2        = CYCLOID  -> kinematics q2  (단독)
  ID3 + ID4  = JOINT1   -> kinematics q3  (쌍, 항상 서로 반대방향)
  ID5 + ID6  = JOINT2   -> kinematics q4  (쌍, 항상 서로 반대방향)
  ID7        = JOINT3(엔드이펙터) -> 이번엔 사용 안 함(hold)

프로토콜 2.0 / 보드레이트 2,000,000 / Extended Position / 1회전=4096틱.

필요: pip install dynamixel-sdk
"""

import math
from dynamixel_sdk import (PortHandler, PacketHandler,
                           GroupSyncWrite, COMM_SUCCESS)

# ── 컨트롤 테이블 주소 (Protocol 2.0, X-시리즈) ──
ADDR_OPERATING_MODE     = 11
ADDR_TORQUE_ENABLE      = 64
ADDR_PROFILE_ACCELERATION = 108
ADDR_PROFILE_VELOCITY   = 112
ADDR_GOAL_POSITION      = 116
ADDR_PRESENT_POSITION   = 132
LEN_GOAL_POSITION       = 4
OP_EXTENDED_POSITION    = 4
TORQUE_ENABLE           = 1
TORQUE_DISABLE          = 0
DXL_RESOLUTION          = 4096          # 1회전 틱

# ══════════════════════════════════════════════════════════════════
#  ★ 관절 -> 모터 매핑 (방향/기어를 여기서 조정)
#    dir       : +1/-1. 실제 로봇에서 관절이 반대로 돌면 이 값만 뒤집으세요.
#    pair_sign : 쌍 모터의 고정 배선(항상 반대). 건드리지 마세요.
#    gear      : 관절 1 rad 당 모터 회전수 비율 = (모터회전 / 관절회전).
#                아래는 URDF 기준 감속비(모터:관절)로 계산한 값.
#                  q1 base    16:40 -> 40/16 = 2.5
#                  q2 cycloid  1:15 -> 15/1  = 15.0
#                  q3 ID3+4   20:72 -> 72/20 = 3.6
#                  q4 ID5+6   15:30 -> 30/15 = 2.0
#                비율 해석(모터:관절 / URDF기준)이 다르면 이 숫자만 고치세요.
# ══════════════════════════════════════════════════════════════════
JOINT_MOTORS = [
    dict(name='q1_base',    ids=[1],    pair_sign=[+1],     dir=+1, gear=2.5),
    dict(name='q2_cycloid', ids=[2],    pair_sign=[+1],     dir=-1, gear=15.0),  # [수정됨] 2번 관절 방향(dir)을 +1에서 -1로 변경
    dict(name='q3_joint1',  ids=[3, 4], pair_sign=[+1, -1], dir=+1, gear=3.6),
    dict(name='q4_joint2',  ids=[5, 6], pair_sign=[+1, -1], dir=+1, gear=2.0),
]

# ── ★ 모터별 속도/가속도 (여기 숫자를 바꾸면 속도가 바뀝니다) ──
#    PROFILE_VELOCITY 단위 0.229 rev/min. 값이 작을수록 느림. (A_arm 방식)
PROFILE_VELOCITY = {1: 30, 2: 30, 3: 60, 4: 60, 5: 45, 6: 45}
PROFILE_ACCELERATION = {1: 5, 2: 5, 3: 4, 4: 4, 5: 2, 6: 2}


class DxlBridge:
    def __init__(self, port='/dev/ttyUSB0', baud=2000000):
        self.port = port
        self.baud = baud
        self.all_ids = [i for j in JOINT_MOTORS for i in j['ids']]
        self.zero_tick = {i: 0 for i in self.all_ids}   # home(=q0) 기준 틱
        self.ph = None
        self.pk = None

    # ---------- 연결/설정 ----------
    def connect(self):
        self.ph = PortHandler(self.port)
        self.pk = PacketHandler(2.0)
        if not self.ph.openPort():
            raise RuntimeError(f'포트 열기 실패: {self.port}')
        if not self.ph.setBaudRate(self.baud):
            raise RuntimeError(f'보드레이트 설정 실패: {self.baud}')
        for i in self.all_ids:
            self._w1(i, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
            self._w1(i, ADDR_OPERATING_MODE, OP_EXTENDED_POSITION)
            self._w4(i, ADDR_PROFILE_VELOCITY, PROFILE_VELOCITY[i])
            self._w4(i, ADDR_PROFILE_ACCELERATION, PROFILE_ACCELERATION[i])
            self._w1(i, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
        print('[DXL] 연결/설정 완료:', self.all_ids)

    def capture_home(self):
        """로봇을 물리적 home(모든 관절 0) 에 둔 상태에서 호출.
           지금 위치를 q=0 기준(zero_tick)으로 저장한다."""
        for i in self.all_ids:
            self.zero_tick[i] = self._present(i)
        print('[DXL] home 기준 저장:', self.zero_tick)

    # ---------- 변환 ----------
    @staticmethod
    def _rad_to_ticks(q_rad, gear):
        return (q_rad * gear) / (2 * math.pi) * DXL_RESOLUTION

    def joints_to_goals(self, q):
        """q(길이4, rad) -> {id: goal_tick}"""
        goals = {}
        for k, j in enumerate(JOINT_MOTORS):
            ticks = self._rad_to_ticks(q[k], j['gear'])
            for m, mid in enumerate(j['ids']):
                delta = j['dir'] * j['pair_sign'][m] * ticks
                goals[mid] = int(round(self.zero_tick[mid] + delta))
        return goals

    def read_joints(self):
        """현재 모터 위치 -> q(길이4, rad). 각 관절의 대표 모터(ids[0])로 역산."""
        q = []
        for j in JOINT_MOTORS:
            mid = j['ids'][0]
            dtick = self._present(mid) - self.zero_tick[mid]
            motor_rad = dtick / DXL_RESOLUTION * 2 * math.pi
            q.append(motor_rad / (j['dir'] * j['pair_sign'][0] * j['gear']))
        return q

    # ---------- 명령 ----------
    def command_joints(self, q):
        """q(길이4, rad) 를 모든 모터에 동시(SyncWrite) 전송."""
        goals = self.joints_to_goals(q)
        sw = GroupSyncWrite(self.ph, self.pk, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)
        for mid, g in goals.items():
            data = [g & 0xFF, (g >> 8) & 0xFF, (g >> 16) & 0xFF, (g >> 24) & 0xFF]
            sw.addParam(mid, bytes(data))
        sw.txPacket()
        sw.clearParam()

    def emergency_stop(self):
        """즉시 정지: 각 모터를 현재 위치에 고정(토크 유지)."""
        sw = GroupSyncWrite(self.ph, self.pk, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)
        for i in self.all_ids:
            g = self._present(i)
            data = [g & 0xFF, (g >> 8) & 0xFF, (g >> 16) & 0xFF, (g >> 24) & 0xFF]
            sw.addParam(i, bytes(data))
        sw.txPacket()
        sw.clearParam()
        print('[DXL] EMERGENCY STOP — 현재 위치 고정')

    def close(self, torque_off=False):
        if self.ph is None:
            return
        if torque_off:
            for i in self.all_ids:
                self._w1(i, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
        self.ph.closePort()
        print('[DXL] 포트 닫음')

    # ---------- 저수준 ----------
    def _present(self, dxl_id):
        val, comm, err = self.pk.read4ByteTxRx(self.ph, dxl_id, ADDR_PRESENT_POSITION)
        if comm != COMM_SUCCESS:
            raise RuntimeError(f'present read 실패 id{dxl_id}')
        # 4바이트 부호 처리
        return val - (1 << 32) if val >= (1 << 31) else val

    def _w1(self, dxl_id, addr, value):
        self.pk.write1ByteTxRx(self.ph, dxl_id, addr, value)

    def _w4(self, dxl_id, addr, value):
        self.pk.write4ByteTxRx(self.ph, dxl_id, addr, value & 0xFFFFFFFF)
