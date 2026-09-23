"""
robot_description / suction_sequence_5axis.py
==================================================
5축 팔 + 레일 + 흡착 : suction 공정 시퀀스

공정 흐름
--------
  [대기]  /detected_holes 수신 (Float64MultiArray)
          [mid_x, mid_y, side_x, side_y, angle_deg, distance_mm]  (A3 mm)

  1) 이동자세 -> 레일 초기위치(rail_home_mm, 기본 0)
  2) 레일 pick_rail_mm(640) 이동 -> 도착 대기 -> rail_pause_sec(1초) 정지
  3) 팔 절대각 pick_pose_deg(0, -45, -65, -70) 이동 -> 흡착 ON
  4) 이동자세 -> 레일 초기위치 복귀
  5) 팔 초기위치(0,0,0,0)
  6) 각도 판정 : angle 이 gate_angles_deg(-90, 150, 30) 중 하나의
     ±gate_tol_deg(10) 안에 들면 q5(ID7)를 CW 20도 회전.
     범위 밖이면 q5 = 0 유지. 부품을 든 채 돌리므로 회전 후
     q5_settle_sec 만큼 진동 감쇠를 기다린다.
  7) 1차 위치 : /detected_holes 좌표, z = pos1_z_mm(80)
  8) 2차 위치 : 같은 좌표, z = pos2_z_mm(50)
  9) 하강 : 2차 위치에서 최대 probe_extra_mm(40) 수직 유지로 내려가며
     ID5/ID6 중 하나라도 부하 |load| >= probe_load_pct(30%) 면 즉시 정지
     -> probe_hold_sec(1.0) 정지
 10) 이동자세 -> 레일 pick_rail_mm(640)
 11) 픽 자세에서 흡착 OFF
 12) 이동자세 -> 레일 초기위치 -> 팔 초기위치 복귀

★ 레일 이동 규칙
  모든 레일 이동 앞에는 반드시
      이동자세 명령 -> settle(엔코더 도달 확인) -> rail_settle_sec 대기
  가 들어간다. 팔이 뻗은 상태로 레일이 출발하는 일을 막는다.

토픽
----
  구독  /detected_holes     구멍 좌표+각도 (Float64MultiArray)
        /suction_cmd        stop / reset / auto on|off / home
        /rail_cmd           레일 원시 명령 통과 (z / p <mm> / m <mm> / s / 0 / ?)
  발행  /suction_state      현재 단계 (String)
        /rail_position      레일 위치 (Float64)

주의
----
  - auto_run:=false(기본) 이면 스텝마다 APPROVE 버튼을 눌러야 넘어간다.
    처음에는 반드시 false 로 확인할 것.
  - 관절각은 모두 '관절' 기준(도). 모터각이 아니다.
  - 탐침 부하 임계 등은 환경변수로도 조정할 수 있다(실기 튜닝용).
      export SUCTION_PROBE_LOAD=15        부하 임계 %
      export SUCTION_PROBE_EXTRA_MM=80    최대 하강 mm
      export SUCTION_PROBE_SPEED=0.01     하강 속도 m/s
      export SUCTION_PROBE_HOLD=1.5       감지 후 정지 초
    우선순위: --ros-args -p ...  >  환경변수  >  기본값
  - ★ 레일은 아두이노 zeroSet 이 true 여야 움직인다. 공정 시작 전 반드시
    원점을 설정할 것.  ros2 topic pub --once /rail_cmd std_msgs/msg/String "{data: 'z'}"
    원점 미설정 상태로 시작하면 공정을 거부한다(require_rail_zero).
  - 레일은 rail_stepmotor.ino 의 S-curve 프로파일(500 RPM = 83.33 mm/s)을
    그대로 사용한다. 노드가 속도를 지정하지 않는다.
"""

import json
import os
import time
import xml.etree.ElementTree as ET
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float64MultiArray, String
import pybullet as p
import pybullet_data

from .kinematics_ee_5axis import (fk, fk_pos, ik, ik_fix, ik_fix_candidates,
                                  N_JOINTS, wrap_to_pi)
from .trajectory_5axis import plan_joint_path
from .path_collision_guard_ee_5axis import CollisionChecker
from .rail_serial_5axis import RailSerial, RAIL_MIN_M, RAIL_MAX_M
from .vertical_path_5axis import (plan_descent_ends_retimed, plan_joint_descent_retimed,
                                  plan_sync_joint_move, retime_path, joint_speed_limits,
                                  fix_reach_report, ik_level_soft, level_error)

URDF_NAME = 'robot_arm_5axis_rail_suction.urdf'
ARM_URDF_NAME = 'robot_arm_5axis_suction.urdf'
RAIL_JOINT_NAME = 'joint_rail'

# ---- A3 -> 로봇 사용자 좌표 (검증본과 동일) ----
A3_W_MM, A3_H_MM = 420.0, 297.0
A3_LT_ROBOT_X_MM = 482.0
A3_LT_ROBOT_Y_MM = 214.55
USER_X_SIGN, USER_Y_SIGN, USER_Z_SIGN = -1.0, +1.0, +1.0


def _env_float(name, default):
    """환경변수에서 실수를 읽는다. 없거나 형식이 틀리면 default."""
    v = os.environ.get(name, '').strip()
    if not v:
        return float(default)
    try:
        return float(v)
    except ValueError:
        print('[WARN] 환경변수 %s=%r 를 숫자로 읽을 수 없어 기본값 %s 사용'
              % (name, v, default))
        return float(default)


def user_mm_to_arm_m(x_mm, y_mm, z_mm):
    return np.array([USER_X_SIGN * float(x_mm) / 1000.0,
                     USER_Y_SIGN * float(y_mm) / 1000.0,
                     USER_Z_SIGN * float(z_mm) / 1000.0], float)


class SuctionSeqNode(Node):

    # ================================================== 초기화
    def __init__(self):
        super().__init__('suction_sequence_5axis')

        d = self.declare_parameter
        d('pkg_dir', os.path.expanduser('~/robot_sim/src/robot_description'))
        d('mesh_dir', os.path.expanduser('~/robot_sim/src/robot_description/meshes_robot_5axis'))
        d('use_robot', False)
        d('use_rail', False)
        d('rail_port', '/dev/ttyACM0')
        d('rail_baud', 115200)
        d('dxl_port', '/dev/ttyUSB0')
        d('check_collision', True)
        d('sim_hz', 240.0)
        d('robot_cmd_hz', 50.0)
        d('render_hz', 60.0)
        d('speed_deg_per_s', 60.0)
        d('auto_run', False)
        # ---- 좌표 / 자세 ----
        d('hole_topic', '/detected_holes')     # Float64MultiArray
        d('hole_point', 'mid')                 # 'mid' 또는 'side'
        d('hole_rail_ref_m', 0.0)              # A3 를 잰 시점의 레일 위치
        d('pos1_z_mm', 80.0)                   # 1차 위치 z
        d('pos2_z_mm', 50.0)                   # 2차 위치 z
        d('probe_extra_mm', _env_float('SUCTION_PROBE_EXTRA_MM', 40.0))  # 최대 하강 거리
        d('probe_hold_sec', _env_float('SUCTION_PROBE_HOLD', 1.0))  # 부하 감지 후 정지 시간
        d('pick_pose_deg', [0.0, -45.0, -65.0, -65.0])
        d('move_pose_deg', [0.0, 50.0, -50.0, -95.0])
        d('drop_arm_xyz', [-0.35, 0.0, 0.05])   # 흡착 해제 좌표 (팔 기준, m)
        d('home_pose_deg', [0.0, 0.0, 0.0, 0.0])
        # ---- 각도 게이트 ----
        d('gate_angles_deg', [-90.0, 150.0, 30.0])
        d('gate_tol_deg', 10.0)
        d('gate_q5_cw_deg', -20.0)             # CW 20도. 방향 반대면 +20.0
        # ---- 레일 ----
        d('rail_home_mm', 0.0)
        d('pick_rail_mm', 630.0)
        d('rail_pause_sec', 1.0)               # 레일 도착 후 정지 시간
        # ★ 레일은 팔이 이동자세에 실제로 도달한 뒤에만 움직인다.
        #   settle 로 엔코더 도달을 확인하고, 그 뒤 이만큼 더 기다린 다음
        #   레일 명령을 보낸다. 잔여 진동이 가라앉을 시간이다.
        d('rail_settle_sec', 1.0)
        d('require_rail_zero', True)           # 원점 미설정이면 공정 거부
        d('rail_reject_timeout', 2.0)          # rail_abs 후 거부(@L) 감시 시간
        # ---- 탐침 ----
        # 부하 임계는 실기에서 찾아야 하는 값이라 환경변수로도 조정할 수 있다.
        #   export SUCTION_PROBE_LOAD=15
        # 우선순위: --ros-args -p probe_load_pct:=  >  환경변수  >  기본값
        d('probe_load_pct', _env_float('SUCTION_PROBE_LOAD', 30.0))
        d('probe_ids', [5, 6])
        d('probe_poll_sec', 0.05)
        d('probe_speed', _env_float('SUCTION_PROBE_SPEED', 0.02))   # 탐침 하강 속도 m/s
        # ---- 경로 ----
        d('descend_speed', 0.05)
        d('strict_ends', 'both')
        d('level_tol_deg', 25.0)
        d('limit_joint_speed', True)
        d('speed_margin', 0.9)
        d('descent_fallback_joint', True)
        d('suction_dwell', 0.5)
        d('q5_settle_sec', 0.5)                # q5 회전 후 진동 감쇠 대기
        # ---- 안전 ----
        d('settle_tol_deg', 1.5)
        d('q5_settle_tol_deg', 2.0)
        d('settle_timeout', 5.0)

        g = self.get_parameter
        self.pkg_dir = g('pkg_dir').value
        self.mesh_dir = g('mesh_dir').value
        self.use_robot = bool(g('use_robot').value)
        self.use_rail = bool(g('use_rail').value)
        self.sim_hz = float(g('sim_hz').value)
        self.robot_cmd_hz = float(g('robot_cmd_hz').value)
        self.render_hz = max(1.0, float(g('render_hz').value))
        self.speed = float(g('speed_deg_per_s').value)
        self.auto_run = bool(g('auto_run').value)

        self.hole_point = str(g('hole_point').value).lower()
        self.hole_rail_ref = float(g('hole_rail_ref_m').value)
        self.pos1_z = float(g('pos1_z_mm').value)
        self.pos2_z = float(g('pos2_z_mm').value)
        self.probe_extra = float(g('probe_extra_mm').value)
        self.probe_hold = float(g('probe_hold_sec').value)
        self.pick_pose = np.radians(np.array([float(v) for v in g('pick_pose_deg').value], float))
        self.move_pose = np.radians(np.array([float(v) for v in g('move_pose_deg').value], float))
        self.drop_arm = np.array([float(v) for v in g('drop_arm_xyz').value], float)
        self.home_pose = np.radians(np.array([float(v) for v in g('home_pose_deg').value], float))

        self.gate_angles = [float(v) for v in g('gate_angles_deg').value]
        self.gate_tol = float(g('gate_tol_deg').value)
        self.gate_q5 = np.radians(float(g('gate_q5_cw_deg').value))

        self.rail_home_mm = float(g('rail_home_mm').value)
        self.pick_rail_mm = float(g('pick_rail_mm').value)
        self.rail_pause = float(g('rail_pause_sec').value)
        self.rail_settle = float(g('rail_settle_sec').value)
        self.require_rail_zero = bool(g('require_rail_zero').value)
        self.rail_reject_timeout = float(g('rail_reject_timeout').value)

        self.probe_load = float(g('probe_load_pct').value)
        self.probe_ids = [int(v) for v in g('probe_ids').value]
        self.probe_poll = float(g('probe_poll_sec').value)
        self.probe_speed = float(g('probe_speed').value)

        self.descend_speed = float(g('descend_speed').value)
        self.strict_ends = str(g('strict_ends').value).lower()
        self.level_tol = np.radians(float(g('level_tol_deg').value))
        self.limit_joint_speed = bool(g('limit_joint_speed').value)
        self.speed_margin = float(g('speed_margin').value)
        self.descent_fallback_joint = bool(g('descent_fallback_joint').value)
        self.suction_dwell = float(g('suction_dwell').value)
        self.q5_settle_sec = float(g('q5_settle_sec').value)

        self.settle_tol = np.radians(float(g('settle_tol_deg').value))
        self.q5_settle_tol = np.radians(float(g('q5_settle_tol_deg').value))
        self.settle_timeout = float(g('settle_timeout').value)

        self.rail_origin = self._read_rail_origin()

        # ---------- PyBullet ----------
        urdf_abs = self._prep(URDF_NAME)
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)
        p.loadURDF('plane.urdf')
        p.resetDebugVisualizerCamera(1.6, 50, -25, [0.0, 0.4, 0.4])
        self.robot = p.loadURDF(urdf_abs, [0, 0, 0], useFixedBase=True)
        self.rev, self.rail_jid = [], None
        for j in range(p.getNumJoints(self.robot)):
            info = p.getJointInfo(self.robot, j)
            nm = info[1].decode() if isinstance(info[1], bytes) else str(info[1])
            if info[2] == p.JOINT_REVOLUTE:
                self.rev.append(j)
            elif info[2] == p.JOINT_PRISMATIC and nm == RAIL_JOINT_NAME:
                self.rail_jid = j

        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.012, rgbaColor=[1, 0, 0, 0.9])
        self.marker = p.createMultiBody(0, baseVisualShapeIndex=vis, basePosition=[10, 10, 10])

        self.b_stop = p.addUserDebugParameter('EMERGENCY STOP', 1, 0, 0)
        self.b_appr = p.addUserDebugParameter('APPROVE (auto off 일 때)', 1, 0, 0)
        self.b_auto = p.addUserDebugParameter('AUTO on/off (click)', 1, 0, 0)
        self.b_von = p.addUserDebugParameter('SUCTION  ON', 1, 0, 0)
        self.b_voff = p.addUserDebugParameter('SUCTION  OFF', 1, 0, 0)
        self.b_home = p.addUserDebugParameter('GO HOME', 1, 0, 0)
        self.b_rzero = p.addUserDebugParameter('RAIL set zero here', 1, 0, 0)
        self._cl = {k: 0 for k in ('stop', 'appr', 'auto', 'von', 'voff', 'home', 'rzero')}
        self._txt = {}

        self.checker = None
        if bool(g('check_collision').value):
            self.checker = CollisionChecker(os.path.join(self.pkg_dir, 'urdf', ARM_URDF_NAME),
                                            self.pkg_dir, N_JOINTS, self.mesh_dir)

        # ---------- 하드웨어 ----------
        self.rail, self.rail_m = None, 0.0
        if self.use_rail:
            try:
                self.rail = RailSerial(port=g('rail_port').value,
                                       baud=int(g('rail_baud').value),
                                       logger=self.get_logger())
                self.rail.connect()
            except Exception as e:
                self.rail = None
                self.get_logger().error('레일 연결 실패: %s (레일 없이 진행)' % e)

        self.bridge, self.vmax = None, None
        if self.use_robot:
            from .dxl_bridge_suction_5axis import DxlBridge, PROFILE_VELOCITY, JOINT_MOTORS
            self.bridge = DxlBridge(port=g('dxl_port').value)
            self.bridge.connect(); self.bridge.capture_home()
            self.get_logger().warn('실로봇: 로봇을 home(관절 0)에 두고 시작하세요.')
            if self.limit_joint_speed:
                try:
                    gears = [j['gear'] for j in JOINT_MOTORS[:N_JOINTS]]
                    self.vmax = joint_speed_limits(PROFILE_VELOCITY, gears)
                    self.get_logger().info('관절 속도한계 q1~q4 = %s deg/s'
                                           % np.degrees(self.vmax).round(2))
                except Exception as e:
                    self.get_logger().warn('속도한계 계산 실패: %s' % e)

        # ---------- 실행 상태 ----------
        self.q_cmd = np.zeros(N_JOINTS)
        self.q5 = 0.0
        self.traj = None
        self.idx = 0
        self.step = None
        self.queue = deque()
        self.phase = 'IDLE'
        self.cycle_id = ''
        self.busy = False
        self.hole_world = None          # 이번 사이클의 구멍 좌표 (월드, z 제외)
        self.hole_angle = None
        self._play_t0 = None
        self._last_cmd_t = 0.0
        self._render_tick = 0
        self._settle_t0 = None
        self._settle_tick = 0
        self._await_approve = False
        self._probe_hit = None

        # ---------- 토픽 ----------
        self.state_pub = self.create_publisher(String, 'suction_state', 10)
        self.rail_pub = self.create_publisher(Float64, 'rail_position', 10)
        self.create_subscription(Float64MultiArray, str(g('hole_topic').value),
                                 self.on_hole, 10)
        self.create_subscription(String, 'suction_cmd', self.on_cmd, 10)
        self.create_subscription(String, 'rail_cmd', self.on_rail_cmd, 10)

        self._apply_sim(self.q_cmd)
        self.create_timer(1.0 / self.sim_hz, self.tick)
        self.get_logger().info('Ready (suction_sequence, use_robot=%s, use_rail=%s, auto=%s)'
                               % (self.use_robot, self.rail is not None, self.auto_run))
        self.get_logger().info('탐침 설정: 부하 %.1f%% (ID %s), 최대 하강 %.0fmm, '
                               '속도 %.3f m/s, 감지 후 %.1fs 정지'
                               % (self.probe_load, self.probe_ids, self.probe_extra,
                                  self.probe_speed, self.probe_hold))
        self.get_logger().info('  부하 임계는 export SUCTION_PROBE_LOAD=15 또는 '
                               '-p probe_load_pct:=15.0 으로 조정')
        self.get_logger().info('각도 게이트 %s ±%.0f도 -> q5 %.0f도'
                               % (self.gate_angles, self.gate_tol,
                                  np.degrees(self.gate_q5)))

    # ================================================== URDF / 좌표
    def _read_rail_origin(self):
        try:
            root = ET.parse(os.path.join(self.pkg_dir, 'urdf', URDF_NAME)).getroot()
            for j in root.findall('joint'):
                if j.get('name') == RAIL_JOINT_NAME:
                    return np.array([float(v) for v in j.find('origin').get('xyz').split()], float)
        except Exception as e:
            self.get_logger().warn('rail origin 읽기 실패(0 가정): %s' % e)
        return np.zeros(3)

    def _prep(self, name):
        with open(os.path.join(self.pkg_dir, 'urdf', name)) as f:
            txt = f.read()
        txt = txt.replace('robot_meshes/', self.mesh_dir.rstrip('/') + '/')
        txt = txt.replace('package://robot_description/', self.pkg_dir + '/')
        out = '/tmp/' + name
        with open(out, 'w') as f:
            f.write(txt)
        return out

    def world_to_arm(self, w):
        return np.asarray(w, float) - (self.rail_origin + np.array([0.0, self.rail_m, 0.0]))

    def arm_to_world(self, a):
        return np.asarray(a, float) + (self.rail_origin + np.array([0.0, self.rail_m, 0.0]))

    def a3_to_world(self, ax, ay, z_mm):
        """A3 mm -> 월드 m. A3 를 잰 시점의 레일 위치를 기준으로 고정."""
        rx = A3_LT_ROBOT_X_MM - float(ay)
        ry = float(ax) - A3_LT_ROBOT_Y_MM
        p_arm = user_mm_to_arm_m(rx, ry, z_mm)
        return p_arm + self.rail_origin + np.array([0.0, self.hole_rail_ref, 0.0])

    # ================================================== 경로 조각
    def _path_ok(self, seg):
        return self.checker is None or self.checker.check_path(seg)[0]

    def _sync_to(self, q_from, q_goal, tag):
        q_goal = wrap_to_pi(np.asarray(q_goal, float).ravel()[:N_JOINTS])
        if self.checker is not None and not self.checker.check_config(q_goal)[0]:
            return None, '%s: 목표 자세 충돌' % tag
        seg, T, mx, info = plan_sync_joint_move(
            q_from, q_goal, hz=self.sim_hz,
            vmax=self.vmax if self.limit_joint_speed else None,
            margin=self.speed_margin)
        if seg is None:
            return None, '%s: %s' % (tag, info)
        if not self._path_ok(seg):
            return None, '%s: 경로 충돌' % tag
        return seg, ''

    def _descend(self, q_from, p_from, p_to, speed=None):
        spd = self.descend_speed if speed is None else float(speed)
        seg, T, mx, msg = plan_descent_ends_retimed(
            q_from, p_from, p_to, hz=self.sim_hz, lin_speed=spd,
            mode=self.strict_ends, level_tol=self.level_tol,
            vmax=self.vmax if self.limit_joint_speed else None, margin=self.speed_margin)
        if seg is not None:
            return seg, '직선', ''
        if not self.descent_fallback_joint:
            return None, '', msg
        seg, T, mx, msg2 = plan_joint_descent_retimed(
            q_from, p_from, p_to, hz=self.sim_hz, lin_speed=spd,
            vmax=self.vmax if self.limit_joint_speed else None, margin=self.speed_margin)
        if seg is None:
            return None, '', msg2 or msg
        return seg, '관절공간', ''

    def _vertical_goto(self, q_from, p_arm, tag):
        """수직 자세로 특정 팔 좌표까지 이동."""
        cands = ik_fix_candidates(p_arm, q0=q_from)
        if cands:
            cands.sort(key=lambda c: np.linalg.norm(wrap_to_pi(c[0] - q_from)))
            q_t = cands[0][0]
        else:
            q_t, e, lv = ik_level_soft(p_arm, q_from)
            if e > 1e-4:
                return None, '%s: 도달 불가 (%s)' % (tag, fix_reach_report(p_arm)['msg'])
            if lv > self.level_tol:
                return None, ('%s: 기울기 %.1f도 > 허용 %.1f도'
                              % (tag, np.degrees(lv), np.degrees(self.level_tol)))
            self.get_logger().warn('%s: 수직에서 %.1f도 기울어집니다.' % (tag, np.degrees(lv)))
        return self._sync_to(q_from, q_t, tag)

    # ================================================== 스텝
    def _S(self, kind, **kw):
        s = dict(kind=kind)
        s.update(kw)
        return s

    def build_sequence(self, w_xy, angle_deg):
        """
        전체 공정 스텝 생성. 좌표 의존 구간은 실행 시점에 계산해야
        레일 위치가 반영되므로, goto_hole 스텝으로 미뤄둔다.
        """
        q5_target = self.gate_q5 if self._angle_in_gate(angle_deg) else 0.0
        gate_txt = ('게이트 통과 -> q5 %.0f도' % np.degrees(q5_target)
                    if q5_target != 0.0 else '게이트 미통과 -> q5 0도')

        def rail_to(mm, tag):
            """
            레일 이동 묶음.

            ★ 레일은 팔이 이동자세에 실제로 도달한 뒤에만 움직인다.
              goto_q 로 이동자세를 명령하고, settle 로 엔코더 도달을
              확인하고, rail_settle_sec 만큼 더 기다린 다음에야
              rail_abs 를 보낸다. 이 순서를 어기면 팔이 아직 뻗어 있는
              상태에서 레일이 출발해 주변과 부딪칠 수 있다.
            """
            return [
                self._S('goto_q', target=self.move_pose,
                        name='이동자세 (레일 %s 전)' % tag),
                self._S('settle', name='이동자세 도달 확인'),
                self._S('wait', sec=self.rail_settle,
                        name='도달 후 %.1f초 대기 (레일 출발 전)' % self.rail_settle),
                self._S('rail_abs', mm=mm, name='레일 %s (%.0fmm)' % (tag, mm)),
                self._S('rail_wait', name='레일 도착 대기'),
            ]

        steps = []

        # 1) 시작 정렬 : 이동자세 -> 레일 초기위치
        steps += rail_to(self.rail_home_mm, '초기위치')

        # 2) 레일 픽 위치. 팔은 이미 이동자세이므로 바로 보낸다.
        steps += [
            self._S('wait', sec=self.rail_settle,
                    name='%.1f초 대기 (레일 출발 전)' % self.rail_settle),
            self._S('rail_abs', mm=self.pick_rail_mm,
                    name='레일 픽 위치 (%.0fmm)' % self.pick_rail_mm),
            self._S('rail_wait', name='레일 도착 대기'),
            self._S('wait', sec=self.rail_pause,
                    name='레일 정지 %.1fs' % self.rail_pause),
        ]

        # 3) 픽 자세 -> 흡착 ON
        steps += [
            self._S('goto_q', target=self.pick_pose,
                    name='픽 자세 %s' % np.degrees(self.pick_pose).round(0).tolist()),
            self._S('settle', name='픽 자세 도달 확인'),
            self._S('suction', on=True, name='흡착 ON'),
            self._S('wait', sec=self.suction_dwell, name='흡착 유지'),
        ]

        # 4) 이동자세 -> 레일 초기위치
        steps += rail_to(self.rail_home_mm, '초기위치 복귀')

        # 5) 팔 초기위치
        steps += [
            self._S('goto_q', target=self.home_pose, name='팔 초기위치'),
            self._S('settle', name='초기위치 도달 확인'),
        ]

        # 6) 각도 게이트 -> q5
        steps += [
            self._S('q5', deg=np.degrees(q5_target), name=gate_txt),
            self._S('settle', check_q5=True, name='그리퍼 도달 확인'),
            self._S('wait', sec=self.q5_settle_sec,
                    name='회전 후 진동 감쇠 %.1fs' % self.q5_settle_sec),
        ]

        # 7) 1차 위치 -> 2차 위치
        steps += [
            self._S('goto_hole', z_mm=self.pos1_z,
                    name='1차 위치 (z=%.0fmm)' % self.pos1_z),
            self._S('settle', name='1차 위치 도달 확인'),
            self._S('goto_hole', z_mm=self.pos2_z,
                    name='2차 위치 (z=%.0fmm)' % self.pos2_z),
            self._S('settle', name='2차 위치 도달 확인'),
        ]

        # 8) 부하 감지까지 수직 유지 하강
        steps += [
            self._S('probe', name='하강 (최대 %.0fmm, 부하 %.0f%% 까지)'
                    % (self.probe_extra, self.probe_load)),
            self._S('wait', sec=self.probe_hold,
                    name='부하 감지 후 %.1f초 정지' % self.probe_hold),
        ]

        # 9) 이동자세 -> 레일 픽 위치
        steps += rail_to(self.pick_rail_mm, '픽 위치')

        # 10) 픽 자세에서 흡착 OFF
        steps += [
            self._S('goto_q', target=self.pick_pose,
                    name='배출 자세 %s' % np.degrees(self.pick_pose).round(0).tolist()),
            self._S('settle', name='배출 자세 도달 확인'),
            self._S('suction', on=False, name='흡착 OFF'),
            self._S('wait', sec=0.5, name='분리 대기'),
        ]

        # 11) 이동자세 -> 레일 초기위치 -> 팔 초기위치
        steps += rail_to(self.rail_home_mm, '초기위치 복귀')
        steps += [
            self._S('goto_q', target=self.home_pose, name='팔 초기위치 복귀'),
        ]
        return steps

    def _angle_in_gate(self, angle_deg):
        if angle_deg is None:
            return False
        a = float(angle_deg)
        for g in self.gate_angles:
            if abs(float(wrap_to_pi(np.radians(a - g)))) <= np.radians(self.gate_tol):
                self.get_logger().info('  각도 %.1f도 -> 게이트 %.0f도 (±%.0f) 통과'
                                       % (a, g, self.gate_tol))
                return True
        self.get_logger().info('  각도 %.1f도 -> 게이트 %s 밖' % (a, self.gate_angles))
        return False

    # ================================================== 콜백
    def on_hole(self, msg):
        """
        /detected_holes : Float64MultiArray
          [mid_x_mm, mid_y_mm, side_x_mm, side_y_mm, angle_deg, distance_mm]
        A3 좌상단 기준 mm. baseplate_hole_wonseok.py 가 STABLE 상태에서 1회 발행.
        """
        if self.busy:
            return
        d = list(msg.data)
        if len(d) < 5:
            self.get_logger().error('[HOLE] 값이 %d개뿐입니다. 6개 필요 '
                                    '[mid_x, mid_y, side_x, side_y, angle, distance]' % len(d))
            return
        mid = (float(d[0]), float(d[1]))
        side = (float(d[2]), float(d[3]))
        ang = float(d[4])
        dist = float(d[5]) if len(d) > 5 else float('nan')

        ax, ay = mid if self.hole_point == 'mid' else side
        if not (0.0 <= ax <= A3_W_MM and 0.0 <= ay <= A3_H_MM):
            self.get_logger().error('[HOLE] A3 영역 밖: (%.1f, %.1f)  '
                                    'A3 캘리브레이션(4점 클릭)을 확인하세요.' % (ax, ay))
            return

        self.cycle_id = time.strftime('%H%M%S')
        self.hole_world = self.a3_to_world(ax, ay, 0.0)
        self.hole_angle = ang
        self.get_logger().info('[HOLE] mid(%.1f, %.1f) side(%.1f, %.1f) angle=%.1f도 '
                               'dist=%.1fmm -> %s 사용, 월드 %s  cycle=%s'
                               % (mid[0], mid[1], side[0], side[1], ang, dist,
                                  self.hole_point, self.hole_world.round(4), self.cycle_id))

        if not self._rail_ready():
            return
        self._start(self.build_sequence(self.hole_world, ang), 'RUN')

    def _rail_ready(self):
        """
        레일 원점이 설정되어 있는지 확인.

        아두이노는 zeroSet 이 false 면 이동 명령을 거부한다(@L). 그대로
        진행하면 레일은 멈춰 있는데 팔만 움직여 엉뚱한 위치에서 흡착한다.
        """
        if self.rail is None:
            if self.require_rail_zero and self.use_rail:
                self.get_logger().error('레일이 연결되지 않았습니다.')
                return False
            return True
        if not self.rail.zero_set:
            self.get_logger().error('레일 원점이 설정되지 않았습니다. 공정을 시작할 수 없습니다.')
            self.get_logger().error('  레일을 원점에 두고:')
            self.get_logger().error("  ros2 topic pub --once /rail_cmd "
                                    "std_msgs/msg/String \"{data: 'z'}\"")
            self.get_logger().error('  또는 PyBullet 의 RAIL set zero here 버튼')
            if not self.require_rail_zero:
                self.get_logger().warn('  require_rail_zero:=false 이므로 그대로 진행합니다.')
                return True
            return False
        return True

    def on_rail_cmd(self, msg):
        """
        아두이노 레일 명령을 그대로 통과시킨다.
          z        현재 위치를 0 mm 로 설정 (이동 전 반드시 1회)
          p <mm>   절대 이동      m <mm>  상대 이동
          s        부드러운 정지  0       즉시 정지     ?  위치 보고
        """
        if self.rail is None:
            self.get_logger().warn('레일 미연결 (use_rail:=true 로 실행).')
            return
        cmd = str(msg.data).strip().lower()
        ok = (cmd in ('z', 's', '0', '?', 'c', 'cw', 'ccw', 'cc', 'von', 'voff', 'v?')
              or cmd.startswith('p ') or cmd.startswith('m '))
        if not ok:
            self.get_logger().warn(
                "알 수 없는 레일 명령: '%s'  (z | p <mm> | m <mm> | s | 0 | ?)" % cmd)
            return
        if self.busy and cmd not in ('s', '0', '?'):
            self.get_logger().warn('공정 진행 중에는 정지 명령만 받습니다.')
            return
        self.rail.send(cmd)
        self.get_logger().info('레일 명령 전송: %s' % cmd)

    def on_cmd(self, msg):
        c = str(msg.data).strip().lower()
        if c == 'stop':
            self._abort('사용자 정지')
        elif c == 'reset':
            self.queue.clear(); self.step = None; self.traj = None
            self.busy = False; self._set_phase('IDLE')
        elif c in ('auto on', 'auto_on'):
            self.auto_run = True; self.get_logger().info('AUTO ON')
        elif c in ('auto off', 'auto_off'):
            self.auto_run = False; self.get_logger().info('AUTO OFF')
        elif c == 'home':
            self._start([self._S('goto_q', target=self.home_pose, name='초기위치 복귀')], 'HOME')
        else:
            self.get_logger().warn('suction_cmd: stop | reset | auto on | auto off | home')

    # ================================================== 스텝 엔진
    def _start(self, steps, phase):
        self.queue = deque(steps)
        self.step = None
        self.busy = True
        self._set_phase(phase)
        self._await_approve = not self.auto_run

    def _set_phase(self, ph):
        self.phase = ph
        m = String(); m.data = '%s|%s' % (ph, self.cycle_id)
        self.state_pub.publish(m)
        self.get_logger().info('--- PHASE: %s ---' % ph)

    def _abort(self, why):
        self.get_logger().error('중단: %s' % why)
        self.queue.clear(); self.step = None; self.traj = None
        self.busy = False
        if self.bridge:
            try:
                self.bridge.emergency_stop()
            except Exception:
                pass
        if self.rail:
            self.rail.estop()
        self._set_phase('ABORT')

    def _next_step(self):
        if not self.queue:
            self.busy = False
            self._set_phase('DONE')
            self.step = None
            return
        self.step = self.queue.popleft()
        self.step['_t0'] = None
        self.step['_done'] = False
        k = self.step['kind']
        self.get_logger().info('[%s] %s' % (k, self.step.get('name', '')))

        if k == 'suction':
            if self.rail is not None:
                self.rail.suction(bool(self.step['on']))
            self.step['_done'] = True
        elif k == 'rail_abs':
            if self.rail is None:
                self.step['_done'] = True
            else:
                self.rail.last_event = ''
                self.rail.move_abs_mm(float(self.step['mm']))
                self.step['_t0'] = time.time()
                self.step['_target_mm'] = float(self.step['mm'])
                # 거부(@L) 여부를 잠시 지켜본 뒤 다음으로 넘어간다
        elif k == 'q5':
            self.q5 = np.radians(float(self.step['deg']))
            if self.bridge:
                self.bridge.command_joints(list(self.q_cmd) + [self.q5])
            self.step['_done'] = True
        elif k == 'goto_q':
            seg, e = self._sync_to(self.q_cmd, self.step['target'], self.step.get('name', 'goto'))
            if seg is None:
                self._abort(e); return
            self._play_seg(seg)
        elif k == 'goto_hole':
            self._start_goto_hole(self.step)
        elif k == 'goto_arm':
            p_arm = np.asarray(self.step['target'], float)
            seg, e = self._vertical_goto(self.q_cmd, p_arm, self.step.get('name', 'goto_arm'))
            if seg is None:
                self._abort(e); return
            p.resetBasePositionAndOrientation(self.marker,
                                              self.arm_to_world(p_arm).tolist(), [0, 0, 0, 1])
            self._play_seg(seg)
        elif k == 'probe':
            self._start_probe(self.step)
        elif k == 'settle':
            self._settle_t0 = None

    def _play_seg(self, seg):
        self.traj = seg
        self.idx = 0
        self._play_t0 = None
        self.step['kind'] = 'traj'

    def _start_goto_hole(self, st):
        """구멍 좌표 + 지정 z 로 수직 이동. 실행 시점의 레일 위치를 반영."""
        if self.hole_world is None:
            self._abort('구멍 좌표가 없습니다.'); return
        w = np.array(self.hole_world, float).copy()
        w[2] = self.arm_to_world(user_mm_to_arm_m(0, 0, float(st['z_mm'])))[2]
        p_arm = self.world_to_arm(w)
        seg, e = self._vertical_goto(self.q_cmd, p_arm, st.get('name', 'hole'))
        if seg is None:
            self._abort(e); return
        p.resetBasePositionAndOrientation(self.marker, w.tolist(), [0, 0, 0, 1])
        self.get_logger().info('    월드 %s -> 팔 %s' % (w.round(4), p_arm.round(4)))
        self._play_seg(seg)

    def _start_probe(self, st):
        """2차 위치에서 probe_extra_mm 만큼 더 하강. 재생 중 부하 감시."""
        p_from = fk_pos(self.q_cmd)
        p_to = p_from - np.array([0.0, 0.0, self.probe_extra / 1000.0])
        seg, mode, msg = self._descend(self.q_cmd, p_from, p_to, speed=self.probe_speed)
        if seg is None:
            self._abort('탐침 하강 경로 실패: %s' % msg); return
        self.get_logger().info('    %s, %d 프레임, 부하 %.0f%% 감시 (ID %s)'
                               % (mode, len(seg), self.probe_load, self.probe_ids))
        st['_poll_t'] = 0.0
        self._probe_hit = None
        self.traj = seg
        self.idx = 0
        self._play_t0 = None
        st['kind'] = 'probe_run'

    # ================================================== 메인 루프
    def tick(self):
        now = time.time()

        if self.rail is not None:
            self.rail_m = float(np.clip(self.rail.position_m, RAIL_MIN_M, RAIL_MAX_M))
            m = Float64(); m.data = self.rail_m; self.rail_pub.publish(m)

        if self._click(self.b_stop, 'stop'):
            if self.rail:
                self.rail.suction(False)
            self._abort('EMERGENCY STOP')
        if self._click(self.b_auto, 'auto'):
            self.auto_run = not self.auto_run
            self.get_logger().info('AUTO = %s' % self.auto_run)
        if self._click(self.b_von, 'von') and self.rail:
            self.rail.suction(True); self.get_logger().info('수동 흡착 ON')
        if self._click(self.b_voff, 'voff') and self.rail:
            self.rail.suction(False); self.get_logger().info('수동 흡착 OFF')
        if self._click(self.b_home, 'home') and not self.busy:
            self._start([self._S('goto_q', target=self.home_pose, name='초기위치 복귀')], 'HOME')
        if self._click(self.b_rzero, 'rzero') and self.rail:
            self.rail.zero()
            self.get_logger().info('레일 현재 위치를 0 으로 설정')
        if self._click(self.b_appr, 'appr'):
            self._await_approve = False

        if self.busy and self._await_approve:
            self._render(); return

        if self.busy and self.step is None:
            self._next_step()

        st = self.step
        if st is not None:
            k = st['kind']
            if k == 'traj':
                self._play(now)
            elif k == 'probe_run':
                self._play_probe(now, st)
            elif k == 'wait':
                if st['_t0'] is None:
                    st['_t0'] = now
                elif (now - st['_t0']) >= float(st['sec']):
                    st['_done'] = True
            elif k == 'settle':
                if self._check_settled(now, st):
                    st['_done'] = True
            elif k == 'rail_abs':
                if st['_t0'] is None:
                    st['_t0'] = now
                if self.rail is None:
                    st['_done'] = True
                elif self.rail.last_event == 'L':
                    self._abort('레일이 명령을 거부했습니다 (원점 미설정 또는 '
                                '소프트리미트 초과: 목표 %.0f mm)' % st.get('_target_mm', 0.0))
                elif self.rail.moving or (now - st['_t0']) >= self.rail_reject_timeout:
                    st['_done'] = True
            elif k == 'rail_wait':
                if st['_t0'] is None:
                    st['_t0'] = now
                if self.rail is None:
                    st['_done'] = True
                elif self.rail.last_event == 'L':
                    self._abort('레일 이동 거부 (@L)')
                elif (now - st['_t0']) > 1.0 and not self.rail.moving:
                    st['_done'] = True
                    self.get_logger().info('    레일 도착 %.3f m' % self.rail_m)
                elif (now - st['_t0']) > 60.0:
                    self._abort('레일 이동 시간초과')

            if st.get('_done'):
                self.step = None
                if not self.auto_run and self.queue:
                    self._await_approve = True

        self._render()

    def _read_probe_load(self):
        """ID5/6 중 최대 |부하%|. 읽기 실패 시 None."""
        if self.bridge is None:
            return None
        try:
            stt = self.bridge.read_status()
        except Exception:
            return None
        vals = []
        for i in self.probe_ids:
            if i in stt and stt[i][0] is not None:
                vals.append(abs(float(stt[i][0])))
        return max(vals) if vals else None

    def _play_probe(self, now, st):
        """탐침 하강 재생. 부하 임계 도달 시 즉시 정지."""
        if self.traj is None:
            st['_done'] = True
            return
        if self._play_t0 is None:
            self._play_t0 = now - self.idx / self.sim_hz

        if (now - st.get('_poll_t', 0.0)) >= self.probe_poll:
            st['_poll_t'] = now
            load = self._read_probe_load()
            if load is not None:
                st['_max_load'] = max(st.get('_max_load', 0.0), load)
                if (now - st.get('_log_t', 0.0)) >= 0.3:
                    st['_log_t'] = now
                    self.get_logger().info('      부하 %.1f%% / 임계 %.1f%%  (z=%.4f m)'
                                           % (load, self.probe_load, fk_pos(self.q_cmd)[2]))
            if load is not None and load >= self.probe_load:
                self._probe_hit = load
                self.get_logger().info('    부하 %.1f%% 감지 -> 하강 정지 (z=%.4f m, 팔기준)'
                                       % (load, fk_pos(self.q_cmd)[2]))
                if self.bridge:
                    self.bridge.command_joints(list(self.q_cmd) + [self.q5])
                self.traj = None
                self._play_t0 = None
                st['_done'] = True
                return

        last = len(self.traj) - 1
        tgt = int(round((now - self._play_t0) * self.sim_hz))
        self.idx = max(self.idx, min(tgt, last))
        self.q_cmd = self.traj[self.idx]
        if self.bridge and (now - self._last_cmd_t) >= (1.0 / self.robot_cmd_hz):
            self._last_cmd_t = now
            self.bridge.command_joints(list(self.q_cmd) + [self.q5])
        if self.idx >= last:
            if self.bridge:
                self.bridge.command_joints(list(self.q_cmd) + [self.q5])
            self.traj = None
            self._play_t0 = None
            st['_done'] = True
            if self._probe_hit is None:
                self.get_logger().warn('    %.0fmm 다 내려갔으나 부하 %.1f%% 미도달 '
                                       '(관측 최대 %.1f%%). 임계를 낮추거나 '
                                       '하강 거리를 늘리세요.'
                                       % (self.probe_extra, self.probe_load,
                                          st.get('_max_load', 0.0)))

    def _check_settled(self, now, st):
        if self.bridge is None:
            return True
        if self._settle_t0 is None:
            self._settle_t0 = now; self._settle_tick = 0
        self._settle_tick += 1
        if self._settle_tick < int(self.sim_hz * 0.1):
            return False
        self._settle_tick = 0
        want_q5 = bool(st.get('check_q5', False))
        try:
            qr = np.array(self.bridge.read_joints(), float).ravel()
            e_arm = float(np.max(np.abs(wrap_to_pi(qr[:N_JOINTS] - self.q_cmd))))
            e_q5 = 0.0
            if want_q5 and qr.size > N_JOINTS:
                e_q5 = float(abs(wrap_to_pi(qr[N_JOINTS] - self.q5)))
            if e_arm <= self.settle_tol and ((not want_q5) or e_q5 <= self.q5_settle_tol):
                self._settle_t0 = None
                self.get_logger().info('    도달 확인 (팔 %.2f도%s)'
                                       % (np.degrees(e_arm),
                                          ', 그리퍼 %.2f도' % np.degrees(e_q5) if want_q5 else ''))
                return True
            if (now - self._settle_t0) > self.settle_timeout:
                self.get_logger().warn('    도달 대기 시간초과 (팔 %.2f도)' % np.degrees(e_arm))
                self._settle_t0 = None
                return True
        except Exception as ex:
            self.get_logger().warn('    관절 읽기 실패: %s' % ex)
            self._settle_t0 = None
            return True
        return False

    def _play(self, now):
        if self.traj is None:
            self.step['_done'] = True
            return
        if self._play_t0 is None:
            self._play_t0 = now - self.idx / self.sim_hz
        last = len(self.traj) - 1
        tgt = int(round((now - self._play_t0) * self.sim_hz))
        self.idx = max(self.idx, min(tgt, last))
        self.q_cmd = self.traj[self.idx]
        if self.bridge and (now - self._last_cmd_t) >= (1.0 / self.robot_cmd_hz):
            self._last_cmd_t = now
            self.bridge.command_joints(list(self.q_cmd) + [self.q5])
        if self.idx >= last:
            if self.bridge:
                self.bridge.command_joints(list(self.q_cmd) + [self.q5])
            self.traj = None
            self._play_t0 = None
            self.step['_done'] = True

    # ================================================== 표시
    def _render(self):
        self._render_tick += 1
        if self._render_tick < max(1, int(round(self.sim_hz / self.render_hz))):
            return
        self._render_tick = 0
        self._apply_sim(self.q_cmd)
        p.stepSimulation()

        lines = [
            ('phase', 'PHASE: %s  %s' % (self.phase, '[AUTO]' if self.auto_run else '[MANUAL]'),
             [1, 1, 0.3], 1.35),
            ('rail', 'RAIL: %.3f m%s   SUCTION: %s   q5: %.0f deg' %
             (self.rail_m,
              '' if (self.rail is None or self.rail.zero_set) else '  [원점 미설정]',
              ('ON' if (self.rail and self.rail.suction_on) else 'OFF'),
              np.degrees(self.q5)),
             [0.3, 1, 0.4] if (self.rail is None or self.rail.zero_set)
             else [1.0, 0.7, 0.2], 1.20),
            ('step', 'STEP: %s   남은 %d' %
             ((self.step or {}).get('name', '-'), len(self.queue)), [0.8, 0.8, 1], 1.05),
        ]
        for key, txt, col, z in lines:
            if self._txt.get(key) is not None:
                p.removeUserDebugItem(self._txt[key])
            self._txt[key] = p.addUserDebugText(txt, [0.0, 0.0, z],
                                                textColorRGB=col, textSize=1.1)

    def _click(self, b, key):
        v = p.readUserDebugParameter(b)
        if v > self._cl[key]:
            self._cl[key] = v
            return True
        return False

    def _apply_sim(self, q):
        if self.rail_jid is not None:
            p.resetJointState(self.robot, self.rail_jid, float(self.rail_m))
        full = list(q) + [self.q5] + [0.0] * (len(self.rev) - N_JOINTS - 1)
        for k, jid in enumerate(self.rev):
            p.resetJointState(self.robot, jid, float(full[k]))


def main(args=None):
    rclpy.init(args=args)
    node = SuctionSeqNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.bridge:
            node.bridge.emergency_stop(); node.bridge.close()
        if node.rail:
            try:
                node.rail.suction(False)
            except Exception:
                pass
            node.rail.close()
        if getattr(node, 'checker', None):
            node.checker.close()
        try:
            p.disconnect()
        except Exception:
            pass
        node.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    main()
