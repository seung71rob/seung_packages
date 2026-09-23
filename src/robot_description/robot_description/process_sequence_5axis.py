"""
robot_description / process_sequence_5axis.py
==================================================
5축 팔 + 레일 + 흡착 : 전체 공정 시퀀스 (suction 전용) v3 fast-sequential

공정
----
  [대기]
    /detected_object 로 top view 판정 수신
      judge == NG  -> 불량 경로
      judge == OK  -> 정상 경로 (다면 검사)

  [공통] 픽
    1차점 접근 -> dwell -> 2차점 수직 하강 -> press_mm 추가 하강
    -> 흡착 ON -> 복귀점(retreat) 까지 수직 유지 상승

  [불량 경로]  REJECT
    이동자세(reject_pose_deg, 0,50,-50,-95) 로 변경
    -> 레일 reject_rail_mm(700) 로 이동, 도착 후 1초 정지
    -> 배출 좌표(-0.35,0,0.05) 로 수직 유지 이동 -> 2초 대기 -> 흡착 OFF
    -> 레일 원위치 -> 초기위치 복귀

  [정상 경로]  다면 검사  (B_arm 과 핸드셰이크)
    1) bottom view (0,15,-30,-80) 로 이동
       -> CHECK_READY / GO_BOTTOM / INSPECT_BOTTOM
    2) back view (0,0,-90,90) 로 이동, B_arm back view 이동
       -> q5 8자세 (0, 45, 90, 135, 180, -45, -90, -135)
          각각 INSPECT_BACK repeat_index=1..8
    3) flash view (0,-10,-50,-95) 로 이동, B_arm (0,0,50,0,-90,180)
       -> q5 8자세로 INSPECT_FLASH repeat_index=1..8
    4) 검사 판정은 /baseplate_side/view_result 를 읽는다.
       한 번이라도 NG (2초 이상 지속) 검출 -> 즉시 ABORT -> 불량 배출로 분기
    5) 전부 OK -> FINISH -> 정상 배출 (-0.35, 0, 0.05) -> 초기위치 복귀

토픽
----
  구독
    /detected_object                     top view 결과 (String)
    /baseplate_side/view_result          다면 검사 결과 (String, JSON)
    /multi_arm/b_arm_status              B_arm 상태 (String, JSON)
    /process_cmd                         start / stop / reset / auto on|off
  발행
    /multi_arm/inspection_command        B_arm 동기 명령 (String, JSON)
    /process_state                       현재 단계 (String)
    /rail_position                       레일 위치 (Float64)
"""

import json
import os
import time
import uuid
import xml.etree.ElementTree as ET
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, String
import pybullet as p
import pybullet_data

from .kinematics_ee_5axis import (fk, fk_pos, ik, ik_fix, ik_fix_candidates,
                                  N_JOINTS, wrap_to_pi)
from .trajectory_5axis import plan_joint_path
from .path_collision_guard_ee_5axis import CollisionChecker
from .rail_serial_5axis import RailSerial, RAIL_MIN_M, RAIL_MAX_M
from .vertical_path_5axis import (plan_descent_ends_retimed, plan_joint_descent_retimed,
                                  plan_sync_joint_move, retime_path, joint_speed_limits,
                                  path_joint_speed, fix_reach_report, ik_level_soft,
                                  level_error)

URDF_NAME = 'robot_arm_5axis_rail_suction.urdf'
ARM_URDF_NAME = 'robot_arm_5axis_suction.urdf'
RAIL_JOINT_NAME = 'joint_rail'

# ---- top view A3 -> 로봇 사용자 좌표 ----
A3_W_MM, A3_H_MM = 420.0, 297.0
A3_LT_ROBOT_X_MM = 482.0
A3_LT_ROBOT_Y_MM = 214.55
USER_X_SIGN, USER_Y_SIGN, USER_Z_SIGN = -1.0, +1.0, +1.0


def user_mm_to_arm_m(x_mm, y_mm, z_mm):
    return np.array([USER_X_SIGN * float(x_mm) / 1000.0,
                     USER_Y_SIGN * float(y_mm) / 1000.0,
                     USER_Z_SIGN * float(z_mm) / 1000.0], float)


class ProcessNode(Node):

    # ================================================== 초기화
    def __init__(self):
        super().__init__('process_sequence_5axis')

        d = self.declare_parameter
        d('pkg_dir', os.path.expanduser('~/robot_sim/src/robot_description'))
        d('mesh_dir', os.path.expanduser('~/robot_sim/src/robot_description/meshes_robot_5axis'))
        d('use_robot', False)
        d('use_rail', False)
        d('rail_port', '/dev/ttyACM0')
        d('rail_baud', 115200)
        d('dxl_port', '/dev/ttyUSB1')
        d('check_collision', True)
        d('sim_hz', 240.0)
        d('robot_cmd_hz', 50.0)
        d('render_hz', 60.0)
        d('speed_deg_per_s', 60.0)
        d('auto_run', False)
        # --- 픽 ---
        d('vision_z_mm', 10.0)
        d('vision_approach_mm', 80.0)
        d('vision_rail_ref_m', 0.0)
        d('press_mm', 5.0)
        d('dwell_sec', 0.5)
        d('suction_dwell', 0.5)
        d('descend_speed', 0.05)
        d('strict_ends', 'both')
        d('level_tol_deg', 25.0)
        d('limit_joint_speed', True)
        d('speed_margin', 0.9)
        d('retreat_arm_xyz', [-0.35, 0.0, 0.15])
        # --- 공정 자세 (5축 관절각, 도) ---
        d('reject_pose_deg', [0.0, 50.0, -50.0, -95.0])
        d('bottom_view_deg', [0.0, 15.0, -25.0, -80.0])
        d('back_view_deg', [0.0, 0.0, -85.0, 85.0])
        d('flash_view_deg', [0.0, 5.0, -60.0, -105.0])
        # back -> flash 경유점. 직행하면 경로 중간(50% 지점)에서 흡착부가
        # link4 를 관통한다(link4-suction 거리 0.000 m). 경유점을 거치면
        # 양 구간 모두 최소 0.087 m 를 유지한다.
        d('back_to_flash_via_deg', [0.0, 25.0, -90.0, 10.0])   # ★ flash view 추가
        d('home_pose_deg', [0.0, 0.0, 0.0, 0.0])
        # --- 배출 ---
        d('reject_rail_mm', 700.0)
        d('reject_rail_pause', 0.0)
        d('drop_settle_sec', 2.0)
        d('reject_drop_arm', [-0.35, 0.0, 0.05])
        d('good_drop_arm', [-0.35, 0.0, 0.05])
        # --- 레일 도착 판정 ---
        d('rail_tol_mm', 2.0)
        d('rail_timeout', 60.0)
        d('rail_require_zero', True)
        d('rail_return', True)
        d('rail_home_mm', 0.0)
        # --- 다면 검사 ---
        d('view_wait_sec', 3.0)
        d('view_result_timeout', 8.0)
        # q5 8자세: 0, +45, +90, +135, +180, -45, -90, -135  (repeat_index 1..8)
        d('q5_cw_deg', [45.0, 90.0, 135.0, 180.0])
        d('q5_ccw_deg', [-45.0, -90.0, -135.0])
        d('side_result_topic', '/baseplate_side/view_result')
        d('barm_cmd_topic', '/multi_arm/inspection_command')
        d('barm_status_topic', '/multi_arm/b_arm_status')
        d('barm_timeout', 30.0)
        d('require_barm', True)
        d('barm_abort_wait', 1.5)
        d('flash_settle_sec', 0.0)
        # --- 안전 ---
        d('settle_tol_deg', 1.5)
        d('settle_timeout', 5.0)
        d('q5_settle_tol_deg', 2.0)
        d('settle_poll_sec', 0.1)
        d('settle_confirm_samples', 2)
        d('collision_stride', 8)
        d('hold_level_warn_deg', 30.0)
        d('abort_release_suction', False)

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

        self.vision_z_mm = float(g('vision_z_mm').value)
        self.vision_approach_mm = float(g('vision_approach_mm').value)
        self.vision_rail_ref = float(g('vision_rail_ref_m').value)
        self.press_mm = float(g('press_mm').value)
        self.dwell_sec = float(g('dwell_sec').value)
        self.suction_dwell = float(g('suction_dwell').value)
        self.descend_speed = float(g('descend_speed').value)
        self.strict_ends = str(g('strict_ends').value).lower()
        self.level_tol = np.radians(float(g('level_tol_deg').value))
        self.limit_joint_speed = bool(g('limit_joint_speed').value)
        self.speed_margin = float(g('speed_margin').value)
        self.retreat_arm = np.array([float(v) for v in g('retreat_arm_xyz').value], float)

        self.reject_pose = np.radians(np.array([float(v) for v in g('reject_pose_deg').value], float))
        self.bottom_view = np.radians(np.array([float(v) for v in g('bottom_view_deg').value], float))
        self.back_view = np.radians(np.array([float(v) for v in g('back_view_deg').value], float))
        self.flash_view = np.radians(np.array([float(v) for v in g('flash_view_deg').value], float))
        self.bf_via = np.radians(np.array([float(v) for v in g('back_to_flash_via_deg').value], float))
        self.home_pose = np.radians(np.array([float(v) for v in g('home_pose_deg').value], float))

        self.reject_rail_mm = float(g('reject_rail_mm').value)
        self.reject_rail_pause = float(g('reject_rail_pause').value)
        self.drop_settle = float(g('drop_settle_sec').value)
        self.reject_drop = np.array([float(v) for v in g('reject_drop_arm').value], float)
        self.good_drop = np.array([float(v) for v in g('good_drop_arm').value], float)

        self.rail_tol_mm = float(g('rail_tol_mm').value)
        self.rail_timeout = float(g('rail_timeout').value)
        self.rail_require_zero = bool(g('rail_require_zero').value)
        self.rail_return = bool(g('rail_return').value)
        self.rail_home_mm = float(g('rail_home_mm').value)

        self.view_wait = float(g('view_wait_sec').value)
        self.view_timeout = float(g('view_result_timeout').value)
        self.q5_cw = [float(v) for v in g('q5_cw_deg').value]
        self.q5_ccw = [float(v) for v in g('q5_ccw_deg').value]
        self.barm_timeout = float(g('barm_timeout').value)
        self.require_barm = bool(g('require_barm').value)
        self.barm_abort_wait = float(g('barm_abort_wait').value)
        self.flash_settle = max(0.0, float(g('flash_settle_sec').value))
        self.settle_tol = np.radians(float(g('settle_tol_deg').value))
        self.settle_timeout = float(g('settle_timeout').value)
        self.q5_settle_tol = np.radians(float(g('q5_settle_tol_deg').value))
        self.settle_poll = max(0.02, float(g('settle_poll_sec').value))
        self.settle_confirm_samples = max(1, int(g('settle_confirm_samples').value))
        self.collision_stride = max(1, int(g('collision_stride').value))
        self.hold_level_warn = np.radians(float(g('hold_level_warn_deg').value))
        self.abort_release_suction = bool(g('abort_release_suction').value)

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
        self._cl = {k: 0 for k in ('stop', 'appr', 'auto', 'von', 'voff', 'home')}
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
                self.rail._moving = False
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
        self.defect_seen = False
        self._play_t0 = None
        self._last_cmd_t = 0.0
        self._render_tick = 0
        self._t_wait = None
        self._settle_t0 = None
        self._settled = False
        self._settle_last_read = 0.0
        self.view_results = deque()
        self.barm_status = deque()
        self._await_approve = False
        self._rail_target_mm = None
        self._holding = False

        # ---------- 토픽 ----------
        self.barm_pub = self.create_publisher(String, str(g('barm_cmd_topic').value), 10)
        self.state_pub = self.create_publisher(String, 'process_state', 10)
        self.rail_pub = self.create_publisher(Float64, 'rail_position', 10)

        self.create_subscription(String, 'detected_object', self.on_top_view, 10)
        self.create_subscription(String, str(g('side_result_topic').value),
                                 self.on_view_result, 10)
        self.create_subscription(String, str(g('barm_status_topic').value),
                                 self.on_barm_status, 10)
        self.create_subscription(String, 'process_cmd', self.on_process_cmd, 10)

        self._apply_sim(self.q_cmd)
        self.create_timer(1.0 / self.sim_hz, self.tick)
        self.get_logger().info('Ready (process_sequence v3, use_robot=%s, use_rail=%s, auto=%s)'
                               % (self.use_robot, self.rail is not None, self.auto_run))

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
        rx = A3_LT_ROBOT_X_MM - float(ay)
        ry = float(ax) - A3_LT_ROBOT_Y_MM
        p_arm = user_mm_to_arm_m(rx, ry, z_mm)
        return p_arm + self.rail_origin + np.array([0.0, self.vision_rail_ref, 0.0])

    # ================================================== 경로 조각
    def _path_ok(self, seg):
        if self.checker is None:
            return True, ''
        n = len(seg)
        if n == 0:
            return True, ''
        idxs = list(range(0, n, self.collision_stride))
        if idxs[-1] != n - 1:
            idxs.append(n - 1)
        for i in idxs:
            ok, why = self.checker.check_config(seg[i])
            if not ok:
                return False, '%s (프레임 %d/%d)' % (why, i, n)
        return True, ''

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
        ok, why = self._path_ok(seg)
        if not ok:
            return None, '%s: 경로 충돌 %s' % (tag, why)
        return seg, ''

    def _descend(self, q_from, p_from, p_to):
        seg, T, mx, msg = plan_descent_ends_retimed(
            q_from, p_from, p_to, hz=self.sim_hz, lin_speed=self.descend_speed,
            mode=self.strict_ends, level_tol=self.level_tol,
            vmax=self.vmax if self.limit_joint_speed else None, margin=self.speed_margin)
        if seg is not None:
            return seg, '직선', ''
        seg, T, mx, msg2 = plan_joint_descent_retimed(
            q_from, p_from, p_to, hz=self.sim_hz, lin_speed=self.descend_speed,
            vmax=self.vmax if self.limit_joint_speed else None, margin=self.speed_margin)
        if seg is None:
            return None, '', msg2 or msg
        return seg, '관절공간', ''

    def _vertical_goto(self, q_from, p_arm_target, tag):
        p_arm_target = np.asarray(p_arm_target, float).ravel()[:3]
        cands = ik_fix_candidates(p_arm_target, q0=q_from)
        q_list = []
        if cands:
            cands.sort(key=lambda c: np.linalg.norm(wrap_to_pi(c[0] - q_from)))
            q_list = [c[0] for c in cands]
        else:
            q_s, e, lv = ik_level_soft(p_arm_target, q_from)
            if e > 1e-4:
                return None, '%s: 도달 불가 (%s)' % (tag, fix_reach_report(p_arm_target)['msg'])
            self.get_logger().warn('%s: 수직에서 %.1f도 기울어집니다.' % (tag, np.degrees(lv)))
            q_list = [q_s]

        last = ''
        for gi, q_t in enumerate(q_list):
            seg, e = self._sync_to(q_from, q_t, tag)
            if seg is not None:
                if gi > 0:
                    self.get_logger().info('%s: 충돌 없는 해 채택 (%d/%d)'
                                           % (tag, gi + 1, len(q_list)))
                return seg, ''
            last = e
        rep = fix_reach_report(p_arm_target)
        return None, '%s (후보 %d개 모두 실패). %s' % (last or '원인 불명',
                                                      len(q_list), rep['msg'])

    # ================================================== 스텝 생성
    def _S(self, kind, **kw):
        d = dict(kind=kind)
        d.update(kw)
        return d

    def build_pick(self, w_appr, w_pick):
        p_high = self.world_to_arm(w_appr)
        p_low = self.world_to_arm(w_pick)
        p_press = p_low - np.array([0.0, 0.0, self.press_mm / 1000.0])

        cands = ik_fix_candidates(p_high, q0=self.q_cmd)
        if not cands:
            q_s, e, lv = ik_level_soft(p_high, self.q_cmd)
            if e > 1e-4 or lv > self.level_tol:
                return None, '1차점 도달 불가: %s' % fix_reach_report(p_high)['msg']
            self.get_logger().warn('1차점이 수직에서 %.1f도 기울어집니다.' % np.degrees(lv))
            cands = [(q_s, e)]
        safe = [q for q, _ in cands if self.checker is None or self.checker.check_config(q)[0]]
        if not safe:
            return None, '1차점 자세 충돌'
        safe.sort(key=lambda q: np.linalg.norm(wrap_to_pi(q - self.q_cmd)))

        last = ''
        for q_a in safe:
            seg1, T1, _n = plan_joint_path(self.q_cmd, q_a, hz=self.sim_hz,
                                           speed_deg_per_s=self.speed)
            if self.vmax is not None and self.limit_joint_speed:
                o, _T, _i = retime_path(seg1, self.vmax, self.sim_hz, margin=self.speed_margin)
                if o is not None:
                    seg1 = o
            ok1, why1 = self._path_ok(seg1)
            if not ok1:
                last = '접근 경로 충돌 %s' % why1; continue

            seg2, m2, e2 = self._descend(q_a, p_high, p_low)
            if seg2 is None:
                last = '하강: %s' % e2; continue
            seg3, m3, e3 = self._descend(seg2[-1], p_low, p_press)
            if seg3 is None:
                last = '압착: %s' % e3; continue

            seg4, e4 = self._vertical_goto(seg3[-1], self.retreat_arm, '복귀')
            if seg4 is None:
                last = e4; continue

            steps = [
                self._S('traj', traj=seg1, name='1차점 접근'),
                self._S('wait', sec=self.dwell_sec, name='1차점 정지', settle=True),
                self._S('traj', traj=seg2, name='2차점 하강(%s)' % m2),
                self._S('traj', traj=seg3, name='압착 %.1fmm(%s)' % (self.press_mm, m3)),
                self._S('settle', name='압착점 도달 확인'),
                self._S('suction', on=True, name='흡착 ON'),
                self._S('wait', sec=self.suction_dwell, name='흡착 유지'),
                self._S('traj', traj=seg4, name='복귀점 이동'),
            ]
            return steps, ''
        return None, last or '원인 불명'

    def _warn_if_tilted(self, q, tag):
        lv = float(level_error(np.asarray(q, float).ravel()[:N_JOINTS]))
        if lv > self.hold_level_warn:
            self.get_logger().warn(
                '%s 자세는 수직에서 %.0f도 기울어져 있습니다. '
                '흡착 부품이 미끄러질 수 있습니다 (경고 기준 %.0f도).'
                % (tag, np.degrees(lv), np.degrees(self.hold_level_warn)))
        return lv

    def _build_drop_tail(self, q_at_drop_start, drop_arm, tag):
        rep = fix_reach_report(np.asarray(drop_arm, float))
        if not rep['ok']:
            return None, None, '%s 도달 불가: %s' % (tag, rep['msg'])

        seg_d, e = self._vertical_goto(q_at_drop_start, drop_arm, tag)
        if seg_d is None:
            return None, None, e
        seg_h, e = self._sync_to(seg_d[-1], self.home_pose, '초기위치')
        if seg_h is None:
            return None, None, e

        steps = [
            self._S('traj', traj=seg_d, name=tag),
            self._S('settle', name='배출 좌표 도달 확인'),
            self._S('wait', sec=self.drop_settle,
                    name='배출 전 %.1f초 대기' % self.drop_settle),
            self._S('suction', on=False, name='흡착 OFF'),
            self._S('wait', sec=0.3, name='분리 대기'),
            self._S('traj', traj=seg_h, name='초기위치 복귀'),
        ]
        return steps, seg_h[-1], ''

    def build_reject(self, q_from, lead_wait=0.0):
        """
        불량 배출: (back 출발이면 초기위치 경유) -> 이동자세 -> 레일
                   -> 배출좌표 -> 도착확인 -> 2초 대기 -> 흡착 OFF.

        back view(0,0,-90,95)는 말단이 위로 꺾여 있어 이동자세로 직행하면
        중간 경로가 위험하다. 그때만 초기위치를 한 번 거친다.
        나머지 출발 자세는 직행한다.
        """
        steps = []
        if lead_wait > 0.0:
            steps.append(self._S('wait', sec=float(lead_wait),
                                 name='B_arm 회피 대기 %.1fs' % lead_wait))

        # back view 에서 출발할 때만 초기위치를 거친다.
        #   back 자세(0,0,-90,95)는 말단이 위로 꺾여 있어 이동자세로 직행하면
        #   중간 경로가 위험하다. 곧게 편 자세를 한 번 지나가면 사라진다.
        #   나머지 자세(복귀점, bottom, flash 등)는 직행해도 안전하므로
        #   불필요한 왕복을 하지 않는다.
        from_back = bool(np.max(np.abs(wrap_to_pi(
            np.asarray(q_from, float).ravel()[:N_JOINTS] - self.back_view)))
            < np.radians(5.0))

        q_lead = np.asarray(q_from, float).ravel()[:N_JOINTS]
        seg_h = None
        if from_back:
            seg_h, e = self._sync_to(q_lead, self.home_pose, '초기위치')
            if seg_h is None:
                return None, e
            q_lead = self.home_pose
            self.get_logger().info('  back view 에서 출발 -> 초기위치를 경유합니다.')

        seg_a, e = self._sync_to(q_lead, self.reject_pose, '이동자세')
        if seg_a is None:
            return None, e
        self._warn_if_tilted(self.reject_pose, '불량 이동')

        tail, _q_end, e = self._build_drop_tail(seg_a[-1],
                                                self.reject_drop, '불량 배출 좌표')
        if tail is None:
            return None, e

        if seg_h is not None:
            steps += [
                self._S('traj', traj=seg_h, name='초기위치 경유 (back 출발)'),
                self._S('settle', name='초기위치 도달 확인'),
            ]
        steps += [
            self._S('traj', traj=seg_a, name='이동자세'),
            self._S('settle', name='이동자세 도달 확인'),
            self._S('rail_abs', mm=self.reject_rail_mm,
                    name='레일 %.0fmm' % self.reject_rail_mm),
            self._S('rail_wait', name='레일 도착 대기'),
            self._S('wait', sec=self.reject_rail_pause,
                    name='레일 정지 %.1fs' % self.reject_rail_pause),
        ]
        steps += tail

        if self.rail_return:
            steps += [
                self._S('rail_abs', mm=self.rail_home_mm,
                        name='레일 복귀 %.0fmm' % self.rail_home_mm),
                self._S('rail_wait', name='레일 복귀 대기'),
            ]
        return steps, ''

    def build_good_drop(self, q_from):
        """정상 배출 (-0.35, 0, 0.05) 수직 유지."""
        tail, _q_end, e = self._build_drop_tail(q_from, self.good_drop, '정상 배출 좌표')
        if tail is None:
            return None, e
        return tail, ''

    def build_multiview(self, q_from):
        """
        다면 검사. B_arm 과 핸드셰이크.

        순서:
          1) CHECK_READY  -> B_arm READY 대기
          2) bottom view 이동 (5축)
          3) GO_BOTTOM    -> B_arm ARRIVED 대기
          4) INSPECT_BOTTOM (repeat_index=1) -> INSPECTION_COMPLETE 대기
          5) back view 이동 (5축)
          6) GO_BACK      -> B_arm ARRIVED 대기
          7) q5 8자세 각각 INSPECT_BACK repeat_index=1..8
          8) flash view 이동 (5축)
          9) GO_FLASH     -> B_arm ARRIVED 대기
         10) q5 8자세 각각 INSPECT_FLASH repeat_index=1..8
         11) FINISH -> B_arm SEQUENCE_COMPLETE 대기
        """
        # --- 이동 경로 미리 생성 ---
        seg_b, e = self._sync_to(q_from, self.bottom_view, 'bottom view')
        if seg_b is None:
            return None, e
        seg_k, e = self._sync_to(self.bottom_view, self.back_view, 'back view')
        if seg_k is None:
            return None, e
        # back -> flash 는 경유점을 거친다. 직행하면 경로 중간에서
        # 흡착부가 link4 를 관통한다(link4-suction 거리 0.000 m).
        seg_v, e = self._sync_to(self.back_view, self.bf_via, 'flash 경유점')
        if seg_v is None:
            return None, e
        seg_f, e = self._sync_to(self.bf_via, self.flash_view, 'flash view')
        if seg_f is None:
            return None, e

        steps = []

        # 1) READY 확인
        steps.append(self._S('barm', cmd='CHECK_READY',
                             wait=['READY'], name='B_arm READY 확인'))

        # 2~3) bottom view
        steps.append(self._S('traj', traj=seg_b, name='bottom view 자세'))
        steps.append(self._S('settle', name='bottom view 도달 확인'))
        steps.append(self._S('barm', cmd='GO_BOTTOM', a_arrived=True,
                             wait=['ARRIVED'], name='B_arm bottom 이동'))

        # 4) bottom 검사 (1회)
        steps.append(self._S('barm', cmd='INSPECT_BOTTOM', a_arrived=True,
                             repeat_index=1, view_key='B_BOTTOM_VIEW',
                             wait=['INSPECTION_COMPLETE'],
                             watch_result=True,
                             name='bottom 검사'))

        # 5~6) back view
        steps.append(self._S('traj', traj=seg_k, name='back view 자세'))
        steps.append(self._S('settle', name='back view 도달 확인'))
        steps.append(self._S('barm', cmd='GO_BACK', a_arrived=True,
                             wait=['ARRIVED'], name='B_arm back 이동'))

        # 7) back 검사 8회 (q5 8자세)
        # q5 순서: 0, +45, +90, +135, +180, -45, -90, -135  (요구사항 대로 cw 4회 후 ccw 4회)
        angles = [0.0] + list(self.q5_cw) + list(self.q5_ccw)
        if len(angles) != 8:
            self.get_logger().warn('back 검사는 8회여야 하는데 q5 자세가 %d개입니다. '
                                   'q5_cw_deg / q5_ccw_deg 를 확인하세요.' % len(angles))
        for i, a in enumerate(angles, start=1):
            steps.append(self._S('q5', deg=a, name='그리퍼 %.0f도 (back %d/8)' % (a, i)))
            steps.append(self._S('settle', check_q5=True,
                                 name='그리퍼 도달 대기 (ID7 확인)'))
            steps.append(self._S('barm', cmd='INSPECT_BACK', a_arrived=True,
                                 repeat_index=i,
                                 view_key='B_BACK_VIEW_%02d' % i,
                                 wait=['INSPECTION_COMPLETE'],
                                 watch_result=True,
                                 name='back 검사 %d/8 (q5 %.0f도)' % (i, a)))

        # 8~9) flash view
        steps.append(self._S('q5', deg=0.0, name='그리퍼 원위치 (flash 전)'))
        steps.append(self._S('settle', check_q5=True, name='그리퍼 도달 대기'))
        steps.append(self._S('traj', traj=seg_v,
                             name='flash 경유점 %s'
                                  % np.degrees(self.bf_via).round(0).tolist()))
        steps.append(self._S('settle', name='경유점 도달 확인'))
        steps.append(self._S('traj', traj=seg_f, name='flash view 자세'))
        steps.append(self._S('settle', name='flash view 도달 확인'))
        steps.append(self._S('wait', sec=self.flash_settle,
                             name='flash view %.1f초 대기' % self.flash_settle))
        steps.append(self._S('barm', cmd='GO_FLASH', a_arrived=True,
                             wait=['ARRIVED'], name='B_arm flash 이동'))

        # 10) flash 검사 8회 (q5 8자세)
        for i, a in enumerate(angles, start=1):
            steps.append(self._S('q5', deg=a, name='그리퍼 %.0f도 (flash %d/8)' % (a, i)))
            steps.append(self._S('settle', check_q5=True,
                                 name='그리퍼 도달 대기 (ID7 확인)'))
            steps.append(self._S('barm', cmd='INSPECT_FLASH', a_arrived=True,
                                 repeat_index=i,
                                 view_key='B_FLASH_VIEW_%02d' % i,
                                 wait=['INSPECTION_COMPLETE'],
                                 watch_result=True,
                                 name='flash 검사 %d/8 (q5 %.0f도)' % (i, a)))

        # 11) 종료
        steps.append(self._S('q5', deg=0.0, name='그리퍼 원위치'))
        steps.append(self._S('settle', check_q5=True, name='그리퍼 도달 대기'))
        steps.append(self._S('barm', cmd='FINISH', a_arrived=True,
                             wait=['SEQUENCE_COMPLETE'], name='B_arm 종료'))
        return steps, ''

    # ================================================== 콜백
    def on_top_view(self, msg):
        if self.busy:
            self.get_logger().warn('[TOP] 공정 진행 중이라 무시합니다.')
            return
        if self._holding:
            self.get_logger().error(
                '[TOP] 흡착이 켜져 있어 새 공정을 시작하지 않습니다. '
                'SUCTION OFF 버튼을 누른 뒤 다시 시도하세요.')
            return
        parts = [s.strip() for s in str(msg.data).split(',')]
        if len(parts) < 3:
            self.get_logger().error('[TOP] 형식 오류: %s' % msg.data)
            return
        try:
            ax, ay = float(parts[1]), float(parts[2])
        except ValueError:
            self.get_logger().error('[TOP] 숫자 변환 실패: %s' % msg.data)
            return
        judge = parts[5].upper() if len(parts) > 5 else 'OK'
        if not (0.0 <= ax <= A3_W_MM and 0.0 <= ay <= A3_H_MM):
            self.get_logger().error('[TOP] A3 영역 밖: (%.1f, %.1f)' % (ax, ay))
            return

        w_pick = self.a3_to_world(ax, ay, self.vision_z_mm)
        w_appr = self.a3_to_world(ax, ay, self.vision_z_mm + self.vision_approach_mm)
        if abs(self.rail_m - self.vision_rail_ref) > 0.005:
            self.get_logger().warn(
                '[TOP] 레일이 비전 기준(%.3f m)이 아니라 %.3f m 에 있습니다.'
                % (self.vision_rail_ref, self.rail_m))
        self.cycle_id = time.strftime('%H%M%S')
        self.defect_seen = (judge == 'NG')
        self.get_logger().info('[TOP] judge=%s  A3(%.1f, %.1f) -> 월드 %s  cycle=%s'
                               % (judge, ax, ay, w_pick.round(4), self.cycle_id))

        steps, why = self.build_pick(w_appr, w_pick)
        if steps is None:
            self.get_logger().error('[TOP] 픽 경로 생성 실패: %s' % why)
            return
        steps.append(self._S('branch', name='판정 분기'))
        self._start(steps, 'PICK')

    def on_view_result(self, msg):
        try:
            d = json.loads(msg.data)
        except Exception:
            return
        self.view_results.append(d)

    def on_barm_status(self, msg):
        try:
            self.barm_status.append(json.loads(msg.data))
        except Exception:
            pass

    def on_process_cmd(self, msg):
        c = str(msg.data).strip().lower()
        if c == 'stop':
            self._abort('사용자 정지')
        elif c == 'reset':
            self.queue.clear(); self.step = None; self.traj = None
            self.busy = False
            self._await_approve = False
            self._rail_target_mm = None
            if self._holding:
                self.get_logger().warn('reset: 흡착이 아직 ON 입니다.')
            self._set_phase('IDLE')
        elif c in ('auto on', 'auto_on'):
            self.auto_run = True; self.get_logger().info('AUTO ON')
        elif c in ('auto off', 'auto_off'):
            self.auto_run = False; self.get_logger().info('AUTO OFF')
        elif c == 'home':
            self._start([self._S('go_home', name='초기위치 복귀')], 'HOME')
        elif c in ('rail zero', 'rail_zero'):
            if self.rail is None:
                self.get_logger().warn('레일 미연결 (use_rail:=true 로 실행).')
            elif self.busy:
                self.get_logger().warn('공정 진행 중에는 원점을 잡을 수 없습니다.')
            else:
                self.rail.zero()
                self.get_logger().info('레일 현재 위치를 0 으로 설정 요청')
        elif c.startswith('rail '):
            if self.rail is None or self.busy:
                self.get_logger().warn('레일 미연결이거나 공정 진행 중입니다.')
            else:
                try:
                    mm = float(c.split(None, 1)[1])
                except ValueError:
                    self.get_logger().error("rail <mm> 형식입니다. 예: 'rail 700'")
                    return
                self._rail_start_move(mm)
        else:
            self.get_logger().warn("process_cmd: stop | reset | auto on | auto off | home"
                                   " | rail zero | rail <mm>")

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
        self._await_approve = False
        self._rail_target_mm = None
        if self.bridge:
            try:
                self.bridge.emergency_stop()
            except Exception:
                pass
        if self.rail:
            self.rail.estop()
            if self.abort_release_suction:
                self.rail.suction(False)
                self._holding = False
                self.get_logger().warn('  흡착 OFF (abort_release_suction:=true)')
            elif self._holding:
                self.get_logger().warn(
                    '  ★ 흡착은 켜진 채로 유지합니다.')
        try:
            self._barm_simple('ABORT', return_home=True)
        except Exception:
            pass
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

        if k == 'traj':
            self.traj = self.step['traj']
            self.idx = 0
            self._play_t0 = None
        elif k == 'suction':
            on = bool(self.step['on'])
            if self.rail is not None:
                self.rail.suction(on)
            self._holding = on
            self.step['_done'] = True
        elif k == 'rail_abs':
            if not self._rail_start_move(float(self.step['mm'])):
                return
            self.step['_done'] = True
        elif k == 'q5':
            self.q5 = np.radians(float(self.step['deg']))
            if self.bridge:
                self.bridge.command_joints(list(self.q_cmd) + [self.q5])
            self.step['_done'] = True
        elif k == 'barm':
            self._send_barm(self.step)
        elif k == 'goto_arm':
            seg, e = self._vertical_goto(self.q_cmd, np.asarray(self.step['target'], float),
                                         self.step.get('name', 'goto'))
            if seg is None:
                self._abort(e); return
            self.traj = seg; self.idx = 0; self._play_t0 = None
            self.step['kind'] = 'traj'
        elif k == 'go_home':
            seg, e = self._sync_to(self.q_cmd, self.home_pose, '초기위치')
            if seg is None:
                self._abort(e); return
            self.traj = seg; self.idx = 0; self._play_t0 = None
            self.step['kind'] = 'traj'
        elif k == 'branch':
            self._do_branch()
        elif k == 'settle':
            self._settle_t0 = None
            self._settled = (self.bridge is None)

    # ---------------- 레일 ----------------
    def _rail_moving(self):
        if self.rail is None:
            return False
        return self.rail.moving and not self.rail.stale(1.0)

    def _rail_start_move(self, target_mm):
        self._rail_target_mm = float(target_mm)
        if self.rail is None:
            self.get_logger().warn('  레일 미연결 -> %.0fmm 이동을 건너뜁니다.' % target_mm)
            return True

        if not (RAIL_MIN_M * 1000.0 - 1e-6) <= target_mm <= (RAIL_MAX_M * 1000.0 + 1e-6):
            self._abort('레일 목표 %.1fmm 가 스트로크(%.0f~%.0fmm) 밖'
                        % (target_mm, RAIL_MIN_M * 1000.0, RAIL_MAX_M * 1000.0))
            return False
        if not self.rail.connected:
            self._abort('레일 시리얼이 닫혀 있습니다.')
            return False
        if self.rail_require_zero and not self.rail.zero_set:
            self._abort("레일 원점 미설정. 먼저 원점을 잡으세요: "
                        "ros2 topic pub --once /process_cmd std_msgs/msg/String "
                        "\"{data: 'rail zero'}\"")
            return False
        if self._rail_moving():
            self._abort('레일이 아직 이동 중입니다.')
            return False

        self.rail.last_event = ''
        self.rail.move_abs_mm(target_mm)
        self.get_logger().info('  레일 -> %.1fmm (현재 %.1fmm)'
                               % (target_mm, self.rail.position_mm))
        return True

    def _do_branch(self):
        self.step['_done'] = True
        if self.defect_seen:
            self.get_logger().warn('판정 NG -> 불량 배출')
            steps, e = self.build_reject(self.q_cmd)
            if steps is None:
                self._abort(e); return
            self.queue.extendleft(reversed(steps))
            self._set_phase('REJECT')
        else:
            self.get_logger().info('판정 OK -> 다면 검사')
            steps, e = self.build_multiview(self.q_cmd)
            if steps is None:
                self._abort(e); return
            steps.append(self._S('branch2', name='다면 결과 분기'))
            self.queue.extendleft(reversed(steps))
            self._set_phase('MULTIVIEW')

    def _send_barm(self, st):
        """
        B_arm 에 동기 명령 발행.
        B_arm 코드가 요구하는 필드:
          - command, cycle_id, request_id (필수)
          - a_arrived (INSPECT_* / GO_* 계열에서 True)
          - repeat_index, view (INSPECT_* 계열)
        """
        pay = {
            'command': st['cmd'],
            'cycle_id': self.cycle_id,
            'request_id': uuid.uuid4().hex[:12],
        }
        if st.get('a_arrived'):
            pay['a_arrived'] = True
        if 'repeat_index' in st:
            pay['repeat_index'] = int(st['repeat_index'])
        if 'view_key' in st:
            pay['view'] = st['view_key']
        if 'return_home' in st:
            pay['return_home'] = bool(st['return_home'])

        m = String(); m.data = json.dumps(pay, ensure_ascii=False)
        self.barm_pub.publish(m)
        self.barm_status.clear()
        if st.get('watch_result'):
            self.view_results.clear()

        st['_t0'] = time.time()
        st['_request_id'] = pay['request_id']
        self.get_logger().info('  -> B_arm %s%s%s' %
                               (st['cmd'],
                                (' repeat=%d' % pay['repeat_index']) if 'repeat_index' in pay else '',
                                (' view=%s' % pay['view']) if 'view' in pay else ''))
        if not st['wait']:
            st['_done'] = True

    def _poll_barm(self, now, st):
        """
        B_arm 상태와(필요하면) 검사 결과를 기다린다.
        검사 결과에서 NG 가 확인되면 즉시 ABORT 후 불량 배출로 분기.
        """
        # 1) 검사 결과 감시 (읽기만)
        if st.get('watch_result'):
            while self.view_results:
                d = self.view_results.popleft()
                # 최신 request_id 결과만 신뢰
                if d.get('request_id') and st.get('_request_id') and \
                        d.get('request_id') != st['_request_id']:
                    continue
                judge = str(d.get('judge', '')).upper()
                self.get_logger().info('     검사 결과 %s (view=%s)' % (judge, d.get('view')))
                if judge == 'NG':
                    self.defect_seen = True
                    self.get_logger().warn('     NG 확정 -> 다면 검사 중단, 불량 배출로 전환')
                    self._barm_simple('ABORT', return_home=True)
                    steps, e = self.build_reject(self.q_cmd,
                                                 lead_wait=self.barm_abort_wait)
                    if steps is None:
                        self._abort(e); return
                    self.queue.clear()
                    self.queue.extend(steps)
                    self._set_phase('REJECT')
                    st['_done'] = True
                    return

        # 2) B_arm 상태 대기
        while self.barm_status:
            d = self.barm_status.popleft()
            state = str(d.get('state', '')).upper()
            if state == 'COMMAND_REJECTED':
                self.get_logger().error('     B_arm 명령 거부: %s' % d.get('reason'))
                self._abort('B_arm 명령 거부 (%s)' % d.get('reason'))
                return
            if state in ('MOTION_ERROR', 'INSPECTION_ERROR'):
                self.get_logger().error('     B_arm 오류: %s / %s'
                                        % (state, d.get('reason')))
                self._abort('B_arm 오류 (%s)' % state)
                return
            if state == 'INSPECTION_TIMEOUT':
                self.get_logger().error('     B_arm 검사 타임아웃')
                self._abort('B_arm 검사 타임아웃')
                return
            if state in st['wait']:
                self.get_logger().info('     B_arm %s' % state)
                st['_done'] = True
                return
            self.get_logger().info('     B_arm %s (대기중)' % state)

        if (now - (st['_t0'] or now)) > self.barm_timeout:
            if self.require_barm:
                self._abort('B_arm 응답 시간초과 (%s, %.0fs)' % (st['cmd'], self.barm_timeout))
            else:
                self.get_logger().warn('     B_arm 응답 없음 -> 무시하고 진행')
                st['_done'] = True

    def _barm_simple(self, cmd, **extra):
        pay = {
            'command': cmd,
            'cycle_id': self.cycle_id,
            'request_id': uuid.uuid4().hex[:12],
        }
        pay.update(extra)
        m = String()
        m.data = json.dumps(pay, ensure_ascii=False)
        self.barm_pub.publish(m)
        self.get_logger().info('  -> B_arm %s %s' % (cmd, extra if extra else ''))

    # ================================================== 메인 루프
    def tick(self):
        now = time.time()

        if self.rail is not None:
            self.rail_m = float(np.clip(self.rail.position_m, RAIL_MIN_M, RAIL_MAX_M))
            m = Float64(); m.data = self.rail_m; self.rail_pub.publish(m)

        # ---- 버튼 ----
        if self._click(self.b_stop, 'stop'):
            if self.rail:
                self.rail.suction(False)
            self._holding = False
            self._abort('EMERGENCY STOP')
        if self._click(self.b_auto, 'auto'):
            self.auto_run = not self.auto_run
            self.get_logger().info('AUTO = %s' % self.auto_run)
        if self._click(self.b_von, 'von'):
            if self.rail:
                self.rail.suction(True)
            self._holding = True
            self.get_logger().info('수동 흡착 ON')
        if self._click(self.b_voff, 'voff'):
            if self.rail:
                self.rail.suction(False)
            self._holding = False
            self.get_logger().info('수동 흡착 OFF')
        if self._click(self.b_home, 'home') and not self.busy:
            self._start([self._S('go_home', name='초기위치 복귀')], 'HOME')
        if self._click(self.b_appr, 'appr'):
            if self._await_approve:
                self._await_approve = False
            else:
                self.get_logger().info('APPROVE: 승인 대기 상태가 아니라 무시합니다.')

        if self.busy and self._await_approve:
            self._render(); return

        if self.busy and self.step is None:
            self._next_step()

        st = self.step
        if st is not None:
            k = st['kind']
            if k == 'traj':
                self._play(now)
            elif k == 'wait':
                if st['_t0'] is None:
                    st['_t0'] = now
                    st['_ok'] = not st.get('settle', False) or self.bridge is None
                if not st.get('_ok', True):
                    if self._check_settled(now, st):
                        st['_t0'] = now; st['_ok'] = True
                elif (now - st['_t0']) >= float(st['sec']):
                    st['_done'] = True
            elif k == 'settle':
                if self._check_settled(now, st):
                    st['_done'] = True
            elif k == 'rail_wait':
                self._poll_rail_wait(now, st)
            elif k == 'barm':
                self._poll_barm(now, st)
            elif k in ('branch', 'branch2'):
                if k == 'branch2':
                    self._do_branch2()
                st['_done'] = True

            if st.get('_done'):
                self.step = None
                if not self.auto_run and self.queue:
                    self._await_approve = True

        self._render()

    def _poll_rail_wait(self, now, st):
        if self.rail is None:
            st['_done'] = True
            return

        pos = self.rail.position_mm
        if st['_t0'] is None:
            st['_t0'] = now
            st['_last_pos'] = pos
            st['_still_t'] = now

        if abs(pos - st['_last_pos']) > 0.05:
            st['_last_pos'] = pos
            st['_still_t'] = now

        ev = self.rail.last_event
        if ev == 'L':
            self._abort('레일 이동 거부 (@L). 현재 %.1fmm' % pos)
            return

        target = self._rail_target_mm
        err = abs(pos - target) if target is not None else 0.0
        settled = (not self._rail_moving()) and ev != 'P'

        if settled and (now - st['_t0']) > 0.3 and (target is None or err <= self.rail_tol_mm):
            st['_done'] = True
            self.get_logger().info('  레일 도착 %.1fmm (목표 %.1fmm, 오차 %.2fmm)'
                                   % (pos, target if target is not None else pos, err))
            return

        if settled and (now - st['_still_t']) > 2.0 and err > self.rail_tol_mm:
            self._abort('레일이 움직이지 않습니다 (목표 %.1fmm, 현재 %.1fmm)'
                        % (target if target is not None else -1.0, pos))
            return

        if (now - st['_t0']) > self.rail_timeout:
            self._abort('레일 이동 시간초과 %.0fs' % self.rail_timeout)

    def _do_branch2(self):
        if self.defect_seen:
            self.get_logger().warn('다면 검사 NG -> 불량 배출')
            steps, e = self.build_reject(self.q_cmd)
            if steps is None:
                self._abort(e); return
            self._set_phase('REJECT')
        else:
            self.get_logger().info('전 공정 OK -> 정상 배출')
            steps, e = self.build_good_drop(self.q_cmd)
            if steps is None:
                self._abort(e); return
            self._set_phase('GOOD')
        self.queue.extendleft(reversed(steps))

    def _check_settled(self, now, st):
        if self.bridge is None:
            return True
        # PyBullet/llvmpipe의 실제 callback 속도와 무관하게 벽시계 기준으로
        # 관절을 확인한다. 기존 sim_hz 기반 tick 카운트는 GUI가 느릴 때
        # 0.1초가 수 초로 늘어나는 문제가 있었다.
        mono = time.monotonic()
        if st.get('_settle_started_mono') is None:
            st['_settle_started_mono'] = mono
            st['_settle_last_read_mono'] = 0.0
            st['_settle_ok_count'] = 0
        if mono - st.get('_settle_last_read_mono', 0.0) < self.settle_poll:
            return False
        st['_settle_last_read_mono'] = mono

        want_q5 = bool(st.get('check_q5', False))
        try:
            qr_all = np.array(self.bridge.read_joints(), float).ravel()
            e_arm = float(np.max(np.abs(wrap_to_pi(qr_all[:N_JOINTS] - self.q_cmd))))
            e_q5 = 0.0
            if want_q5 and qr_all.size > N_JOINTS:
                e_q5 = float(abs(wrap_to_pi(qr_all[N_JOINTS] - self.q5)))
            ok_arm = e_arm <= self.settle_tol
            ok_q5 = (not want_q5) or (e_q5 <= self.q5_settle_tol)
            if ok_arm and ok_q5:
                st['_settle_ok_count'] = st.get('_settle_ok_count', 0) + 1
            else:
                st['_settle_ok_count'] = 0
            if st['_settle_ok_count'] >= self.settle_confirm_samples:
                self.get_logger().info('     도달 확인 (팔 %.2f도%s, %d회 연속)'
                                       % (np.degrees(e_arm),
                                          ', 그리퍼 %.2f도' % np.degrees(e_q5)
                                          if want_q5 else '',
                                          self.settle_confirm_samples))
                return True
            if mono - st['_settle_started_mono'] > self.settle_timeout:
                self.get_logger().warn('     도달 대기 시간초과 (팔 %.2f도%s)'
                                       % (np.degrees(e_arm),
                                          ', 그리퍼 %.2f도' % np.degrees(e_q5) if want_q5 else ''))
                return True
        except Exception as ex:
            self.get_logger().warn('     관절 읽기 실패: %s' % ex)
            if mono - st['_settle_started_mono'] > self.settle_timeout:
                self.get_logger().warn('     관절 읽기 시간초과 -> 다음 단계로 진행')
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
            ('rail', 'RAIL: %.3f m   SUCTION: %s' %
             (self.rail_m, ('ON' if (self.rail and self.rail.suction_on) else 'OFF')),
             [0.3, 1, 0.4], 1.20),
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
    node = ProcessNode()
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
