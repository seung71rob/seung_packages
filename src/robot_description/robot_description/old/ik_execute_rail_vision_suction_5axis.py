"""
robot_description / ik_execute_rail_vision_suction_5axis.py
--------------------------------------------------
suction 그리퍼 + 리니어 레일 + 탑뷰 비전 입력.

기존 rail 노드에 /detected_object (탑뷰 YOLO 결과) 구독을 추가한 버전.

좌표 흐름
  A3 mm (비전)
    -> 로봇 사용자 mm      X = A3_LT_X - Y_A3,  Y = ±(X_A3 - A3_LT_Y)
    -> 팔(link1) 기준 m    (USER_*_SIGN 적용, mm -> m)
    -> 월드(rail) 기준 m   + rail_origin + [0, rail_m, 0]        ★레일 오프셋
    -> 노드 내부에서 다시 world_to_arm() 으로 팔 기준 복귀

  중간에 월드로 올렸다가 다시 내리는 이유:
  A3 를 잰 시점의 레일 위치(vision_rail_ref_m)와 지금 레일 위치가
  다를 수 있기 때문이다. 월드에 고정된 물체 좌표를 유지하면서
  레일이 움직인 만큼 팔 목표가 자동으로 보정된다.

동작
  /detected_object 수신 -> 접근점(z + approach) + 파지점 두 점 생성
  -> 기존 두 점 수직경로 로직 그대로 (접근 -> 정지 -> 수직 하강)
  -> 미리보기 후 APPROVE 로 실행

  vision_auto:=false 이면 미리보기만 만들고 자동 실행하지 않는다(기본).

기존 ik_execute_suction_5axis.py 와 동작이 동일하고, 여기에 레일이 추가된다.
 - 아두이노(rail_stepmotor.ino)가 스텝모터를 돌리고 절대 스텝수를 시리얼로 보고
 - 이 노드가 그 값을 읽어 PyBullet 의 prismatic 관절(joint_rail)에 그대로 반영
 - PyBullet 버튼으로도 레일 CW / CCW / STOP 을 보낼 수 있다

레일 제원: 1600 PPR, 리드 10 mm -> 160 pulse/mm.
           스트로크 0 ~ 900 mm (= 0 ~ 144,000 pulse), 500 RPM 에서 약 10.8 s

좌표계
  /target_point 는 '월드' 좌표(rail 링크 원점 기준)로 받는다.
  팔의 IK 는 link1 프레임에서 풀리므로, 레일 위치만큼 빼서 넘긴다.
      target_arm = target_world - (rail_origin + [0, rail_m, 0])
  레일을 안 쓰면(use_rail=false, rail_m=0) 기존 노드와 좌표계가 동일하다.

충돌검사
  팔 전용 URDF(robot_arm_5axis_suction.urdf)를 그대로 사용한다.
  레일 URDF 는 prismatic 관절이 앞에 붙어 관절 인덱스가 밀리므로
  CollisionChecker 에는 넘기지 않는다. 레일은 팔 아래에 있어 자기충돌과 무관.

파라미터: use_robot, use_rail, rail_port, rail_baud, fix_mode(초기값),
          check_collision, gripper_deg, sim_hz, robot_cmd_hz,
          speed_deg_per_s, dxl_port, mesh_dir, pkg_dir

실행:
  ros2 run robot_description ik_execute_rail_suction_5axis --ros-args \
      -p use_robot:=false -p use_rail:=true -p rail_port:=/dev/ttyACM0
좌표:
  ros2 topic pub --once /target_point geometry_msgs/msg/Point "{x: -0.25, y: 0.05, z: 0.35}"
레일 토픽(선택):
  ros2 topic pub --once /rail_cmd std_msgs/msg/String "{data: 'c'}"
"""
import os
import time
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from std_msgs.msg import Float64, Float64MultiArray, String
import pybullet as p
import pybullet_data

from .kinematics_ee_5axis import (fk, fk_pos, ik, ik_fix, ik_fix_candidates,
                                  N_JOINTS, wrap_to_pi)
from .trajectory_5axis import plan_joint_path
from .path_collision_guard_ee_5axis import CollisionChecker
from .rail_serial_5axis import RailSerial, RAIL_MIN_M, RAIL_MAX_M, M_PER_STEP
from .vertical_path_5axis import (sort_by_height, plan_level_descent,
                                  plan_descent_ends_retimed, plan_joint_descent_retimed,
                                  retime_path, plan_sync_joint_move,
                                  joint_speed_limits,
                                  path_joint_speed, ik_level_soft,
                                  fix_reach_report, level_error)

URDF_NAME = 'robot_arm_5axis_rail_suction.urdf'      # PyBullet 표시용 (레일 포함)
ARM_URDF_NAME = 'robot_arm_5axis_suction.urdf'       # 충돌검사용 (팔만)
RAIL_JOINT_NAME = 'joint_rail'

# ============================================================
# 탑뷰 비전 (A3) -> 로봇 사용자 좌표
#
# A3 좌표계 : 좌상단 (0,0), 우상단 (420,0), 좌하단 (0,297)
# 실측한 A3 모서리의 로봇 사용자 좌표:
#   좌상단 -> (X=482.0,  Y=214.55)
#   우상단 -> (X=482.0,  Y=-205.45)
#   좌하단 -> (X=185.0,  Y=214.55)
# 따라서
#   X_robot = A3_LT_ROBOT_X - Y_A3
#   Y_robot = X_A3 - A3_LT_ROBOT_Y          (vision_y_sign = +1, 원본 코드와 동일)
#
# 원본 execute_centric_vision_preview 에서 실기 검증된 식을 그대로 쓴다.
# (그 파일의 주석에는 214.55 - X_A3 로 적혀 있으나 실제 구현은 위 식이며,
#  실기에서 맞는 것으로 확인됨. 혹시 방향이 반대로 나오면
#  -p vision_y_sign:=-1.0 으로 뒤집을 수 있다.)
# ============================================================
A3_W_MM = 420.0
A3_H_MM = 297.0
A3_LT_ROBOT_X_MM = 482.0
A3_LT_ROBOT_Y_MM = 214.55

# 로봇 사용자 mm -> 팔(link1) 기준 m 부호
USER_X_SIGN = -1.0
USER_Y_SIGN = +1.0
USER_Z_SIGN = +1.0

VISION_TOPIC = '/detected_object'
BUSY_FLAG_PATH = '/tmp/robot_auto_busy.flag'


def user_mm_to_arm_m(x_mm, y_mm, z_mm):
    """로봇 사용자 좌표[mm] -> 팔(link1) 기준 [m]"""
    return np.array([USER_X_SIGN * float(x_mm) / 1000.0,
                     USER_Y_SIGN * float(y_mm) / 1000.0,
                     USER_Z_SIGN * float(z_mm) / 1000.0], float)


class ExecNode(Node):
    def __init__(self):
        super().__init__('ik_execute_rail_vision_suction_5axis')
        self.declare_parameter('pkg_dir', os.path.expanduser('~/robot_sim/src/robot_description'))
        self.declare_parameter('mesh_dir', os.path.expanduser('~/robot_sim/src/robot_description/meshes_robot_5axis'))
        self.declare_parameter('use_robot', False)
        self.declare_parameter('use_rail', False)
        self.declare_parameter('rail_port', '/dev/ttyACM0')
        self.declare_parameter('rail_baud', 115200)
        self.declare_parameter('fix_mode', False)
        self.declare_parameter('check_collision', True)
        self.declare_parameter('gripper_deg', 0.0)
        self.declare_parameter('sim_hz', 240.0)
        self.declare_parameter('robot_cmd_hz', 50.0)
        self.declare_parameter('speed_deg_per_s', 60.0)
        self.declare_parameter('dxl_port', '/dev/ttyUSB0')
        self.declare_parameter('dwell_sec', 0.5)          # 1차점 정지 시간
        self.declare_parameter('descend_speed', 0.05)     # 하강 직선속도 m/s (상한)
        self.declare_parameter('limit_joint_speed', True) # 모터 속도한계로 자동 감속
        self.declare_parameter('speed_margin', 0.9)       # 한계 대비 여유 (0.9 = 90%)
        # 수직조건 엄격도
        #   'both' : 1차점/2차점 모두 정확히 수직 (중간만 완화)
        #   'end'  : 2차점(흡착 지점)만 수직. 접근 자세는 기울여도 됨
        #   'none' : 제한 없음
        self.declare_parameter('strict_ends', 'both')
        self.declare_parameter('level_tol_deg', 25.0)     # 완화 구간 허용 기울기(도)
        self.declare_parameter('descent_fallback_joint', True)  # 직선 하강 실패 시 관절공간 하강
        # ---- 비전 ----
        self.declare_parameter('use_vision', True)        # /detected_object 구독
        self.declare_parameter('vision_topic', VISION_TOPIC)
        self.declare_parameter('vision_z_mm', 15.0)       # 파지 높이(로봇 사용자 Z)
        self.declare_parameter('vision_approach_mm', 80.0)  # 파지점 위 접근 높이
        self.declare_parameter('vision_rail_ref_m', 0.0)  # A3 를 잰 시점의 레일 위치
        self.declare_parameter('vision_y_sign', +1.0)     # Y_robot 부호 (원본 코드와 동일)
        self.declare_parameter('a3_lt_x_mm', A3_LT_ROBOT_X_MM)
        self.declare_parameter('a3_lt_y_mm', A3_LT_ROBOT_Y_MM)
        self.declare_parameter('vision_only_ok', True)    # NG 판정이면 무시
        self.declare_parameter('vision_fallback_single', True)  # 두 점 실패 시 단일점
        self.declare_parameter('vision_min_gap_mm', 5.0)  # 접근 높이가 이보다 작으면 처음부터 단일점
        # ---- 픽 시퀀스 ----
        self.declare_parameter('pick_sequence', True)     # 하강-압착-흡착-상승 자동 시퀀스
        self.declare_parameter('press_mm', 5.0)           # 2차점 아래로 더 내려갈 거리
        self.declare_parameter('suction_dwell', 0.5)      # 흡착 ON 후 유지 시간
        # 폴백(두 점 / 단일점) 경로의 흡착 후 복귀점. ★팔(link1) 기준 좌표, m
        #   레일 오프셋을 적용하지 않는다. 레일이 어디 있든 팔에 대해 같은 자세.
        #   (-0.35, 0, 0.15) 는 어깨에서 0.259m 로 적당히 떨어져 있어
        #   복귀에 약 1.6초. 원점 근처(0,0,0.15)로 잡으면 팔이 크게 접혀
        #   q2(감속비 15:1)가 120도나 돌아 12초 넘게 걸린다.
        self.declare_parameter('retreat_arm_xyz', [-0.35, 0.0, 0.15])
        self.declare_parameter('settle_tol_deg', 1.0)     # 1차점 도달 판정 허용오차
        self.declare_parameter('settle_timeout', 5.0)     # 도달 대기 최대 시간(초)
        self.declare_parameter('render_hz', 60.0)         # PyBullet 화면 갱신 주기
        # ---- 티칭(조그) / 자세 저장 ----
        self.declare_parameter('allow_jog', True)         # 조그 명령 허용
        self.declare_parameter('jog_max_deg', 20.0)       # 1회 조그 최대 각도
        self.declare_parameter('pose_file',
                               os.path.expanduser('~/robot_sim/saved_poses.yaml'))

        self.pkg_dir = self.get_parameter('pkg_dir').value
        self.mesh_dir = self.get_parameter('mesh_dir').value
        self.use_robot = bool(self.get_parameter('use_robot').value)
        self.use_rail = bool(self.get_parameter('use_rail').value)
        self.fix_mode = bool(self.get_parameter('fix_mode').value)
        self.sim_hz = float(self.get_parameter('sim_hz').value)
        self.robot_cmd_hz = float(self.get_parameter('robot_cmd_hz').value)
        self.speed = float(self.get_parameter('speed_deg_per_s').value)
        self.q5 = np.radians(float(self.get_parameter('gripper_deg').value))
        self.dwell_sec = float(self.get_parameter('dwell_sec').value)
        self.descend_speed = float(self.get_parameter('descend_speed').value)
        self.limit_joint_speed = bool(self.get_parameter('limit_joint_speed').value)
        self.speed_margin = float(self.get_parameter('speed_margin').value)
        self.strict_ends = str(self.get_parameter('strict_ends').value).lower()
        if self.strict_ends not in ('both', 'end', 'none'):
            self.get_logger().warn("strict_ends 는 both/end/none 중 하나. 'both' 로 진행.")
            self.strict_ends = 'both'
        self.level_tol = np.radians(float(self.get_parameter('level_tol_deg').value))
        self.descent_fallback_joint = bool(self.get_parameter('descent_fallback_joint').value)
        self.use_vision = bool(self.get_parameter('use_vision').value)
        self.vision_z_mm = float(self.get_parameter('vision_z_mm').value)
        self.vision_approach_mm = float(self.get_parameter('vision_approach_mm').value)
        self.vision_rail_ref = float(self.get_parameter('vision_rail_ref_m').value)
        self.vision_y_sign = float(self.get_parameter('vision_y_sign').value)
        self.a3_lt_x = float(self.get_parameter('a3_lt_x_mm').value)
        self.a3_lt_y = float(self.get_parameter('a3_lt_y_mm').value)
        self.vision_only_ok = bool(self.get_parameter('vision_only_ok').value)
        self.vision_fallback_single = bool(self.get_parameter('vision_fallback_single').value)
        self.vision_min_gap_mm = float(self.get_parameter('vision_min_gap_mm').value)
        self.pick_sequence = bool(self.get_parameter('pick_sequence').value)
        self.press_mm = float(self.get_parameter('press_mm').value)
        self.suction_dwell = float(self.get_parameter('suction_dwell').value)
        self.retreat_arm = np.array(
            [float(v) for v in self.get_parameter('retreat_arm_xyz').value], float)
        self.vmax = None                                  # connect 후 브리지에서 계산
        self.settle_tol = np.radians(float(self.get_parameter('settle_tol_deg').value))
        self.settle_timeout = float(self.get_parameter('settle_timeout').value)
        self.render_hz = max(1.0, float(self.get_parameter('render_hz').value))
        self.allow_jog = bool(self.get_parameter('allow_jog').value)
        self.jog_max = np.radians(float(self.get_parameter('jog_max_deg').value))
        self.pose_file = os.path.expanduser(str(self.get_parameter('pose_file').value))

        # 레일 원점 오프셋을 URDF 에서 직접 읽는다 (하드코딩 방지)
        self.rail_origin = self._read_rail_origin()

        urdf_abs = self._prep(URDF_NAME)
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)
        p.loadURDF('plane.urdf')
        p.resetDebugVisualizerCamera(1.6, 50, -25, [0.0, 0.4, 0.4])
        self.robot = p.loadURDF(urdf_abs, [0, 0, 0], useFixedBase=True)

        self.rev = []
        self.rail_jid = None
        for j in range(p.getNumJoints(self.robot)):
            info = p.getJointInfo(self.robot, j)
            name = info[1].decode() if isinstance(info[1], bytes) else str(info[1])
            if info[2] == p.JOINT_REVOLUTE:
                self.rev.append(j)
            elif info[2] == p.JOINT_PRISMATIC and name == RAIL_JOINT_NAME:
                self.rail_jid = j
        if self.rail_jid is None:
            self.get_logger().error('URDF 에서 %s (prismatic) 를 못 찾음.' % RAIL_JOINT_NAME)

        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.012, rgbaColor=[1, 0, 0, 0.9])
        self.marker = p.createMultiBody(0, baseVisualShapeIndex=vis, basePosition=[10, 10, 10])
        vis2 = p.createVisualShape(p.GEOM_SPHERE, radius=0.012, rgbaColor=[0, 0.5, 1, 0.9])
        self.marker2 = p.createMultiBody(0, baseVisualShapeIndex=vis2, basePosition=[10, 10, 10])
        self._pair_line = None

        self.b_appr = p.addUserDebugParameter('APPROVE -> run robot', 1, 0, 0)
        self.b_stop = p.addUserDebugParameter('EMERGENCY STOP', 1, 0, 0)
        self.b_home = p.addUserDebugParameter('GO TO HOME', 1, 0, 0)
        self.b_fix = p.addUserDebugParameter('FIX MODE on/off (click)', 1, 0, 0)
        self.b_rcw = p.addUserDebugParameter('RAIL  -> end (0.9m)', 1, 0, 0)
        self.b_rccw = p.addUserDebugParameter('RAIL  -> origin (0.0m)', 1, 0, 0)
        self.b_rstop = p.addUserDebugParameter('RAIL  smooth STOP', 1, 0, 0)
        self.b_restop = p.addUserDebugParameter('RAIL  IMMEDIATE STOP', 1, 0, 0)
        self.b_rzero = p.addUserDebugParameter('RAIL  set zero here', 1, 0, 0)
        self.b_von = p.addUserDebugParameter('SUCTION  ON', 1, 0, 0)
        self.b_voff = p.addUserDebugParameter('SUCTION  OFF', 1, 0, 0)
        self.s_rgoto = p.addUserDebugParameter('RAIL  goto (m)', 0.0, 0.9, 0.0)
        self.b_rgoto = p.addUserDebugParameter('RAIL  GO to slider', 1, 0, 0)
        self._al = 0; self._sl = 0; self._hl = 0; self._fl = 0
        self._rcw = 0; self._rccw = 0; self._rstop = 0; self._rzero = 0
        self._restop = 0; self._rgoto = 0
        self._von = 0; self._voff = 0
        self._suc_txt = None
        self._fix_txt = None
        self._load_txt = None
        self._rail_txt = None
        self._load_tick = 0
        self._rail_tick = 0

        # 충돌검사는 팔 전용 URDF 사용 (레일 관절이 인덱스를 밀지 않도록)
        self.checker = None
        if bool(self.get_parameter('check_collision').value):
            self.checker = CollisionChecker(os.path.join(self.pkg_dir, 'urdf', ARM_URDF_NAME),
                                            self.pkg_dir, N_JOINTS, self.mesh_dir)

        # 레일 시리얼
        self.rail = None
        self.rail_m = 0.0
        if self.use_rail:
            try:
                self.rail = RailSerial(port=self.get_parameter('rail_port').value,
                                       baud=int(self.get_parameter('rail_baud').value),
                                       logger=self.get_logger())
                self.rail.connect()
            except Exception as e:
                self.rail = None
                self.get_logger().error('레일 시리얼 연결 실패: %s (레일 없이 계속)' % e)

        self.bridge = None
        if self.use_robot:
            from .dxl_bridge_suction_5axis import DxlBridge
            self.bridge = DxlBridge(port=self.get_parameter('dxl_port').value)
            self.bridge.connect(); self.bridge.capture_home()
            self.get_logger().warn('실로봇: 로봇을 home(관절 0)에 두고 시작하세요.')
            self._calc_speed_limits()

        self.q_cmd = np.zeros(N_JOINTS)
        self.traj = None; self.idx = 0; self.mode = 'idle'; self.path_safe = False
        self.pending = np.zeros(N_JOINTS)
        self._play_t0 = None        # 재생 시작 시각 (벽시계 기준 재생)
        self._last_cmd_t = 0.0      # 마지막 모터 명령 시각
        self._render_tick = 0
        self._plan_traj = None      # 계획한 경로 원본 (미리보기가 끝나도 유지)
        self._plan_pause_at = -1    # 그 경로의 정지 인덱스
        self.pause_at = -1          # 이 인덱스에서 멈춘다 (-1 이면 정지 없음)
        # 픽 시퀀스용 이벤트: [{'idx':int,'kind':str,'dwell':float,'done':bool,'t0':float}]
        #   kind : 'pause'(그냥 대기) / 'suction_on' / 'suction_off'
        self.events = []
        self._plan_events = []
        self._pause_t0 = None       # 정지 시작 시각
        self._pause_done = False
        self._settled = False       # 실로봇이 1차점에 실제로 도달했는가
        self._settle_tick = 0
        self.send_every = max(int(round(self.sim_hz / self.robot_cmd_hz)), 1)
        self._apply_sim(self.q_cmd)
        self._show_fix_state()

        self.create_subscription(Point, 'target_point', self.on_target, 10)
        self.create_subscription(Float64MultiArray, 'target_pair', self.on_target_pair, 10)
        self.create_subscription(Float64MultiArray, 'jog_joint', self.on_jog_joint, 10)
        self.create_subscription(Float64MultiArray, 'goto_joint', self.on_goto_joint, 10)
        self.create_subscription(String, 'pose_save', self.on_pose_save, 10)
        self.create_subscription(String, 'pose_goto', self.on_pose_goto, 10)
        self.create_subscription(String, 'pose_list', self.on_pose_list, 10)
        self.create_subscription(String, 'pose_undo', self.on_pose_undo, 10)
        self.create_subscription(String, 'pose_clear', self.on_pose_clear, 10)
        self.create_subscription(String, 'home_joint', self.on_home_joint, 10)
        if self.use_vision:
            topic = str(self.get_parameter('vision_topic').value)
            self.create_subscription(String, topic, self.on_vision, 10)
            self.get_logger().info('비전 구독: %s  (파지 z=%.0fmm, 접근 +%.0fmm, '
                                   'A3 기준 레일 %.3fm)'
                                   % (topic, self.vision_z_mm,
                                      self.vision_approach_mm, self.vision_rail_ref))
        self.create_subscription(String, 'rail_cmd', self.on_rail_cmd, 10)
        self.create_subscription(Float64, 'rail_target_m', self.on_rail_target, 10)
        self.rail_pub = self.create_publisher(Float64, 'rail_position', 10)
        self.create_timer(1.0 / self.sim_hz, self.step)
        self.get_logger().info('Ready (suction+rail, fix_mode=%s, use_robot=%s, use_rail=%s).'
                               % (self.fix_mode, self.use_robot, self.rail is not None))

    # ---------- URDF ----------
    def _read_rail_origin(self):
        """joint_rail 의 origin xyz 를 URDF 에서 읽어 (3,) 반환."""
        path = os.path.join(self.pkg_dir, 'urdf', URDF_NAME)
        try:
            root = ET.parse(path).getroot()
            for j in root.findall('joint'):
                if j.get('name') == RAIL_JOINT_NAME:
                    o = j.find('origin')
                    return np.array([float(v) for v in o.get('xyz').split()], float)
        except Exception as e:
            self.get_logger().warn('rail origin 읽기 실패(0으로 가정): %s' % e)
        return np.zeros(3)

    def _prep(self, urdf_name):
        with open(os.path.join(self.pkg_dir, 'urdf', urdf_name)) as f:
            txt = f.read()
        txt = txt.replace('robot_meshes/', self.mesh_dir.rstrip('/') + '/')
        txt = txt.replace('package://robot_description/', self.pkg_dir + '/')
        out = '/tmp/' + urdf_name
        with open(out, 'w') as f:
            f.write(txt)
        return out

    # ---------- 좌표계 변환 ----------
    def world_to_arm(self, target_world):
        """월드 좌표 -> link1(팔 베이스) 기준 좌표. 레일 위치를 뺀다."""
        off = self.rail_origin + np.array([0.0, self.rail_m, 0.0])
        return np.asarray(target_world, float) - off

    def arm_to_world(self, p_arm):
        off = self.rail_origin + np.array([0.0, self.rail_m, 0.0])
        return np.asarray(p_arm, float) + off

    # ---------- 레일 명령 ----------
    def on_rail_cmd(self, msg):
        cmd = str(msg.data).strip().lower()
        if self.rail is None:
            self.get_logger().warn('레일이 연결되지 않음 (use_rail:=true 로 실행).')
            return
        ok = (cmd in ('c', 'cw', 'ccw', 'cc', 's', '0', 'z', '?',
                      'von', 'v1', 'voff', 'v0', 'v?')
              or cmd.startswith('p ') or cmd.startswith('m '))
        if ok:
            self.rail.send(cmd)
            self.get_logger().info('레일 명령 전송: %s' % cmd)
        else:
            self.get_logger().warn(
                "알 수 없는 레일 명령: '%s'  "
                "(z | p <mm> | m <mm> | c | ccw | s | 0 | ? | von | voff)" % cmd)

    def on_rail_target(self, msg):
        """레일을 절대 위치(m)로 이동. 예: ros2 topic pub /rail_target_m ... 0.5"""
        if self.rail is None:
            self.get_logger().warn('레일 미연결 (use_rail:=true 로 실행).')
            return
        m = float(msg.data)
        if not (RAIL_MIN_M - 1e-9) <= m <= (RAIL_MAX_M + 1e-9):
            self.get_logger().error('레일 목표가 범위를 벗어남: %.3f m (허용 %.1f~%.1f)'
                                    % (m, RAIL_MIN_M, RAIL_MAX_M))
            return
        self.rail.move_abs_m(m)
        self.get_logger().info('레일 -> %.3f m' % m)

    # ---------- 비전 입력 ----------
    def a3_to_world(self, a3_x_mm, a3_y_mm, z_mm):
        """
        A3 mm -> 월드(rail 원점 기준) m.

        1) A3 -> 로봇 사용자 mm
        2) 사용자 mm -> 팔(link1) 기준 m
        3) 팔 기준 -> 월드 : + rail_origin + [0, vision_rail_ref, 0]
           (A3 를 잰 시점의 레일 위치를 기준으로 월드에 고정시킨다)
        """
        rx_mm = self.a3_lt_x - float(a3_y_mm)
        ry_mm = self.vision_y_sign * (float(a3_x_mm) - self.a3_lt_y)
        p_arm = user_mm_to_arm_m(rx_mm, ry_mm, z_mm)
        off = self.rail_origin + np.array([0.0, self.vision_rail_ref, 0.0])
        return p_arm + off, np.array([rx_mm, ry_mm, float(z_mm)])

    def on_vision(self, msg):
        """
        /detected_object : "baseplate,X_A3_mm,Y_A3_mm,0.0,0,OK|NG"
        파지점과 그 위 접근점을 만들어 두 점 수직경로로 넘긴다.
        """
        if os.path.exists(BUSY_FLAG_PATH):
            self.get_logger().warn('[VISION] 로봇 busy 플래그가 있어 무시합니다.')
            return

        parts = [s.strip() for s in str(msg.data).split(',')]
        if len(parts) < 3:
            self.get_logger().error('[VISION] 형식 오류: %s' % msg.data)
            return
        try:
            a3_x = float(parts[1]); a3_y = float(parts[2])
        except ValueError:
            self.get_logger().error('[VISION] 숫자 변환 실패: %s' % msg.data)
            return

        judge = parts[5].upper() if len(parts) > 5 else 'OK'
        if self.vision_only_ok and judge not in ('OK',):
            self.get_logger().warn('[VISION] 판정 %s -> 무시 (vision_only_ok)' % judge)
            return

        if not (0.0 <= a3_x <= A3_W_MM and 0.0 <= a3_y <= A3_H_MM):
            self.get_logger().error('[VISION] A3 영역 밖: X=%.1f Y=%.1f' % (a3_x, a3_y))
            return

        w_pick, user_mm = self.a3_to_world(a3_x, a3_y, self.vision_z_mm)
        w_appr, _ = self.a3_to_world(a3_x, a3_y,
                                     self.vision_z_mm + self.vision_approach_mm)

        arm_pick = self.world_to_arm(w_pick)
        self.get_logger().info(
            '[VISION] A3(%.1f, %.1f) -> 사용자(%.1f, %.1f, %.1f)mm '
            '-> 월드 %s -> 현재 레일 %.3fm 기준 팔 %s'
            % (a3_x, a3_y, user_mm[0], user_mm[1], user_mm[2],
               w_pick.round(4), self.rail_m, arm_pick.round(4)))

        # 접근 높이가 거의 0 이면 두 점이 겹치므로 처음부터 단일점
        if abs(self.vision_approach_mm) < self.vision_min_gap_mm:
            self.get_logger().info('[VISION] 접근 높이 %.1fmm -> 두 점이 겹침, 단일점 경로'
                                   % self.vision_approach_mm)
            self._goto_single(w_pick)
            return

        if self.pick_sequence:
            if self._build_pick_sequence(w_appr, w_pick):
                return
            self.get_logger().warn('[VISION] 픽 시퀀스 실패 -> 두 점 경로로 재시도')

        ok, why = self._build_pair_retreat(w_appr, w_pick)
        if ok:
            return
        self.get_logger().warn('[VISION] 두 점 경로 실패(%s)' % why)

        if not self.vision_fallback_single:
            self.get_logger().error('  단일점 폴백을 쓰려면 -p vision_fallback_single:=true')
            return

        self.get_logger().warn('  -> 파지점 단일점으로 재시도')
        self._goto_single(w_pick)

    def _descend(self, q_from, p_from, p_to):
        """
        수직 유지 하강 한 구간. 직선 -> 실패 시 관절공간 폴백.
        반환 (traj, mode, msg).  실패 시 (None, '', 사유)
        """
        seg, T, mx, msg = plan_descent_ends_retimed(
            q_from, p_from, p_to, hz=self.sim_hz,
            lin_speed=self.descend_speed,
            mode=self.strict_ends, level_tol=self.level_tol,
            vmax=self.vmax if self.limit_joint_speed else None,
            margin=self.speed_margin)
        if seg is not None:
            return seg, '직선', msg
        if not self.descent_fallback_joint:
            return None, '', msg
        seg, T, mx, msg2 = plan_joint_descent_retimed(
            q_from, p_from, p_to, hz=self.sim_hz,
            lin_speed=self.descend_speed,
            vmax=self.vmax if self.limit_joint_speed else None,
            margin=self.speed_margin)
        if seg is None:
            return None, '', msg2
        return seg, '관절공간', msg2

    def _retreat_seg(self, q_from):
        """
        흡착 후 복귀 구간. 목표는 팔(link1) 기준 고정 좌표라 레일 오프셋을
        적용하지 않는다. 수직해를 우선 쓰고, 없으면 위치 IK 로 간다.
        반환 (seg, msg). 실패 시 (None, 사유)
        """
        p_ret = self.retreat_arm
        cands = ik_fix_candidates(p_ret, q0=q_from)
        if cands:
            cands.sort(key=lambda c: np.linalg.norm(wrap_to_pi(c[0] - q_from)))
            q_ret = cands[0][0]
        else:
            q_ret, e = ik(p_ret, q0=q_from)
            if e > 1e-3:
                return None, '복귀점 %s 도달 불가 (잔차 %.4f m)' % (p_ret.round(3), e)
            self.get_logger().warn('복귀점 수직해 없음 -> 위치 IK 사용')

        if self.checker is not None and not self.checker.check_config(q_ret)[0]:
            return None, '복귀 자세가 충돌'

        # 복귀도 수직 유지. 공통 시간 프로파일 보간이라 양 끝이 수직이면
        # 중간도 수직이다. plan_joint_path 는 중간점을 넣어서 이게 깨진다.
        seg, T, mx, info = plan_sync_joint_move(
            q_from, q_ret, hz=self.sim_hz,
            vmax=self.vmax if self.limit_joint_speed else None,
            margin=self.speed_margin)
        if seg is None:
            return None, '복귀 경로 생성 실패: %s' % info
        if mx > np.radians(1.0):
            self.get_logger().warn('복귀 구간 수직 이탈 최대 %.2f도 '
                                   '(출발 자세가 이미 기울어져 있음)' % np.degrees(mx))
        if self.checker is not None and not self.checker.check_path(seg)[0]:
            return None, '복귀 경로가 충돌'
        self.get_logger().info('복귀 %.2f초, 수직 이탈 최대 %.3f도' % (T, np.degrees(mx)))
        self.get_logger().info('  복귀 자세 q(deg)=%s   <- 저장하려면 '
                               "/pose_save 로 이름을 보내세요"
                               % np.degrees(seg[-1]).round(2))
        return seg, ''

    def _press_suction_retreat(self, traj, q_at, p_target):
        """
        traj 뒤에 [압착 하강] + [흡착 ON 이벤트] + [복귀] 를 붙인다.
        반환 (traj_full, events, mode, msg). 실패 시 (None, None, '', 사유)
        """
        p_press = np.asarray(p_target, float) - np.array([0.0, 0.0, self.press_mm / 1000.0])
        seg_press, m, msg = self._descend(q_at, p_target, p_press)
        if seg_press is None:
            # 출발 자세가 수직이 아니면(완화해로 도착한 경우) strict 하강이
            # 거부된다. 압착은 5mm 짧은 구간이라 자세를 그대로 유지한 채
            # 관절공간으로 내려가도 문제 없다.
            prev = self.strict_ends
            self.strict_ends = 'none'
            try:
                seg_press, m, msg = self._descend(q_at, p_target, p_press)
            finally:
                self.strict_ends = prev
            if seg_press is not None:
                self.get_logger().warn('압착 구간은 수직조건을 완화해서 생성했습니다.')
        if seg_press is None:
            return None, None, '', '압착 하강 실패: %s' % msg

        seg_ret, msg2 = self._retreat_seg(seg_press[-1])
        if seg_ret is None:
            return None, None, '', msg2

        full = np.vstack([traj, seg_press, seg_ret])
        idx_suction = len(traj) + len(seg_press) - 1
        events = [dict(idx=idx_suction, kind='suction_on', dwell=self.suction_dwell,
                       done=False, t0=None, settled=False, acted=False, tick=0)]
        return full, events, m, ''

    def _commit(self, traj, pause_at, events, w_high=None, w_low=None):
        """생성한 경로를 미리보기 상태로 등록."""
        self.traj = traj
        self.idx = 0
        self.pending = traj[-1]
        self.mode = 'preview'
        self.path_safe = True
        self._clear_pause()
        self.pause_at = int(pause_at)
        self.events = [dict(e) for e in (events or [])]
        self._plan_traj = traj.copy()
        self._plan_pause_at = int(pause_at)
        self._plan_events = [dict(e) for e in (events or [])]
        self._play_t0 = None
        if w_high is not None:
            p.resetBasePositionAndOrientation(self.marker, list(w_high), [0, 0, 0, 1])
        if w_low is not None:
            p.resetBasePositionAndOrientation(self.marker2, list(w_low), [0, 0, 0, 1])

    def _build_pick_sequence(self, w_high, w_low):
        """
        공정: 1차점 접근 -> 정지 -> 2차점 하강 -> press_mm 추가 하강
              -> 흡착 ON 유지 -> 1차점까지 수직 유지 상승

        상승 구간은 하강 경로를 그대로 뒤집어 쓴다. 같은 자세열을 거꾸로
        지나가므로 수직조건과 도달 가능성이 자동으로 보장되고, 관절 속도도
        이미 한계 안으로 재배분된 값이라 그대로 쓸 수 있다.

        반환 True/False
        """
        p_high = self.world_to_arm(w_high)
        p_low = self.world_to_arm(w_low)
        p_press = p_low - np.array([0.0, 0.0, self.press_mm / 1000.0])

        rep = fix_reach_report(p_press)
        if not rep['ok'] and self.strict_ends == 'both':
            self.get_logger().error('압착점(2차점 -%.1fmm)이 수직 도달 불가. %s'
                                    % (self.press_mm, rep['msg']))
            return False

        # 1차점 자세 후보
        cands = ik_fix_candidates(p_high, q0=self.q_cmd)
        if not cands:
            q_soft, e_soft, lv_soft = ik_level_soft(p_high, self.q_cmd)
            if e_soft > 1e-4 or lv_soft > self.level_tol:
                self.get_logger().error('1차점 도달 불가. %s'
                                        % fix_reach_report(p_high)['msg'])
                return False
            self.get_logger().warn('1차점이 수직에서 %.1f도 기울어집니다.'
                                   % np.degrees(lv_soft))
            cands = [(q_soft, e_soft)]

        safe = [q for q, _e in cands
                if self.checker is None or self.checker.check_config(q)[0]]
        if not safe:
            self.get_logger().error('1차점 도착 자세가 모두 충돌.')
            return False
        safe.sort(key=lambda qg: np.linalg.norm(wrap_to_pi(qg - self.q_cmd)))

        last = ''
        for gi, q_a in enumerate(safe):
            seg1, T1, _n = plan_joint_path(self.q_cmd, q_a, hz=self.sim_hz,
                                           speed_deg_per_s=self.speed)
            if self.checker is not None and not self.checker.check_path(seg1)[0]:
                last = '접근 경로 충돌'; continue

            seg2, m2, msg2 = self._descend(q_a, p_high, p_low)
            if seg2 is None:
                last = '하강1: %s' % msg2; continue

            seg3, m3, msg3 = self._descend(seg2[-1], p_low, p_press)
            if seg3 is None:
                last = '압착 하강: %s' % msg3; continue

            down = np.vstack([seg2, seg3])
            up = down[::-1].copy()                 # 상승 = 하강의 역순
            traj = np.vstack([seg1, down, up])

            if self.checker is not None and not self.checker.check_path(traj)[0]:
                last = '전체 경로 충돌'; continue

            n1 = len(seg1)
            n_press_end = n1 + len(down) - 1       # 압착 완료 지점
            self.traj = traj
            self.idx = 0
            self.pending = traj[-1]
            self.mode = 'preview'
            self.path_safe = True
            self._clear_pause()
            self.pause_at = n1                     # 1차점 정지
            self.events = [dict(idx=n_press_end, kind='suction_on',
                                dwell=self.suction_dwell, done=False, t0=None,
                                settled=False, acted=False, tick=0)]
            self._plan_traj = traj.copy()
            self._plan_pause_at = n1
            self._plan_events = [dict(e) for e in self.events]
            self._play_t0 = None

            p.resetBasePositionAndOrientation(self.marker, w_high.tolist(), [0, 0, 0, 1])
            p.resetBasePositionAndOrientation(self.marker2, w_low.tolist(), [0, 0, 0, 1])
            if self._pair_line is not None:
                p.removeUserDebugItem(self._pair_line)
            self._pair_line = p.addUserDebugLine(w_high.tolist(), w_low.tolist(),
                                                 lineColorRGB=[0, 0.8, 1], lineWidth=2.0)
            if gi > 0:
                self.get_logger().info('충돌 없는 해 채택(%d/%d).' % (gi + 1, len(safe)))
            self.get_logger().info(
                '픽 시퀀스: 접근 -> %.1fs 정지 -> 하강(%s) -> 압착 %.1fmm(%s) '
                '-> 흡착 ON %.1fs -> 상승  (총 %d 프레임)'
                % (self.dwell_sec, m2, self.press_mm, m3, self.suction_dwell, len(traj)))
            return True

        self.get_logger().error('픽 시퀀스 생성 실패: %s' % (last or '원인 불명'))
        return False

    def _build_pair_retreat(self, w_high, w_low):
        """
        폴백 A: 두 점 경로 + 압착 + 흡착 + 복귀(팔 기준 고정점).

        _build_pick_sequence 와 달리 1차점으로 되돌아가지 않고,
        retreat_arm_xyz 로 빠진다. 접근 자세의 수직 제약도 'end' 로
        완화해서(2차점만 엄격) 성공 확률을 높인다.
        """
        p_high = self.world_to_arm(w_high)
        p_low = self.world_to_arm(w_low)

        cands = ik_fix_candidates(p_high, q0=self.q_cmd)
        if not cands:
            q_soft, e_soft, lv_soft = ik_level_soft(p_high, self.q_cmd)
            if e_soft > 1e-4 or lv_soft > self.level_tol:
                return False, '1차점 도달 불가: %s' % fix_reach_report(p_high)['msg']
            self.get_logger().warn('[폴백] 1차점이 수직에서 %.1f도 기울어집니다.'
                                   % np.degrees(lv_soft))
            cands = [(q_soft, e_soft)]

        safe = [q for q, _e in cands
                if self.checker is None or self.checker.check_config(q)[0]]
        if not safe:
            return False, '1차점 도착 자세가 모두 충돌'
        safe.sort(key=lambda qg: np.linalg.norm(wrap_to_pi(qg - self.q_cmd)))

        last = ''
        for q_a in safe:
            seg1, T1, _n = plan_joint_path(self.q_cmd, q_a, hz=self.sim_hz,
                                           speed_deg_per_s=self.speed)
            if self.checker is not None and not self.checker.check_path(seg1)[0]:
                last = '접근 경로 충돌'; continue
            seg2, m2, msg2 = self._descend(q_a, p_high, p_low)
            if seg2 is None:
                last = '하강 실패: %s' % msg2; continue

            traj = np.vstack([seg1, seg2])
            full, events, m3, msg3 = self._press_suction_retreat(traj, seg2[-1], p_low)
            if full is None:
                last = msg3; continue
            if self.checker is not None and not self.checker.check_path(full)[0]:
                last = '전체 경로 충돌'; continue

            self._commit(full, len(seg1), events, w_high, w_low)
            self.get_logger().info(
                '  구간 프레임: 접근 %d / 하강 %d / 압착+복귀 %d = 총 %d (%.1f초)'
                % (len(seg1), len(seg2), len(full) - len(seg1) - len(seg2),
                   len(full), len(full) / self.sim_hz))
            self.get_logger().info(
                '[폴백A] 두 점 + 압착: 접근 -> %.1fs 정지 -> 하강(%s) -> 압착 %.1fmm(%s) '
                '-> 흡착 ON %.1fs -> 복귀 %s  (%d 프레임)'
                % (self.dwell_sec, m2, self.press_mm, m3, self.suction_dwell,
                   self.retreat_arm.round(3), len(full)))
            return True, ''
        return False, last or '원인 불명'

    def _goto_single(self, w_target):
        """
        폴백 B: 파지점 한 점 + 압착 + 흡착 + 복귀.

        접근점을 포기하고 파지점으로 바로 간다. 도착 자세는 수직을
        우선하되(ik_fix_candidates), 없으면 완화해를 쓴다.
        """
        p_low = self.world_to_arm(w_target)

        cands = ik_fix_candidates(p_low, q0=self.q_cmd)
        if cands:
            cands.sort(key=lambda c: np.linalg.norm(wrap_to_pi(c[0] - self.q_cmd)))
            q_list = [c[0] for c in cands]
        else:
            q_soft, e_soft, lv_soft = ik_level_soft(p_low, self.q_cmd)
            if e_soft > 1e-4:
                self.get_logger().error('[폴백B] 파지점 도달 불가: %s'
                                        % fix_reach_report(p_low)['msg'])
                return False
            self.get_logger().warn('[폴백B] 파지점이 수직에서 %.1f도 기울어집니다.'
                                   % np.degrees(lv_soft))
            q_list = [q_soft]

        last = ''
        for q_b in q_list:
            if self.checker is not None and not self.checker.check_config(q_b)[0]:
                last = '도착 자세 충돌'; continue
            seg1, T1, _n = plan_joint_path(self.q_cmd, q_b, hz=self.sim_hz,
                                           speed_deg_per_s=self.speed)
            if self.vmax is not None and self.limit_joint_speed:
                out, _T, _i = retime_path(seg1, self.vmax, self.sim_hz,
                                          margin=self.speed_margin)
                if out is not None:
                    seg1 = out
            if self.checker is not None and not self.checker.check_path(seg1)[0]:
                last = '이동 경로 충돌'; continue

            full, events, m3, msg3 = self._press_suction_retreat(seg1, q_b, p_low)
            if full is None:
                last = msg3; continue
            if self.checker is not None and not self.checker.check_path(full)[0]:
                last = '전체 경로 충돌'; continue

            self._commit(full, -1, events, w_target, w_target)
            self.get_logger().info(
                '  구간 프레임: 이동 %d / 압착+복귀 %d = 총 %d (%.1f초)'
                % (len(seg1), len(full) - len(seg1), len(full), len(full) / self.sim_hz))
            self.get_logger().info(
                '[폴백B] 단일점 + 압착: 이동 -> 압착 %.1fmm(%s) -> 흡착 ON %.1fs '
                '-> 복귀 %s  (%d 프레임)'
                % (self.press_mm, m3, self.suction_dwell,
                   self.retreat_arm.round(3), len(full)))
            return True

        self.get_logger().error('[폴백B] 단일점 경로 실패: %s' % (last or '원인 불명'))
        return False

    # ================= 티칭 (조그 / 자세 저장) =================
    #
    # 복귀 자세를 손으로 다듬어 저장해 두면, 나중에 하드코딩 경로로 쓸 수 있다.
    # 조그는 미리보기 상태(idle/await)에서만 받는다. 경로 재생 중에는 무시.
    #
    #   ros2 topic pub --once /jog_joint  std_msgs/msg/Float64MultiArray "{data: [0,0,5,0]}"
    #   ros2 topic pub --once /goto_joint std_msgs/msg/Float64MultiArray "{data: [0,-27.7,-64.5,-87.9]}"
    #   ros2 topic pub --once /pose_save  std_msgs/msg/String "{data: 'retreat'}"
    #   ros2 topic pub --once /pose_goto  std_msgs/msg/String "{data: '2'}"
    #   ros2 topic pub --once /pose_list  std_msgs/msg/String "{data: ''}"
    #   ros2 topic pub --once /pose_list  std_msgs/msg/String "{data: 'py'}"
    #   ros2 topic pub --once /pose_undo  std_msgs/msg/String "{data: ''}"
    #   ros2 topic pub --once /pose_clear std_msgs/msg/String "{data: 'yes'}"
    #   ros2 topic pub --once /home_joint std_msgs/msg/String "{data: '1,3'}"

    def _jog_ready(self):
        if not self.allow_jog:
            self.get_logger().warn('조그가 꺼져 있습니다 (-p allow_jog:=true).')
            return False
        if self.mode in ('preview', 'run') and self.traj is not None:
            self.get_logger().warn('경로 재생 중에는 조그할 수 없습니다.')
            return False
        return True

    def _move_direct(self, q_goal, tag):
        """조그/자세이동 전용. APPROVE 없이 바로 재생한다."""
        q_goal = wrap_to_pi(np.asarray(q_goal, float).ravel()[:N_JOINTS])
        if self.checker is not None and not self.checker.check_config(q_goal)[0]:
            self.get_logger().error('%s: 목표 자세가 충돌' % tag)
            return False
        seg, T, mx, info = plan_sync_joint_move(
            self.q_cmd, q_goal, hz=self.sim_hz,
            vmax=self.vmax if self.limit_joint_speed else None,
            margin=self.speed_margin)
        if seg is None:
            self.get_logger().error('%s: 경로 생성 실패 (%s)' % (tag, info))
            return False
        if self.checker is not None and not self.checker.check_path(seg)[0]:
            self.get_logger().error('%s: 경로 충돌' % tag)
            return False

        self.traj = seg
        self.idx = 0
        self.pending = q_goal
        self.path_safe = True
        self._clear_pause()
        self._plan_traj = seg.copy()
        self._plan_pause_at = -1
        self._plan_events = []
        self._play_t0 = None
        self._last_cmd_t = 0.0
        self.mode = 'run' if self.bridge else 'preview'
        self.get_logger().info('%s -> q(deg)=%s  (%.2f초, %d 프레임)'
                               % (tag, np.degrees(q_goal).round(2), T, len(seg)))
        return True

    def on_jog_joint(self, msg):
        """상대 조그 [dq1..dq4] (도)."""
        if not self._jog_ready():
            return
        d = list(msg.data)
        if len(d) < N_JOINTS:
            self.get_logger().error('jog_joint 은 값 %d개가 필요합니다 (도).' % N_JOINTS)
            return
        dq = np.radians(np.array(d[:N_JOINTS], float))
        if float(np.max(np.abs(dq))) > self.jog_max:
            self.get_logger().error('한 번에 %.1f도를 넘을 수 없습니다 (-p jog_max_deg).'
                                    % np.degrees(self.jog_max))
            return
        self._move_direct(self.q_cmd + dq, '조그')

    def on_goto_joint(self, msg):
        """절대 관절각 [q1..q4] (도)."""
        if not self._jog_ready():
            return
        d = list(msg.data)
        if len(d) < N_JOINTS:
            self.get_logger().error('goto_joint 은 값 %d개가 필요합니다 (도).' % N_JOINTS)
            return
        self._move_direct(np.radians(np.array(d[:N_JOINTS], float)), '관절 이동')

    def on_home_joint(self, msg):
        """
        지정한 관절만 0 으로 보낸다. 나머지는 현재 각도를 유지.

          ros2 topic pub --once /home_joint std_msgs/msg/String "{data: '1'}"
          ros2 topic pub --once /home_joint std_msgs/msg/String "{data: '1,3'}"
          ros2 topic pub --once /home_joint std_msgs/msg/String "{data: 'all'}"

        조그와 달리 각도 제한(jog_max_deg)을 두지 않는다. 목적지가 0 으로
        정해져 있어 예상 못한 큰 이동이 생기지 않기 때문이다.
        다만 충돌검사와 속도 재배분은 그대로 거친다.
        """
        if not self._jog_ready():
            return

        raw = str(msg.data).strip().lower()
        if raw in ('all', '*', ''):
            sel = list(range(N_JOINTS))
        else:
            sel = []
            for tok in raw.replace(' ', '').split(','):
                if not tok:
                    continue
                if not tok.isdigit():
                    self.get_logger().error("home_joint: '%s' 는 숫자가 아닙니다. "
                                            "예: '1' / '1,3' / 'all'" % tok)
                    return
                k = int(tok)
                if not (1 <= k <= N_JOINTS):
                    self.get_logger().error('home_joint: 관절 번호는 1~%d (받은 값 %d)'
                                            % (N_JOINTS, k))
                    return
                sel.append(k - 1)
            sel = sorted(set(sel))
        if not sel:
            self.get_logger().error("home_joint: 보낼 관절을 지정하세요. 예: '1' / '1,3' / 'all'")
            return

        q_goal = np.asarray(self.q_cmd, float).ravel()[:N_JOINTS].copy()
        moved = []
        for k in sel:
            if abs(float(q_goal[k])) > np.radians(0.05):
                moved.append('q%d %.2f->0' % (k + 1, np.degrees(q_goal[k])))
            q_goal[k] = 0.0

        if not moved:
            self.get_logger().info('home_joint: 지정한 관절이 이미 0 입니다.')
            return

        self.get_logger().info('관절 원점 복귀: %s' % ', '.join(moved))
        if 0 in sel and 1 in sel and 2 in sel and 3 in sel:
            self.get_logger().warn('  전 관절 0 = 팔이 곧게 선 자세(특이점). '
                                   '흡착 중이면 부품이 떨어집니다.')
        self._move_direct(q_goal, 'home_joint')

    # ---------- 자세 파일 (누적 목록) ----------
    #
    # 한 파일에 순번을 붙여 계속 쌓는다. 같은 이름이어도 덮어쓰지 않는다.
    #   poses:
    #     - {index: 1, label: approach, q_deg: [...], q_rad: [...], ...}
    #     - {index: 2, label: pick,     q_deg: [...], ...}
    # 나중에 이 순서대로 재생하면 하드코딩 경로가 된다.

    def _load_poses(self):
        """저장 파일을 리스트로 읽는다. 예전 dict 형식도 리스트로 변환."""
        import yaml
        if not os.path.exists(self.pose_file):
            return []
        try:
            with open(self.pose_file) as fp:
                data = yaml.safe_load(fp) or {}
        except Exception as e:
            self.get_logger().error('자세 파일 읽기 실패: %s' % e)
            return []

        if isinstance(data, dict) and 'poses' in data:
            lst = data.get('poses') or []
        elif isinstance(data, dict):
            lst = [data[k] for k in sorted(data)]      # 예전 이름-키 형식
        elif isinstance(data, list):
            lst = data
        else:
            lst = []
        for i, d in enumerate(lst):
            d['index'] = i + 1
        return lst

    def _write_poses(self, lst):
        import yaml
        for i, d in enumerate(lst):
            d['index'] = i + 1
        try:
            os.makedirs(os.path.dirname(self.pose_file) or '.', exist_ok=True)
            with open(self.pose_file, 'w') as fp:
                yaml.safe_dump({'poses': lst}, fp,
                               allow_unicode=True, sort_keys=False, default_flow_style=None)
            return True
        except Exception as e:
            self.get_logger().error('자세 저장 실패: %s' % e)
            return False

    def pose_snapshot(self, label):
        """현재 자세를 dict 로. 하드코딩에 필요한 정보를 함께 남긴다."""
        q = np.asarray(self.q_cmd, float).ravel()[:N_JOINTS]
        p_arm = fk_pos(q)
        return {
            'index': 0,
            'label': str(label),
            'q_deg': [round(float(v), 4) for v in np.degrees(q)],
            'q_rad': [round(float(v), 6) for v in q],
            'q5_deg': round(float(np.degrees(self.q5)), 4),
            'tcp_arm_m': [round(float(v), 5) for v in p_arm],
            'tcp_world_m': [round(float(v), 5) for v in self.arm_to_world(p_arm)],
            'rail_m': round(float(self.rail_m), 5),
            'level_err_deg': round(float(np.degrees(level_error(q))), 4),
            'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        }

    def on_pose_save(self, msg):
        """현재 자세를 목록 끝에 추가. 이름은 참고용이라 중복돼도 된다."""
        label = str(msg.data).strip() or 'p%s' % time.strftime('%H%M%S')
        lst = self._load_poses()
        lst.append(self.pose_snapshot(label))
        if not self._write_poses(lst):
            return
        d = lst[-1]
        self.get_logger().info("[%d] '%s' 저장 -> %s  (총 %d개)"
                               % (d['index'], d['label'], self.pose_file, len(lst)))
        self.get_logger().info('    q(deg)=%s  TCP(팔)=%s  레일 %.3fm  수직이탈 %.2f도'
                               % (d['q_deg'], d['tcp_arm_m'], d['rail_m'],
                                  d['level_err_deg']))

    def on_pose_goto(self, msg):
        """순번('2') 또는 이름으로 이동. 이름이 여러 개면 마지막 것."""
        if not self._jog_ready():
            return
        key = str(msg.data).strip()
        lst = self._load_poses()
        if not lst:
            self.get_logger().error('저장된 자세가 없습니다 (%s)' % self.pose_file)
            return

        target = None
        if key.isdigit():
            i = int(key)
            if 1 <= i <= len(lst):
                target = lst[i - 1]
            else:
                self.get_logger().error('순번 %d 없음 (1~%d)' % (i, len(lst)))
                return
        else:
            for d in lst:
                if str(d.get('label')) == key:
                    target = d
            if target is None:
                self.get_logger().error("이름 '%s' 없음. 목록: %s"
                                        % (key, ', '.join(str(d.get('label')) for d in lst)))
                return

        self._move_direct(np.array(target['q_rad'], float),
                          "[%d] '%s'" % (target['index'], target.get('label')))

    def on_pose_list(self, msg):
        """목록 출력. data 가 'py' 면 하드코딩용 파이썬 코드로 찍는다."""
        lst = self._load_poses()
        if not lst:
            self.get_logger().info('저장된 자세 없음 (%s)' % self.pose_file)
            return

        if str(msg.data).strip().lower() == 'py':
            self.get_logger().info('# 하드코딩용 (%s)' % self.pose_file)
            self.get_logger().info('PATH_Q = [')
            for d in lst:
                self.get_logger().info('    %s,   # [%d] %s'
                                       % (d['q_rad'], d['index'], d.get('label')))
            self.get_logger().info(']')
            return

        self.get_logger().info('저장된 자세 %d개 (%s)' % (len(lst), self.pose_file))
        for d in lst:
            self.get_logger().info('  [%d] %-14s q(deg)=%s  TCP(팔)=%s  레일 %.3f'
                                   % (d['index'], d.get('label'), d['q_deg'],
                                      d['tcp_arm_m'], d.get('rail_m', 0.0)))

    def on_pose_undo(self, msg):
        """마지막 항목 삭제."""
        lst = self._load_poses()
        if not lst:
            self.get_logger().warn('삭제할 자세가 없습니다.')
            return
        d = lst.pop()
        if self._write_poses(lst):
            self.get_logger().info("마지막 [%d] '%s' 삭제 (남은 %d개)"
                                   % (d['index'], d.get('label'), len(lst)))

    def on_pose_clear(self, msg):
        """전체 삭제. 실수 방지를 위해 data 가 'yes' 일 때만."""
        if str(msg.data).strip().lower() != 'yes':
            self.get_logger().warn("전체 삭제하려면 data: 'yes' 로 보내세요.")
            return
        if self._write_poses([]):
            self.get_logger().info('자세 목록 전체 삭제')

    # ---------- 관절 속도 한계 ----------
    def _calc_speed_limits(self):
        """
        브리지의 PROFILE_VELOCITY 와 기어비로 관절별 최대 각속도를 구한다.

        모터가 못 내는 속도를 명령하면 관절마다 뒤처지는 정도가 달라져
        수직조건이 깨지고 움직임이 끊겨 보인다. 그래서 경로 생성 단계에서
        미리 속도를 낮춘다.
        """
        if not self.limit_joint_speed:
            return
        try:
            from .dxl_bridge_suction_5axis import PROFILE_VELOCITY, JOINT_MOTORS
            gears = [j['gear'] for j in JOINT_MOTORS[:N_JOINTS]]
            self.vmax = joint_speed_limits(PROFILE_VELOCITY, gears)
            self.get_logger().info('관절 속도한계 q1~q4 = %s deg/s'
                                   % np.degrees(self.vmax).round(2))
        except Exception as e:
            self.vmax = None
            self.get_logger().warn('속도한계 계산 실패, 제한 없이 진행: %s' % e)

    # ---------- 두 점 수직자세 경로 ----------
    def on_target_pair(self, msg):
        """
        Float64MultiArray [x1,y1,z1, x2,y2,z2] (월드 좌표).
        z 가 높은 점이 1차점, 낮은 점이 2차점.
          1차점에 수직자세로 도달 -> dwell_sec 정지 -> 2차점까지 수직 유지 하강.
        """
        d = list(msg.data)
        if len(d) < 6:
            self.get_logger().error('target_pair 는 값 6개가 필요합니다 [x1,y1,z1,x2,y2,z2].')
            return False

        w_a = np.array(d[0:3], float)
        w_b = np.array(d[3:6], float)
        w_high, w_low = sort_by_height(w_a, w_b)
        if abs(w_high[2] - w_low[2]) < 1e-6:
            self.get_logger().warn('두 점의 z 가 같습니다. 수평 이동이 됩니다.')

        p_high = self.world_to_arm(w_high)
        p_low = self.world_to_arm(w_low)

        # 1) 1차점 fix IK 후보
        cands = ik_fix_candidates(p_high, q0=self.q_cmd)
        if not cands:
            rep_h = fix_reach_report(p_high)
            if self.strict_ends == 'both':
                q, e = ik_fix(p_high, q0=self.q_cmd)
                self.get_logger().error('1차점 수직자세 도달 불가 (잔차 %.4f m). %s'
                                        % (e, rep_h['msg']))
                self.get_logger().error('  2차점: %s' % fix_reach_report(p_low)['msg'])
                self.get_logger().error("  1차점 기울기를 허용하려면 -p strict_ends:=end")
                return False
            # 완화: 위치를 1순위로 맞추고 자세를 허용치 안에서 기울인다
            q_soft, e_soft, lv_soft = ik_level_soft(p_high, self.q_cmd)
            if e_soft > 1e-4:
                self.get_logger().error('1차점 위치조차 도달 불가 (잔차 %.4f m). %s'
                                        % (e_soft, rep_h['msg']))
                return False
            if lv_soft > self.level_tol:
                self.get_logger().error(
                    '1차점 기울기 %.1f도 > 허용 %.1f도. %s'
                    % (np.degrees(lv_soft), np.degrees(self.level_tol), rep_h['msg']))
                return False
            self.get_logger().warn('완화 모드: 1차점이 수직에서 %.1f도 기울어집니다.'
                                   % np.degrees(lv_soft))
            cands = [(q_soft, e_soft)]

        # 2) 도착자세 충돌검사 -> 현재자세에서 가까운 순
        safe = [q for q, _e in cands
                if self.checker is None or self.checker.check_config(q)[0]]
        if not safe:
            self.get_logger().error('1차점 도착 자세가 모두 충돌.')
            return False
        safe.sort(key=lambda qg: np.linalg.norm(wrap_to_pi(qg - self.q_cmd)))

        # 3) 후보별로 [접근 + 정지 + 하강] 전체 경로 생성/검사
        chosen = None
        last_msg = ''
        for gi, q_a in enumerate(safe):
            seg1, T1, _n = plan_joint_path(self.q_cmd, q_a, hz=self.sim_hz,
                                           speed_deg_per_s=self.speed)
            if self.checker is not None and not self.checker.check_path(seg1)[0]:
                last_msg = '접근 경로 충돌'
                continue

            # 형상 생성 후 관절 속도한계에 맞춰 '구간별로' 시간을 재배분한다.
            # 전체를 균일하게 늦추면 경계 근처 한 지점 때문에 전 구간이 느려진다.
            seg2, T2, mx_lv, msg2 = plan_descent_ends_retimed(
                q_a, p_high, p_low, hz=self.sim_hz,
                lin_speed=self.descend_speed,
                mode=self.strict_ends, level_tol=self.level_tol,
                vmax=self.vmax if self.limit_joint_speed else None,
                margin=self.speed_margin)
            desc_mode = '직선'
            if seg2 is None and self.descent_fallback_joint:
                # 직선 하강 실패 -> 관절공간 하강으로 재시도.
                # 두 끝점이 모두 수직이면 공통 시간 프로파일 보간이
                # q2+q3+q4 = pi 를 그대로 보존하므로 수직은 유지된다.
                # 잃는 것은 TCP 가 직선으로 내려가지 않는다는 점뿐.
                self.get_logger().warn('직선 하강 실패(%s) -> 관절공간 하강으로 재시도'
                                       % (msg2 or '원인 불명'))
                seg2, T2, mx_lv, msg2 = plan_joint_descent_retimed(
                    q_a, p_high, p_low, hz=self.sim_hz,
                    lin_speed=self.descend_speed,
                    vmax=self.vmax if self.limit_joint_speed else None,
                    margin=self.speed_margin)
                desc_mode = '관절공간'

            if seg2 is None:
                last_msg = msg2
                continue
            if msg2:
                self.get_logger().info(msg2)
            T2 = float(T2)
            if self.checker is not None and not self.checker.check_path(seg2)[0]:
                last_msg = '하강 경로 충돌'
                continue

            # 정지는 프레임 반복이 아니라 벽시계 시간으로 처리한다.
            # (프레임 반복은 실로봇이 명령보다 뒤처질 때 실제 정지로 이어지지 않는다)
            traj = np.vstack([seg1, seg2])
            chosen = (seg2[-1], traj, T1 + self.dwell_sec + T2, T1, T2, len(seg1))
            if gi > 0:
                self.get_logger().info('충돌 없는 해 채택(%d/%d).' % (gi + 1, len(safe)))
            break

        if chosen is None:
            self.get_logger().error('두 점 경로 생성 실패: %s' % (last_msg or '원인 불명'))
            return False

        q_goal, self.traj, T, T1, T2, n1 = chosen
        self.idx = 0
        self.pending = q_goal
        self.mode = 'preview'
        self.path_safe = True
        self._clear_pause()
        self.pause_at = int(n1)          # 접근 끝 = 1차점. 여기서 멈춘다.
        self._plan_traj = self.traj.copy()
        self._plan_pause_at = int(n1)
        self._play_t0 = None

        p.resetBasePositionAndOrientation(self.marker, w_high.tolist(), [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(self.marker2, w_low.tolist(), [0, 0, 0, 1])
        if self._pair_line is not None:
            p.removeUserDebugItem(self._pair_line)
        self._pair_line = p.addUserDebugLine(w_high.tolist(), w_low.tolist(),
                                             lineColorRGB=[0, 0.8, 1], lineWidth=2.0)

        # 수직조건 유지 확인 (경로 전체 최대 이탈각)
        lv = max(level_error(q) for q in self.traj[n1:])
        self.get_logger().info(
            '두 점 경로%s: 1차 %s -> %.1fs 접근, %.1fs 정지, %.1fs 하강 -> 2차 %s '
            '(총 %.2fs, 수직 이탈 최대 %.2f도)'
            % (' [%s, 허용 %.0f도, 하강 %s]'
               % (self.strict_ends, np.degrees(self.level_tol), desc_mode),
               w_high.round(3), T1, self.dwell_sec, T2, w_low.round(3), T,
               np.degrees(lv)))
        return True

    # ---------- 좌표 -> IK 해 선택 -> 경로 -> 미리보기 ----------
    def on_target(self, msg):
        target_world = np.array([msg.x, msg.y, msg.z], float)
        target = self.world_to_arm(target_world)

        if self.fix_mode:
            cands = ik_fix_candidates(target, q0=self.q_cmd)
            if not cands:
                q, e = ik_fix(target, q0=self.q_cmd)
                cands = [(wrap_to_pi(q), e)]
                self.get_logger().warn('fix IK 수렴 약함(잔차 %.2e).' % e)

            safe = []
            for q_goal, err in cands:
                if self.checker is None or self.checker.check_config(q_goal)[0]:
                    safe.append(q_goal)
            if not safe:
                self.get_logger().error('도착 자세가 모두 충돌 -> 다른 좌표 시도.')
                return
            safe.sort(key=lambda qg: np.linalg.norm(wrap_to_pi(qg - self.q_cmd)))
            chosen = None
            for gi, q_goal in enumerate(safe):
                traj, T, n = plan_joint_path(self.q_cmd, q_goal, hz=self.sim_hz,
                                             speed_deg_per_s=self.speed)
                if self.checker is None or self.checker.check_path(traj)[0]:
                    chosen = (q_goal, traj, T)
                    if gi > 0:
                        self.get_logger().info('충돌 없는 해 채택(%d/%d).' % (gi + 1, len(safe)))
                    break
            if chosen is None:
                self.get_logger().error('도착 자세는 되나 경로가 모두 충돌 -> 다른 좌표 시도.')
                return
            self.path_safe = True
        else:
            q_goal, err = ik(target, q0=self.q_cmd)
            q_goal = wrap_to_pi(q_goal)
            if err > 1e-3:
                self.get_logger().warn('IK 잔차 큼(%.2e): 작업영역 밖/특이점 가능.' % err)
            traj, T, n = plan_joint_path(self.q_cmd, q_goal, hz=self.sim_hz,
                                         speed_deg_per_s=self.speed)
            self.path_safe = True
            if self.checker is not None and not self.checker.check_path(traj)[0]:
                self.path_safe = False
                self.get_logger().error('경로 충돌 위험 -> APPROVE 차단.')
            chosen = (q_goal, traj, T)

        q_goal, self.traj, T = chosen
        self.idx = 0; self.pending = q_goal; self.mode = 'preview'
        self._clear_pause()
        self._plan_traj = self.traj.copy(); self._plan_pause_at = -1
        self._play_t0 = None
        p.resetBasePositionAndOrientation(self.marker, target_world.tolist(), [0, 0, 0, 1])
        self.get_logger().info('미리보기: world=%s (arm=%s) q(deg)=%s (%.2fs), rail=%.3fm.'
                               % (target_world.round(3), target.round(3),
                                  np.degrees(q_goal).round(1), T, self.rail_m))

    def plan_to_home(self):
        q_goal = np.zeros(N_JOINTS)
        self.traj, T, n = plan_joint_path(self.q_cmd, q_goal, hz=self.sim_hz,
                                          speed_deg_per_s=self.speed)
        self.idx = 0; self.pending = q_goal; self.path_safe = True; self.mode = 'preview'
        self._clear_pause()
        self._plan_traj = self.traj.copy(); self._plan_pause_at = -1
        self._play_t0 = None
        self.get_logger().info('HOME(관절 0) 미리보기. APPROVE 로 이동하거나 새 좌표 전송.')

    # ---------- 메인 루프 ----------
    def step(self):
        # --- 레일 위치 갱신 (아두이노가 보고한 스텝수를 그대로 반영) ---
        if self.rail is not None:
            m = self.rail.position_m
            self.rail_m = float(np.clip(m, RAIL_MIN_M, RAIL_MAX_M))

        # --- 버튼 ---
        if self._click(self.b_stop, '_sl'):
            if self.bridge:
                self.bridge.emergency_stop()
            if self.rail is not None:
                self.rail.estop()
                self.rail.suction(False)
            self.traj = None; self.mode = 'idle'
            self.get_logger().error('EMERGENCY STOP (팔 + 레일)')
        if self._click(self.b_home, '_hl'):
            self.plan_to_home()
        if self._click(self.b_fix, '_fl'):
            self.fix_mode = not self.fix_mode
            self._show_fix_state()
            self.get_logger().info('FIX MODE = %s' % self.fix_mode)
        if self._click(self.b_rcw, '_rcw'):
            self._rail_do(lambda r: r.to_end(), '레일 -> 스트로크 끝')
        if self._click(self.b_rccw, '_rccw'):
            self._rail_do(lambda r: r.to_origin(), '레일 -> 원점')
        if self._click(self.b_rstop, '_rstop'):
            self._rail_do(lambda r: r.smooth_stop(), '레일 부드러운 정지')
        if self._click(self.b_restop, '_restop'):
            self._rail_do(lambda r: r.estop(), '레일 즉시 정지')
        if self._click(self.b_rzero, '_rzero'):
            self._rail_do(lambda r: r.zero(), '레일 현재 위치를 0 으로 설정')
        if self._click(self.b_von, '_von'):
            self._rail_do(lambda r: r.suction(True), '흡착 ON')
        if self._click(self.b_voff, '_voff'):
            self._rail_do(lambda r: r.suction(False), '흡착 OFF')
        if self._click(self.b_rgoto, '_rgoto'):
            tgt = float(p.readUserDebugParameter(self.s_rgoto))
            self._rail_do(lambda r: r.move_abs_m(tgt), '레일 -> %.3f m' % tgt)
        if self._click(self.b_appr, '_al'):
            if self.mode == 'await' and self.path_safe:
                if self._plan_traj is None:
                    self.get_logger().error('실행할 경로가 없습니다. 좌표를 다시 보내세요.')
                else:
                    # 미리보기가 끝나면 self.traj 는 None 이 되므로
                    # 계획 원본(self._plan_traj)에서 되살린다.
                    base = self._plan_traj
                    if self.bridge:
                        q_now = np.array(self.bridge.read_joints()[:N_JOINTS], float)
                        # 정지 지점이나 이벤트(압착/흡착)가 있는 경로는 절대
                        # 통째로 재계획하면 안 된다. 재계획하면 pending(=경로의
                        # 마지막 자세, 보통 복귀점)까지 직행해버려서 압착과
                        # 흡착이 통째로 사라진다.
                        if self._plan_pause_at >= 0 or self._plan_events:
                            # 두 점 수직경로: 경로 전체를 반드시 보존해야 한다.
                            # 통째로 재계획하면 1차점 정지와 수직조건이 사라진다.
                            # 실제 현재값 -> 경로 시작점 구간만 앞에 이어붙인다.
                            lead, _, _ = plan_joint_path(q_now, base[0], hz=self.sim_hz,
                                                         speed_deg_per_s=self.speed)
                            self.traj = np.vstack([lead, base])
                            self.pause_at = (self._plan_pause_at + len(lead)
                                             if self._plan_pause_at >= 0 else -1)
                            self.events = [dict(e, idx=e['idx'] + len(lead))
                                           for e in self._plan_events]
                        else:
                            self.traj, _, _ = plan_joint_path(q_now, self.pending,
                                                              hz=self.sim_hz,
                                                              speed_deg_per_s=self.speed)
                            self.pause_at = -1
                            self.events = []
                    else:
                        self.traj = base.copy()
                        self.pause_at = self._plan_pause_at
                        self.events = [dict(e) for e in self._plan_events]
                    # 미리보기에서 소비한 정지를 실행 때 다시 하도록 초기화
                    self._reset_pause_state()
                    self._play_t0 = None
                    self._last_cmd_t = 0.0
                    self.idx = 0; self.mode = 'run'
                    self.get_logger().info(
                        'APPROVE: 실행 시작 (%d 프레임, 정지 %s, 이벤트 %d개%s)'
                        % (len(self.traj),
                           '있음' if self.pause_at >= 0 else '없음',
                           len(self.events),
                           ''.join(' [%s@%d]' % (e['kind'], e['idx'])
                                   for e in self.events)))
            elif self.mode == 'await' and not self.path_safe:
                self.get_logger().error('충돌 위험 경로라 승인 불가.')

        # --- 경로 재생 (벽시계 시간 기준) ---
        #
        # 예전에는 타이머 콜백 1회당 인덱스를 1칸씩 올렸다. 그러면 콜백이
        # sim_hz 로 정확히 돌아야만 계획한 속도가 나오는데, llvmpipe 렌더링
        # 환경에서는 콜백 간격이 불규칙해서 움직임이 끊겨 보인다.
        # 이제는 경과 시간으로 인덱스를 직접 계산하므로, 프레임이 밀려도
        # 실제 이동 속도와 명령 간격이 일정하게 유지된다.
        if self.mode in ('preview', 'run') and self.traj is not None:
            now = time.time()

            if self._handle_events():
                self._play_t0 = now - self.idx / self.sim_hz
                self._render(force=True)
                return
            if self._handle_pause():
                # 정지 중에는 재생 시계를 현재 인덱스에 고정해 둔다
                self._play_t0 = now - self.idx / self.sim_hz
                self._render(force=True)
                return

            if self._play_t0 is None:
                self._play_t0 = now - self.idx / self.sim_hz

            last = len(self.traj) - 1
            tgt = int(round((now - self._play_t0) * self.sim_hz))
            tgt = max(self.idx, min(tgt, last))

            # 정지 지점을 건너뛰지 않도록 클램프
            if self.pause_at >= 0 and not self._pause_done and tgt > self.pause_at:
                tgt = self.pause_at
            ev = self._next_event()
            if ev is not None and tgt > ev['idx']:
                tgt = ev['idx']          # 이벤트 지점을 건너뛰지 않는다

            self.idx = tgt
            self.q_cmd = self.traj[self.idx]

            # 모터 명령도 시간 기준으로 일정 간격 전송
            if self.mode == 'run' and self.bridge:
                if (now - self._last_cmd_t) >= (1.0 / self.robot_cmd_hz):
                    self._last_cmd_t = now
                    self.bridge.command_joints(list(self.q_cmd) + [self.q5])

            if self.idx >= last:
                if self.mode == 'run' and self.bridge:
                    self.bridge.command_joints(list(self.q_cmd) + [self.q5])
                self.traj = None
                self._play_t0 = None
                self.mode = 'await' if self.mode == 'preview' else 'idle'
                if self.mode == 'await':
                    self.get_logger().info('미리보기 완료. APPROVE 로 실행하거나 새 좌표 전송.')

        self._show_loads()
        self._show_rail()
        self._render()

    def _render(self, force=False):
        """PyBullet 화면 갱신. 매 콜백마다 하면 느린 렌더러에서 부담이 크므로
        render_hz 로 솎아낸다. 경로 재생 자체는 시간 기준이라 영향 없음."""
        self._render_tick += 1
        need = max(1, int(round(self.sim_hz / self.render_hz)))
        if not force and self._render_tick < need:
            return
        self._render_tick = 0
        self._apply_sim(self.q_cmd)
        p.stepSimulation()

    # ---------- 1차점 정지 처리 ----------
    def _clear_pause(self):
        """정지 지점과 이벤트를 모두 없앤다 (단일 좌표 경로 등)."""
        self.pause_at = -1
        self.events = []
        self._plan_events = []
        self._reset_pause_state()

    def _reset_pause_state(self):
        """pause_at 은 유지한 채 정지 진행상태만 초기화.

        미리보기에서 정지를 한 번 소비하면 _pause_done 이 True 로 남아
        APPROVE 후 실행 때 멈추지 않는다. 그래서 재생 시작 시마다 초기화한다.
        """
        self._pause_t0 = None
        self._pause_done = False
        self._settled = False
        self._settle_tick = 0
        for ev in self.events:
            ev['done'] = False
            ev['t0'] = None
            ev['settled'] = False
            ev['acted'] = False
            ev['tick'] = 0

    def _next_event(self):
        """아직 처리 안 된 이벤트 중 가장 앞의 것."""
        for ev in self.events:
            if not ev['done']:
                return ev
        return None

    def _handle_events(self):
        """
        경로 중간 이벤트(정지 / 흡착 on·off) 처리.
        아직 머물러야 하면 True 를 돌려준다.

        정지와 마찬가지로 실로봇이 명령보다 뒤처지므로, 실제 관절각이
        해당 지점에 도달한 뒤에 대기 시간을 세기 시작한다.
        """
        ev = self._next_event()
        if ev is None or self.idx != ev['idx']:
            return False

        self.q_cmd = self.traj[self.idx]
        now = time.time()

        if ev['t0'] is None:
            ev['t0'] = now
            ev['settled'] = (self.bridge is None or self.mode != 'run')
            ev['tick'] = 0
            ev['acted'] = False
            self.get_logger().info('[%s] 지점 도달 -> %.2f초 대기' % (ev['kind'], ev['dwell']))

        # 실로봇 도달 확인
        if not ev['settled']:
            ev['tick'] += 1
            if ev['tick'] >= int(self.sim_hz * 0.1):
                ev['tick'] = 0
                try:
                    qr = np.array(self.bridge.read_joints()[:N_JOINTS], float)
                    e = float(np.max(np.abs(wrap_to_pi(qr - self.q_cmd))))
                    if e <= self.settle_tol:
                        ev['settled'] = True; ev['t0'] = now
                        self.get_logger().info('  실로봇 도달 확인 (오차 %.2f deg)' % np.degrees(e))
                    elif (now - ev['t0']) > self.settle_timeout:
                        ev['settled'] = True; ev['t0'] = now
                        self.get_logger().warn('  도달 대기 시간초과 (오차 %.2f deg). 진행합니다.'
                                               % np.degrees(e))
                except Exception as ex:
                    ev['settled'] = True; ev['t0'] = now
                    self.get_logger().warn('  관절 읽기 실패, 대기 생략: %s' % ex)
            if self.mode == 'run' and self.bridge:
                self.bridge.command_joints(list(self.q_cmd) + [self.q5])
            return True

        # 도달 후 동작 1회 실행
        if not ev['acted']:
            ev['acted'] = True
            if ev['kind'] == 'suction_on':
                if self.rail is not None and self.mode == 'run':
                    self.rail.suction(True)
                    self.get_logger().info('  흡착 ON')
                else:
                    self.get_logger().info('  흡착 ON (미리보기라 실제 전송 안 함)')
            elif ev['kind'] == 'suction_off':
                if self.rail is not None and self.mode == 'run':
                    self.rail.suction(False)
                    self.get_logger().info('  흡착 OFF')

        if (now - ev['t0']) < ev['dwell']:
            if self.mode == 'run' and self.bridge:
                self.bridge.command_joints(list(self.q_cmd) + [self.q5])
            return True

        ev['done'] = True
        self.get_logger().info('  대기 완료 -> 다음 구간')
        return False

    def _handle_pause(self):
        """
        1차점에서 멈춰야 하면 True 를 돌려준다 (이번 프레임은 idx 를 진행하지 않음).

        실로봇일 때는 명령만 멈추는 것으로 부족하다. PROFILE_VELOCITY 때문에
        팔이 명령보다 뒤처져 있으므로, 실제 관절각이 1차점에 도달할 때까지
        기다린 뒤에 dwell 시간을 세기 시작한다.
        """
        if self.pause_at < 0 or self._pause_done or self.idx != self.pause_at:
            return False

        self.q_cmd = self.traj[self.idx]
        now = time.time()

        # 정지 진입
        if self._pause_t0 is None:
            self._pause_t0 = now
            self._settled = (self.bridge is None or self.mode != 'run')
            self.get_logger().info('1차점 도달 -> %.2f초 정지' % self.dwell_sec)

        # 실로봇이면 실제 도달 확인 (0.1초마다)
        if not self._settled:
            self._settle_tick += 1
            if self._settle_tick >= int(self.sim_hz * 0.1):
                self._settle_tick = 0
                try:
                    qr = np.array(self.bridge.read_joints()[:N_JOINTS], float)
                    e = float(np.max(np.abs(wrap_to_pi(qr - self.q_cmd))))
                    if e <= self.settle_tol:
                        self._settled = True
                        self._pause_t0 = now
                        self.get_logger().info('실로봇 1차점 도달 확인 (오차 %.2f deg)'
                                               % np.degrees(e))
                    elif (now - self._pause_t0) > self.settle_timeout:
                        self._settled = True
                        self._pause_t0 = now
                        self.get_logger().warn(
                            '1차점 도달 대기 시간초과 (오차 %.2f deg). 그대로 진행합니다.'
                            % np.degrees(e))
                except Exception as ex:
                    self._settled = True
                    self._pause_t0 = now
                    self.get_logger().warn('관절 읽기 실패, 대기 생략: %s' % ex)

            if self.mode == 'run' and self.bridge:
                self.bridge.command_joints(list(self.q_cmd) + [self.q5])
            return True

        # 도달했으면 dwell 시간 계수
        if (now - self._pause_t0) < self.dwell_sec:
            if self.mode == 'run' and self.bridge:
                self.bridge.command_joints(list(self.q_cmd) + [self.q5])
            return True

        self._pause_done = True
        self.get_logger().info('정지 완료 -> 수직 유지 하강 시작')
        return False

    # ---------- 유틸 ----------
    def _rail_do(self, fn, msg):
        if self.rail is None:
            self.get_logger().warn('레일 미연결 (use_rail:=true 로 실행).')
            return
        fn(self.rail)
        self.get_logger().info(msg)

    def _show_rail(self):
        """약 0.2초마다 레일 위치를 화면/토픽에 표시."""
        self._rail_tick += 1
        if self._rail_tick < int(self.sim_hz * 0.2):
            return
        self._rail_tick = 0

        msg = Float64(); msg.data = float(self.rail_m)
        self.rail_pub.publish(msg)

        if self.rail is None:
            txt = 'RAIL: 미연결 (0.000 m)'
            col = [0.6, 0.6, 0.6]
        else:
            steps = self.rail.steps
            txt = 'RAIL: %.3f m  (%d pulse)' % (self.rail_m, steps)
            col = [0.3, 1.0, 0.4]
            if not self.rail.zero_set:
                txt += '  [원점 미설정 - z 필요]'
                col = [1.0, 0.7, 0.2]
            elif self.rail.moving:
                txt += '  [이동중]'
            if self.rail.stale(5.0):
                txt += '  [수신 없음]'
                col = [1.0, 0.4, 0.4]
        if self._rail_txt is not None:
            p.removeUserDebugItem(self._rail_txt)
        self._rail_txt = p.addUserDebugText(txt, [0.0, 0.0, 1.20],
                                            textColorRGB=col, textSize=1.2)

        # 흡착 상태
        if self.rail is not None:
            son = self.rail.suction_on
            stxt = 'SUCTION: ON' if son else 'SUCTION: OFF'
            scol = [1.0, 0.3, 0.3] if son else [0.5, 0.5, 0.5]
        else:
            stxt = 'SUCTION: 미연결'
            scol = [0.5, 0.5, 0.5]
        if self._suc_txt is not None:
            p.removeUserDebugItem(self._suc_txt)
        self._suc_txt = p.addUserDebugText(stxt, [0.0, 0.0, 1.35],
                                           textColorRGB=scol, textSize=1.2)

    def _show_loads(self):
        if not self.bridge:
            return
        self._load_tick += 1
        if self._load_tick < int(self.sim_hz * 0.25):
            return
        self._load_tick = 0
        try:
            st = self.bridge.read_status()
        except Exception:
            return
        parts = []
        for mid in sorted(st):
            load, volt = st[mid]
            ls = '--' if load is None else '%+5.1f%%' % load
            vs = '--' if volt is None else '%.1fV' % volt
            parts.append('ID%d %s/%s' % (mid, vs, ls))
        txt = '모터 전압/부하  ' + '  '.join(parts)
        if self._load_txt is not None:
            p.removeUserDebugItem(self._load_txt)
        self._load_txt = p.addUserDebugText(txt, [0.0, 0.0, 0.9],
                                            textColorRGB=[0.9, 0.9, 0.2], textSize=1.1)

    def _show_fix_state(self):
        if self._fix_txt is not None:
            p.removeUserDebugItem(self._fix_txt)
        msg = 'FIX MODE: ON (말단 지면 고정)' if self.fix_mode else 'FIX MODE: OFF (방향 자유)'
        col = [1, 0.5, 0] if self.fix_mode else [0, 0.6, 1]
        self._fix_txt = p.addUserDebugText(msg, [0.0, 0.0, 1.05], textColorRGB=col, textSize=1.3)

    def _click(self, b, a):
        v = p.readUserDebugParameter(b)
        if v > getattr(self, a):
            setattr(self, a, v); return True
        return False

    def _apply_sim(self, q):
        if self.rail_jid is not None:
            p.resetJointState(self.robot, self.rail_jid, float(self.rail_m))
        full = list(q) + [self.q5] + [0.0] * (len(self.rev) - N_JOINTS - 1)
        for k, jid in enumerate(self.rev):
            p.resetJointState(self.robot, jid, float(full[k]))


def main(args=None):
    rclpy.init(args=args)
    node = ExecNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.bridge:
            node.bridge.emergency_stop(); node.bridge.close()
        if getattr(node, 'rail', None):
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
