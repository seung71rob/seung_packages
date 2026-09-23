"""
robot_description / ik_execute_centric_5axis.py
--------------------------------------------------
centric 그리퍼: 좌표 입력 -> IK 경로 -> PyBullet 미리보기(흔들림 없음)
-> APPROVE 승인 -> 실로봇 작동 / EMERGENCY STOP / GO TO HOME.
PyBullet 버튼으로 말단 고정(FIX MODE) on/off 를 실행 중에 토글할 수 있다.

fix 모드: 목표점의 여러 fix IK 해 중 (1) 도착 자세가 충돌 안 나고
(2) 현재 자세에서 가까운 순으로 골라 (3) 경로까지 충돌 없는 해를 채택.

파라미터: use_robot, fix_mode(초기값), check_collision, gripper_deg,
          sim_hz, robot_cmd_hz, speed_deg_per_s, dxl_port, mesh_dir

실행:
  ros2 run robot_description ik_execute_centric_5axis --ros-args -p use_robot:=false
좌표:
  ros2 topic pub --once /target_point geometry_msgs/msg/Point "{x: -0.25, y: 0.05, z: 0.35}"
"""
import os
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import pybullet as p
import pybullet_data

from .kinematics_ee_5axis import fk, ik, ik_fix, ik_fix_candidates, N_JOINTS, wrap_to_pi
from .trajectory_5axis import plan_joint_path
from .path_collision_guard_ee_5axis import CollisionChecker

URDF_NAME = 'robot_arm_5axis_centric.urdf'


class ExecNode(Node):
    def __init__(self):
        super().__init__('ik_execute_centric_5axis')
        self.declare_parameter('pkg_dir', os.path.expanduser('~/robot_sim/src/robot_description'))
        self.declare_parameter('mesh_dir', os.path.expanduser('~/robot_sim/src/robot_description/meshes_robot_5axis'))
        self.declare_parameter('use_robot', False)
        self.declare_parameter('fix_mode', False)
        self.declare_parameter('check_collision', True)
        self.declare_parameter('gripper_deg', 0.0)
        self.declare_parameter('sim_hz', 240.0)
        self.declare_parameter('robot_cmd_hz', 50.0)
        self.declare_parameter('speed_deg_per_s', 60.0)
        self.declare_parameter('dxl_port', '/dev/ttyUSB0')

        self.pkg_dir = self.get_parameter('pkg_dir').value
        self.mesh_dir = self.get_parameter('mesh_dir').value
        self.use_robot = bool(self.get_parameter('use_robot').value)
        self.fix_mode = bool(self.get_parameter('fix_mode').value)
        self.sim_hz = float(self.get_parameter('sim_hz').value)
        self.robot_cmd_hz = float(self.get_parameter('robot_cmd_hz').value)
        self.speed = float(self.get_parameter('speed_deg_per_s').value)
        self.q5 = np.radians(float(self.get_parameter('gripper_deg').value))

        urdf_abs = self._prep()
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)
        p.loadURDF('plane.urdf')
        p.resetDebugVisualizerCamera(1.2, 50, -25, [0.0, 0, 0.45])
        self.robot = p.loadURDF(urdf_abs, [0, 0, 0], useFixedBase=True)
        self.rev = [j for j in range(p.getNumJoints(self.robot))
                    if p.getJointInfo(self.robot, j)[2] == p.JOINT_REVOLUTE]
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.012, rgbaColor=[1, 0, 0, 0.9])
        self.marker = p.createMultiBody(0, baseVisualShapeIndex=vis, basePosition=[10, 10, 10])

        self.b_appr = p.addUserDebugParameter('APPROVE -> run robot', 1, 0, 0)
        self.b_stop = p.addUserDebugParameter('EMERGENCY STOP', 1, 0, 0)
        self.b_home = p.addUserDebugParameter('GO TO HOME', 1, 0, 0)
        self.b_fix = p.addUserDebugParameter('FIX MODE on/off (click)', 1, 0, 0)
        self._al = 0; self._sl = 0; self._hl = 0; self._fl = 0
        self._fix_txt = None
        self._load_txt = None
        self._load_tick = 0

        self.checker = None
        if bool(self.get_parameter('check_collision').value):
            self.checker = CollisionChecker(os.path.join(self.pkg_dir, 'urdf', URDF_NAME),
                                            self.pkg_dir, N_JOINTS, self.mesh_dir)

        self.bridge = None
        if self.use_robot:
            from .dxl_bridge_centric_5axis import DxlBridge
            self.bridge = DxlBridge(port=self.get_parameter('dxl_port').value)
            self.bridge.connect(); self.bridge.capture_home()
            self.get_logger().warn('실로봇: 로봇을 home(관절 0)에 두고 시작하세요.')

        self.q_cmd = np.zeros(N_JOINTS)
        self.traj = None; self.idx = 0; self.mode = 'idle'; self.path_safe = False
        self.pending = np.zeros(N_JOINTS)
        self.send_every = max(int(round(self.sim_hz / self.robot_cmd_hz)), 1)
        self._apply_sim(self.q_cmd)
        self._show_fix_state()

        self.create_subscription(Point, 'target_point', self.on_target, 10)
        self.create_timer(1.0 / self.sim_hz, self.step)
        self.get_logger().info('Ready (centric, fix_mode=%s, use_robot=%s).'
                               % (self.fix_mode, self.use_robot))

    def _prep(self):
        with open(os.path.join(self.pkg_dir, 'urdf', URDF_NAME)) as f:
            txt = f.read()
        txt = txt.replace('robot_meshes/', self.mesh_dir.rstrip('/') + '/')
        txt = txt.replace('package://robot_description/', self.pkg_dir + '/')
        out = '/tmp/' + URDF_NAME
        with open(out, 'w') as f:
            f.write(txt)
        return out

    # ---------- 좌표 -> IK 해 선택 -> 경로 -> 미리보기 ----------
    def on_target(self, msg):
        target = np.array([msg.x, msg.y, msg.z], float)

        if self.fix_mode:
            cands = ik_fix_candidates(target, q0=self.q_cmd)
            if not cands:
                q, e = ik_fix(target, q0=self.q_cmd)
                cands = [(wrap_to_pi(q), e)]
                self.get_logger().warn('fix IK 수렴 약함(잔차 %.2e).' % e)

            # 1) 도착 자세가 충돌 안 나는 후보만 (위치 검증)
            safe = []
            for q_goal, err in cands:
                if self.checker is None or self.checker.check_config(q_goal)[0]:
                    safe.append(q_goal)
            if not safe:
                self.get_logger().error('도착 자세가 모두 충돌 -> 다른 좌표 시도.')
                return
            # 2) 현재 자세에서 가까운 순
            safe.sort(key=lambda qg: np.linalg.norm(wrap_to_pi(qg - self.q_cmd)))
            # 3) 가까운 해부터 경로 생성 -> 경로 검사 -> 통과하는 첫 해
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
        p.resetBasePositionAndOrientation(self.marker, target.tolist(), [0, 0, 0, 1])
        self.get_logger().info('미리보기: target=%s q(deg)=%s (%.2fs).'
                               % (target.round(3), np.degrees(q_goal).round(1), T))

    def plan_to_home(self):
        q_goal = np.zeros(N_JOINTS)
        self.traj, T, n = plan_joint_path(self.q_cmd, q_goal, hz=self.sim_hz,
                                          speed_deg_per_s=self.speed)
        self.idx = 0; self.pending = q_goal; self.path_safe = True; self.mode = 'preview'
        self.get_logger().info('HOME(관절 0) 미리보기. APPROVE 로 이동하거나 새 좌표 전송.')

    # ---------- 메인 루프 ----------
    def step(self):
        if self._click(self.b_stop, '_sl'):
            if self.bridge:
                self.bridge.emergency_stop()
            self.traj = None; self.mode = 'idle'
            self.get_logger().error('EMERGENCY STOP')
        if self._click(self.b_home, '_hl'):
            self.plan_to_home()
        if self._click(self.b_fix, '_fl'):
            self.fix_mode = not self.fix_mode
            self._show_fix_state()
            self.get_logger().info('FIX MODE = %s' % self.fix_mode)
        if self._click(self.b_appr, '_al'):
            if self.mode == 'await' and self.path_safe:
                if self.bridge:
                    self.traj, _, _ = plan_joint_path(np.array(self.bridge.read_joints()[:N_JOINTS]),
                                                      self.pending, hz=self.sim_hz,
                                                      speed_deg_per_s=self.speed)
                self.idx = 0; self.mode = 'run'
                self.get_logger().info('APPROVE: 실행 시작')
            elif self.mode == 'await' and not self.path_safe:
                self.get_logger().error('충돌 위험 경로라 승인 불가.')

        if self.mode in ('preview', 'run') and self.traj is not None:
            self.q_cmd = self.traj[self.idx]
            if self.mode == 'run' and self.bridge and (self.idx % self.send_every == 0):
                self.bridge.command_joints(list(self.q_cmd) + [self.q5])
            if self.idx < len(self.traj) - 1:
                self.idx += 1
            else:
                if self.mode == 'run' and self.bridge:
                    self.bridge.command_joints(list(self.q_cmd) + [self.q5])
                self.traj = None
                self.mode = 'await' if self.mode == 'preview' else 'idle'
                if self.mode == 'await':
                    self.get_logger().info('미리보기 완료. APPROVE 로 실행하거나 새 좌표 전송.')

        self._show_loads()
        self._apply_sim(self.q_cmd)
        p.stepSimulation()

    # ---------- 유틸 ----------
    def _show_loads(self):
        # 실로봇 연결 시에만, 약 0.25초마다 모터별 전압+부하 표시
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
        self.get_logger().info(txt)
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
        if getattr(node, 'checker', None):
            node.checker.close()
        try:
            p.disconnect()
        except Exception:
            pass
        node.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    main()
