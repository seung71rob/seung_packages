"""
robot_description / motion_execute_node_5axis.py
--------------------------------------------
테스트용 메커니즘:
  좌표 입력(토픽) -> 역기구학 경로 계산 -> PyBullet 에서 미리보기
  -> 사람이 [APPROVE] 버튼 승인 -> 실제 로봇팔 작동
  -> 언제든 [EMERGENCY STOP] 버튼으로 즉시 정지

실행:
  ros2 run robot_description motion_execute_node_5axis --ros-args -p use_robot:=false
좌표 전송:
  ros2 topic pub --once /target_point geometry_msgs/msg/Point "{x: 0.25, y: 0.05, z: 0.35}"
"""

import os
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point

import pybullet as p
import pybullet_data

from .kinematics_5axis import fk, ik, N_JOINTS, wrap_to_pi
from .trajectory_5axis import plan_joint_path
from .path_collision_guard_5axis import CollisionChecker


class MotionExecuteNode(Node):
    def __init__(self):
        super().__init__('motion_execute_node_5axis')

        self.declare_parameter('pkg_dir', os.path.expanduser('~/robot_sim/src/robot_description'))
        self.declare_parameter('use_robot', False)       # 실로봇 연결 여부
        self.declare_parameter('sim_hz', 240.0)          # PyBullet 스텝
        self.declare_parameter('robot_cmd_hz', 50.0)     # 실모터 전송 주기
        self.declare_parameter('speed_deg_per_s', 60.0)
        self.declare_parameter('dxl_port', '/dev/ttyUSB0')
        self.declare_parameter('check_collision', True)   # 경로 충돌 검사(바닥+링크)

        self.pkg_dir = self.get_parameter('pkg_dir').value
        self.use_robot = bool(self.get_parameter('use_robot').value)
        self.sim_hz = float(self.get_parameter('sim_hz').value)
        self.robot_cmd_hz = float(self.get_parameter('robot_cmd_hz').value)
        self.speed = float(self.get_parameter('speed_deg_per_s').value)

        # ---- PyBullet ----
        urdf_abs = self._prepare_urdf()
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadURDF('plane.urdf')
        p.resetDebugVisualizerCamera(1.1, 50, -30, [0.1, 0, 0.35])
        self.robot = p.loadURDF(urdf_abs, [0, 0, 0], useFixedBase=True)
        self.joint_ids = [j for j in range(p.getNumJoints(self.robot))
                          if p.getJointInfo(self.robot, j)[2] == p.JOINT_REVOLUTE][:N_JOINTS]
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.012, rgbaColor=[1, 0, 0, 0.9])
        self.marker = p.createMultiBody(0, baseVisualShapeIndex=vis, basePosition=[10, 10, 10])
        self.btn_approve = p.addUserDebugParameter('APPROVE -> run robot', 1, 0, 0)
        self.btn_estop = p.addUserDebugParameter('EMERGENCY STOP', 1, 0, 0)
        self._approve_last = 0
        self._estop_last = 0

        # ---- 실로봇 브리지 (선택) ----
        self.bridge = None
        if self.use_robot:
            from .dxl_bridge_5axis import DxlBridge
            self.bridge = DxlBridge(port=self.get_parameter('dxl_port').value)
            self.bridge.connect()
            self.bridge.capture_home()      # 로봇을 home 에 둔 채 시작해야 함!
            self.get_logger().warn('실로봇 모드: 로봇이 home(모든 관절 0)에 있는지 확인하세요.')

        # ---- 상태 ----
        self.q_cmd = np.zeros(N_JOINTS)
        self.traj = None
        self.idx = 0
        self.mode = 'idle'          # idle | preview | await | run
        self.robot_send_every = max(int(round(self.sim_hz / self.robot_cmd_hz)), 1)
        self._apply_sim(self.q_cmd, reset=True)

        # 경로 충돌 검사기 (별도 DIRECT 서버). 바닥+비인접 링크 충돌 검사.
        self.checker = None
        self.path_safe = False
        if bool(self.get_parameter('check_collision').value):
            self.checker = CollisionChecker(
                os.path.join(self.pkg_dir, 'urdf', 'robot_arm_5axis.urdf'),
                self.pkg_dir, N_JOINTS)
            self.get_logger().info('경로 충돌 검사 ON (바닥 + 비인접 링크)')

        self.sub = self.create_subscription(Point, 'target_point', self.on_target, 10)
        self.timer = self.create_timer(1.0 / self.sim_hz, self.step)
        self.get_logger().info('Ready (execute). use_robot=%s. 좌표를 /target_point 로 보내세요.'
                               % self.use_robot)

    # ---------- URDF ----------
    def _prepare_urdf(self):
        src = os.path.join(self.pkg_dir, 'urdf', 'robot_arm_5axis.urdf')
        with open(src) as f:
            txt = f.read().replace('package://robot_description/', self.pkg_dir + '/')
        out = '/tmp/robot_arm_5axis_abs.urdf'
        with open(out, 'w') as f:
            f.write(txt)
        return out

    # ---------- 좌표 수신 -> 경로 계산 -> 미리보기 ----------
    def on_target(self, msg):
        target = np.array([msg.x, msg.y, msg.z], float)
        q_goal, err = ik(target, q0=self.q_cmd)
        q_goal = np.clip(wrap_to_pi(q_goal), -3.14, 3.14)
        if err > 1e-3:
            self.get_logger().warn('IK 잔차 큼(%.2e): 작업영역 밖/특이점 가능.' % err)

        self.traj, T, n = plan_joint_path(self.q_cmd, q_goal,
                                          hz=self.sim_hz, speed_deg_per_s=self.speed)
        self.idx = 0

        # 확정된 경로 충돌 검사 -> 위험하면 승인 차단
        self.path_safe = True
        if self.checker is not None:
            ok, bad_idx, reason = self.checker.check_path(self.traj)
            if not ok:
                self.path_safe = False
                self.get_logger().error(
                    '경로 충돌 위험: 스텝 %d/%d, 사유=%s -> APPROVE 차단됨'
                    % (bad_idx, len(self.traj), reason))

        self.mode = 'preview'
        self.pending_goal = q_goal
        p.resetBasePositionAndOrientation(self.marker, target.tolist(), [0, 0, 0, 1])
        self.get_logger().info(
            '미리보기 시작: target=%s q_goal(deg)=%s (%.2fs). '
            '확인 후 APPROVE 버튼을 누르면 실로봇 작동.'
            % (target.round(3), np.degrees(q_goal).round(1), T))

    # ---------- 메인 루프 ----------
    def step(self):
        # 1) 비상정지 항상 최우선
        if self._clicked(self.btn_estop, '_estop_last'):
            if self.bridge:
                self.bridge.emergency_stop()
            self.traj = None
            self.mode = 'idle'
            self.get_logger().error('EMERGENCY STOP')

        # 2) 승인 -> 실로봇 실행 준비
        if self._clicked(self.btn_approve, '_approve_last'):
            if self.mode == 'await' and self.traj is not None and self.path_safe:
                if self.bridge:
                    q_now = np.array(self.bridge.read_joints())
                    self.traj, _, _ = plan_joint_path(q_now, self.pending_goal,
                                                      hz=self.sim_hz, speed_deg_per_s=self.speed)
                self.idx = 0
                self.mode = 'run'
                self.get_logger().info('APPROVE: 실로봇 작동 시작' if self.bridge
                                       else 'APPROVE: (실로봇 미연결) 시뮬만 재생')
            elif self.mode == 'await' and not self.path_safe:
                self.get_logger().error('충돌 위험 경로라 승인할 수 없습니다. 다른 좌표를 보내세요.')
            else:
                self.get_logger().info('승인할 미리보기가 없습니다. 좌표를 먼저 보내세요.')

        # 3) 경로 재생
        if self.mode in ('preview', 'run') and self.traj is not None:
            self.q_cmd = self.traj[self.idx]
            if self.mode == 'run' and self.bridge and (self.idx % self.robot_send_every == 0):
                self.bridge.command_joints(self.q_cmd)
            if self.idx < len(self.traj) - 1:
                self.idx += 1
            else:
                if self.mode == 'run' and self.bridge:
                    self.bridge.command_joints(self.q_cmd)
                self.traj = None
                self.mode = 'await' if self.mode == 'preview' else 'idle'
                if self.mode == 'await':
                    self.get_logger().info('미리보기 완료. APPROVE 로 실행하거나 새 좌표 전송.')

        self._apply_sim(self.q_cmd, reset=False)
        p.stepSimulation()

    # ---------- 유틸 ----------
    def _clicked(self, btn, last_attr):
        v = p.readUserDebugParameter(btn)
        if v > getattr(self, last_attr):
            setattr(self, last_attr, v)
            return True
        return False

    def _apply_sim(self, q, reset=False):
        for k, jid in enumerate(self.joint_ids):
            if reset:
                p.resetJointState(self.robot, jid, float(q[k]))
            else:
                p.setJointMotorControl2(self.robot, jid, p.POSITION_CONTROL,
                                        targetPosition=float(q[k]), maxVelocity=3.0, force=50.0)


def main(args=None):
    rclpy.init(args=args)
    node = MotionExecuteNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.bridge:
            node.bridge.emergency_stop()
            node.bridge.close()
        if getattr(node, 'checker', None):
            node.checker.close()
        try:
            p.disconnect()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
