"""
robot_description / ik_execute_node_5axis.py
--------------------------------------------
테스트용 메커니즘:
  좌표 입력(토픽) -> 역기구학 경로 계산 -> PyBullet 에서 미리보기
  -> 사람이 [APPROVE] 버튼 승인 -> 실제 로봇팔 작동
  -> 언제든 [EMERGENCY STOP] 버튼으로 즉시 정지
  -> [GO TO HOME] 버튼으로 안전하게 초기자세 복귀

PyBullet GUI 우측 슬라이더 영역에 버튼이 생깁니다:
  - "APPROVE -> run robot"   : 미리본 경로를 실제 로봇에 전송
  - "EMERGENCY STOP"         : 즉시 정지(현재 위치 고정)
  - "GO TO HOME"             : 초기 자세(0도)로 복귀 경로 생성

실로봇 없이 미리보기만 하려면 use_robot:=false (기본).
실로봇을 붙이면 use_robot:=true 로 실행하세요.

실행:
  ros2 run robot_description ik_execute_node_5axis --ros-args -p use_robot:=false
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


class IKExecuteNode(Node):
    def __init__(self):
        super().__init__('ik_execute_node_5axis')

        self.declare_parameter('pkg_dir', os.path.expanduser('~/robot_sim/src/robot_description'))
        self.declare_parameter('use_robot', False)       # 실로봇 연결 여부
        self.declare_parameter('sim_hz', 240.0)          # PyBullet 스텝
        self.declare_parameter('robot_cmd_hz', 50.0)     # 실모터 전송 주기
        self.declare_parameter('speed_deg_per_s', 60.0)
        self.declare_parameter('dxl_port', '/dev/ttyUSB0')

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
        self.btn_home = p.addUserDebugParameter('GO TO HOME', 1, 0, 0)    # [추가됨] 홈 버튼
        self._approve_last = 0
        self._estop_last = 0
        self._home_last = 0                                               # [추가됨] 홈 버튼 상태

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
        self.mode = 'preview'
        self.pending_goal = q_goal
        p.resetBasePositionAndOrientation(self.marker, target.tolist(), [0, 0, 0, 1])
        self.get_logger().info(
            '미리보기 시작: target=%s q_goal(deg)=%s (%.2fs). '
            '확인 후 APPROVE 버튼을 누르면 실로봇 작동.'
            % (target.round(3), np.degrees(q_goal).round(1), T))

    # ---------- 홈(Home) 포지션 계획 [추가됨] ----------
    def plan_to_home(self):
        # 모든 관절 각도를 0으로 설정
        q_goal = np.zeros(N_JOINTS)
        
        self.traj, T, n = plan_joint_path(self.q_cmd, q_goal,
                                          hz=self.sim_hz, speed_deg_per_s=self.speed)
        self.idx = 0
        self.mode = 'preview'
        self.pending_goal = q_goal
        
        self.get_logger().info(
            'HOME(모든 관절 0도) 미리보기 시작. '
            '확인 후 APPROVE 버튼을 누르면 실로봇이 홈으로 이동합니다.'
        )

    # ---------- 메인 루프 ----------
    def step(self):
        # 1) 비상정지 항상 최우선
        if self._clicked(self.btn_estop, '_estop_last'):
            if self.bridge:
                self.bridge.emergency_stop()
            self.traj = None
            self.mode = 'idle'
            self.get_logger().error('EMERGENCY STOP')

        # [추가됨] 홈 버튼 클릭 확인
        if self._clicked(self.btn_home, '_home_last'):
            self.plan_to_home()

        # 2) 승인 -> 실로봇 실행 준비
        if self._clicked(self.btn_approve, '_approve_last'):
            if self.mode == 'await' and self.traj is not None:
                if self.bridge:
                    q_now = np.array(self.bridge.read_joints())    # 실로봇 현재에서 재계획(점프 방지)
                    self.traj, _, _ = plan_joint_path(q_now, self.pending_goal,
                                                      hz=self.sim_hz, speed_deg_per_s=self.speed)
                self.idx = 0
                self.mode = 'run'
                self.get_logger().info('APPROVE: 실로봇 작동 시작' if self.bridge
                                       else 'APPROVE: (실로봇 미연결) 시뮬만 재생')
            else:
                self.get_logger().info('승인할 미리보기가 없습니다. 좌표를 먼저 보내거나 GO TO HOME을 누르세요.')

        # 3) 경로 재생
        if self.mode in ('preview', 'run') and self.traj is not None:
            self.q_cmd = self.traj[self.idx]
            # 실로봇 전송(run 모드에서만, robot_cmd_hz 로 다운샘플)
            if self.mode == 'run' and self.bridge and (self.idx % self.robot_send_every == 0):
                self.bridge.command_joints(self.q_cmd)
            if self.idx < len(self.traj) - 1:
                self.idx += 1
            else:
                if self.mode == 'run' and self.bridge:
                    self.bridge.command_joints(self.q_cmd)   # 마지막 목표 확실히 전송
                
                # [수정됨] preview가 끝났을 때 traj를 지우지 않고 유지합니다.
                if self.mode == 'preview':
                    self.mode = 'await'
                else:
                    self.mode = 'idle'
                    self.traj = None  # run 모드가 완전히 끝났을 때만 궤적 삭제
                    
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
    node = IKExecuteNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.bridge:
            node.bridge.emergency_stop()
            node.bridge.close()
        try:
            p.disconnect()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
