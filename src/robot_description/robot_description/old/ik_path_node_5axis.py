"""
robot_description / ik_path_node_5axis.py
-----------------------------------------
좌표 토픽 /target_point (geometry_msgs/Point) 를 받으면
 1) IK 로 목표 관절값을 구하고
 2) 초기(현재)위치 -> 목표 사이에 경유점을 자동 생성해 S-커브로 부드럽게 보간
 3) PyBullet 에서 240 Hz 로 그 궤적을 재생

2단계(수직 하강)는 목표점을 두 번 보내면 그대로 재사용된다.
(예: 먼저 목표1 위쪽 점 -> 그다음 z 만 낮춘 목표2)

실행:
  ros2 run robot_description ik_path_node_5axis
좌표 전송(다른 터미널):
  ros2 topic pub --once /target_point geometry_msgs/msg/Point "{x: 0.25, y: 0.05, z: 0.35}"

명령 갱신 주기(기본 240Hz)는 파라미터로 조정 가능:
  ros2 run robot_description ik_path_node_5axis --ros-args -p command_hz:=240.0
  (실기 브리지로 옮길 땐 이 궤적을 20~60Hz 로 다시 샘플링해 모터로 전송)
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


class IKPathNode(Node):
    def __init__(self):
        super().__init__('ik_path_node_5axis')

        default_pkg = os.path.expanduser('~/robot_sim/src/robot_description')
        self.declare_parameter('pkg_dir', default_pkg)
        self.declare_parameter('command_hz', 240.0)     # PyBullet 검증용 기본값
        self.declare_parameter('speed_deg_per_s', 60.0)  # 이동시간 = 최대이동각/이 값
        self.pkg_dir = self.get_parameter('pkg_dir').value
        self.hz = float(self.get_parameter('command_hz').value)
        self.speed = float(self.get_parameter('speed_deg_per_s').value)

        urdf_abs = self._prepare_urdf()

        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadURDF('plane.urdf')
        p.resetDebugVisualizerCamera(cameraDistance=1.1, cameraYaw=50,
                                     cameraPitch=-30, cameraTargetPosition=[0.1, 0, 0.35])

        self.robot = p.loadURDF(urdf_abs, basePosition=[0, 0, 0], useFixedBase=True)
        self.joint_ids = self._revolute_joint_ids()

        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.012, rgbaColor=[1, 0, 0, 0.9])
        self.marker = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                                        basePosition=[10, 10, 10])

        self.q_cmd = np.zeros(N_JOINTS)      # 현재 명령 자세
        self.traj = None                     # 재생 중인 궤적
        self.idx = 0
        self._apply_joints(self.q_cmd, reset=True)

        self.sub = self.create_subscription(Point, 'target_point', self.on_target, 10)
        self.timer = self.create_timer(1.0 / self.hz, self.step)
        self.get_logger().info('Ready (path, %.0f Hz). Publish target to /target_point.' % self.hz)

    def _prepare_urdf(self):
        src = os.path.join(self.pkg_dir, 'urdf', 'robot_arm_5axis.urdf')
        with open(src, 'r') as f:
            txt = f.read()
        txt = txt.replace('package://robot_description/', self.pkg_dir + '/')
        out = '/tmp/robot_arm_5axis_abs.urdf'
        with open(out, 'w') as f:
            f.write(txt)
        return out

    def _revolute_joint_ids(self):
        ids = []
        for j in range(p.getNumJoints(self.robot)):
            if p.getJointInfo(self.robot, j)[2] == p.JOINT_REVOLUTE:
                ids.append(j)
        return ids[:N_JOINTS]

    def on_target(self, msg):
        target = np.array([msg.x, msg.y, msg.z], float)
        q_goal, err = ik(target, q0=self.q_cmd)          # 현재 자세에서 출발
        q_goal = np.clip(wrap_to_pi(q_goal), -3.14, 3.14)

        if err > 1e-3:
            self.get_logger().warn('IK 잔차 큼(%.2e): 작업영역 밖/특이점 가능. 그래도 진행.' % err)

        # 현재 자세 -> 목표 로 부드러운 관절공간 경로 생성
        self.traj, T, n = plan_joint_path(self.q_cmd, q_goal, hz=self.hz,
                                          speed_deg_per_s=self.speed)
        self.idx = 0
        self.get_logger().info(
            'target=%s | q_goal(deg)=%s | 이동시간 %.2fs (%d스텝) | 잔차=%.2e'
            % (target.round(4), np.degrees(q_goal).round(2), T, n, err))
        p.resetBasePositionAndOrientation(self.marker, target.tolist(), [0, 0, 0, 1])

    def step(self):
        # 궤적 재생: 매 스텝 다음 경유 자세로
        if self.traj is not None:
            self.q_cmd = self.traj[self.idx]
            if self.idx < len(self.traj) - 1:
                self.idx += 1
            else:
                self.traj = None            # 도착: 목표에서 정지 유지
        self._apply_joints(self.q_cmd, reset=False)
        p.stepSimulation()

    def _apply_joints(self, q, reset=False):
        for k, jid in enumerate(self.joint_ids):
            if reset:
                p.resetJointState(self.robot, jid, float(q[k]))
            else:
                p.setJointMotorControl2(self.robot, jid, p.POSITION_CONTROL,
                                        targetPosition=float(q[k]),
                                        maxVelocity=3.0, force=50.0)


def main(args=None):
    rclpy.init(args=args)
    node = IKPathNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            p.disconnect()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
