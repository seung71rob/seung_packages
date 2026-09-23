"""
robot_description / ik_pybullet_node.py
---------------------------------------
ROS2 노드:
  - PyBullet(GUI)를 띄우고 URDF 로봇팔을 로드
  - 토픽 /target_point (geometry_msgs/Point) 로 목표 좌표를 받으면
  - kinematics.py 의 (DH 기반) 역기구학으로 관절각을 계산하고
  - PyBullet 로봇을 그 자세로 움직여 표시
  - 목표 위치에 빨간 마커를 찍어 도달 여부를 눈으로 검증

실행:
  ros2 run robot_description ik_pybullet_node
좌표 전송(다른 터미널):
  ros2 topic pub --once /target_point geometry_msgs/msg/Point "{x: 0.30, y: 0.0, z: 0.30}"
"""

import os
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point

import pybullet as p
import pybullet_data

from .kinematics import fk, ik, N_JOINTS, wrap_to_pi


class IKPyBulletNode(Node):
    def __init__(self):
        super().__init__('ik_pybullet_node')

        # 패키지 src 경로 (메시/URDF 위치). 환경이 다르면 파라미터로 변경.
        default_pkg = os.path.expanduser('~/robot_sim/src/robot_description')
        self.declare_parameter('pkg_dir', default_pkg)
        self.pkg_dir = self.get_parameter('pkg_dir').value

        urdf_abs = self._prepare_urdf()

        # ---- PyBullet 시작 ----
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadURDF('plane.urdf')
        p.resetDebugVisualizerCamera(cameraDistance=1.0, cameraYaw=50,
                                     cameraPitch=-30, cameraTargetPosition=[0.2, 0, 0.25])

        self.robot = p.loadURDF(urdf_abs, basePosition=[0, 0, 0], useFixedBase=True)
        self.joint_ids = self._revolute_joint_ids()
        self.tip_idx = self._tip_link_index()
        self.get_logger().info('Revolute joints: %s, tip link idx: %d'
                               % (self.joint_ids, self.tip_idx))

        # 목표 마커(빨간 구)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.012, rgbaColor=[1, 0, 0, 0.9])
        self.marker = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                                        basePosition=[10, 10, 10])

        self.q_cmd = np.zeros(N_JOINTS)
        self._apply_joints(self.q_cmd, reset=True)

        self.sub = self.create_subscription(Point, 'target_point', self.on_target, 10)
        self.timer = self.create_timer(1.0 / 240.0, self.step)
        self.get_logger().info('Ready. Publish a target to /target_point.')

    # ---------- URDF 준비 : package:// -> 절대경로 ----------
    def _prepare_urdf(self):
        src = os.path.join(self.pkg_dir, 'urdf', 'robot_arm.urdf')
        with open(src, 'r') as f:
            txt = f.read()
        txt = txt.replace('package://robot_description/', self.pkg_dir + '/')
        out = '/tmp/robot_arm_abs.urdf'
        with open(out, 'w') as f:
            f.write(txt)
        return out

    def _revolute_joint_ids(self):
        ids = []
        for j in range(p.getNumJoints(self.robot)):
            if p.getJointInfo(self.robot, j)[2] == p.JOINT_REVOLUTE:
                ids.append(j)
        return ids[:N_JOINTS]

    def _tip_link_index(self):
        # 마지막(고정) 자식 링크 = 엔드이펙터
        return p.getNumJoints(self.robot) - 1

    # ---------- 목표 수신 → IK ----------
    def on_target(self, msg):
        target = np.array([msg.x, msg.y, msg.z], float)
        q_sol, err = ik(target, q0=self.q_cmd)          # 현재 자세에서 출발(최소 이동 경향)
        q_sol = wrap_to_pi(q_sol)

        # 관절 한계(±3.14) 클램프
        q_sol = np.clip(q_sol, -3.14, 3.14)
        self.q_cmd = q_sol

        reach = fk(q_sol)[:3, 3]
        self.get_logger().info(
            'target=%s | q(deg)=%s | IK 잔차=%.2e | DH-FK 도달=%s'
            % (target.round(4), np.degrees(q_sol).round(2), err, reach.round(4)))
        if err > 1e-3:
            self.get_logger().warn('수렴 부족: 목표가 작업영역 밖이거나 특이점일 수 있음')

        p.resetBasePositionAndOrientation(self.marker, target.tolist(), [0, 0, 0, 1])

    # ---------- 매 스텝 : 관절 명령 + 시뮬레이션 ----------
    def step(self):
        self._apply_joints(self.q_cmd, reset=False)
        p.stepSimulation()

    def _apply_joints(self, q, reset=False):
        for k, jid in enumerate(self.joint_ids):
            if reset:
                p.resetJointState(self.robot, jid, float(q[k]))
            else:
                p.setJointMotorControl2(self.robot, jid, p.POSITION_CONTROL,
                                        targetPosition=float(q[k]), force=50.0)


def main(args=None):
    rclpy.init(args=args)
    node = IKPyBulletNode()
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
