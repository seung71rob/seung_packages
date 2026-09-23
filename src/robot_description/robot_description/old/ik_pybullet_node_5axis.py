"""
robot_description / ik_pybullet_node_5axis.py
---------------------------------------------
새 5축(4-DOF) 로봇팔용 ROS2 노드.
좌표 토픽 /target_point (geometry_msgs/Point) 를 받으면
kinematics_5axis 의 역기구학으로 관절각을 계산하고, PyBullet 로봇을
초기 자세에서 그 위치로 이동시킨다.

실행:
  ros2 run robot_description ik_pybullet_node_5axis
좌표 전송(다른 터미널):
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


class IK5AxisNode(Node):
    def __init__(self):
        super().__init__('ik_pybullet_node_5axis')

        default_pkg = os.path.expanduser('~/robot_sim/src/robot_description')
        self.declare_parameter('pkg_dir', default_pkg)
        self.pkg_dir = self.get_parameter('pkg_dir').value

        urdf_abs = self._prepare_urdf()

        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadURDF('plane.urdf')
        p.resetDebugVisualizerCamera(cameraDistance=1.1, cameraYaw=50,
                                     cameraPitch=-30, cameraTargetPosition=[0.1, 0, 0.35])

        self.robot = p.loadURDF(urdf_abs, basePosition=[0, 0, 0], useFixedBase=True)
        self.joint_ids = self._revolute_joint_ids()
        self.tip_idx = p.getNumJoints(self.robot) - 1
        self.get_logger().info('Revolute joints: %s, tip idx: %d'
                               % (self.joint_ids, self.tip_idx))

        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.012, rgbaColor=[1, 0, 0, 0.9])
        self.marker = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                                        basePosition=[10, 10, 10])

        self.q_cmd = np.zeros(N_JOINTS)
        self._apply_joints(self.q_cmd, reset=True)

        self.sub = self.create_subscription(Point, 'target_point', self.on_target, 10)
        self.timer = self.create_timer(1.0 / 240.0, self.step)
        self.get_logger().info('Ready (5-axis). Publish a target to /target_point.')

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
        q_sol, err = ik(target, q0=self.q_cmd)         # 현재 자세에서 출발
        q_sol = np.clip(wrap_to_pi(q_sol), -3.14, 3.14)
        self.q_cmd = q_sol

        reach = fk(q_sol)[:3, 3]
        self.get_logger().info(
            'target=%s | q(deg)=%s | IK 잔차=%.2e | 도달=%s'
            % (target.round(4), np.degrees(q_sol).round(2), err, reach.round(4)))
        if err > 1e-3:
            self.get_logger().warn('수렴 부족: 목표가 작업영역 밖이거나 특이점일 수 있음')

        p.resetBasePositionAndOrientation(self.marker, target.tolist(), [0, 0, 0, 1])

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
    node = IK5AxisNode()
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
