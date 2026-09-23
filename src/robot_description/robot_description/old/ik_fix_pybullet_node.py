"""
robot_description / ik_fix_pybullet_node.py
---------------------------------------------
앞 노드(ik_pybullet_node)와 동일한 흐름이되,
엔드이펙터를 '지면과 평행'(초기 자세의 방향, 말단 z축이 연직 아래)으로
유지하면서 위치 역기구학을 푼다.  (kinematics.ik_level 사용)

실행:
  ros2 run robot_description ik_fix_pybullet_node
좌표 전송(다른 터미널):
  ros2 topic pub --once /target_point geometry_msgs/msg/Point "{x: 0.30, y: 0.05, z: 0.30}"
"""

import os
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point

import pybullet as p
import pybullet_data

from .kinematics_fix import fk, ik_fix, N_JOINTS, wrap_to_pi


class IKFixPyBulletNode(Node):
    def __init__(self):
        super().__init__('ik_fix_pybullet_node')

        default_pkg = os.path.expanduser('~/robot_sim/src/robot_description')
        self.declare_parameter('pkg_dir', default_pkg)
        self.pkg_dir = self.get_parameter('pkg_dir').value

        urdf_abs = self._prepare_urdf()

        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadURDF('plane.urdf')
        p.resetDebugVisualizerCamera(cameraDistance=1.0, cameraYaw=50,
                                     cameraPitch=-30, cameraTargetPosition=[0.2, 0, 0.25])

        self.robot = p.loadURDF(urdf_abs, basePosition=[0, 0, 0], useFixedBase=True)
        self.joint_ids = self._revolute_joint_ids()
        self.tip_idx = p.getNumJoints(self.robot) - 1
        self.get_logger().info('Revolute joints: %s, tip idx: %d'
                               % (self.joint_ids, self.tip_idx))

        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.012, rgbaColor=[1, 0, 0, 0.9])
        self.marker = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                                        basePosition=[10, 10, 10])
        self.axis_line = None   # 말단 z축 표시용 디버그 라인

        self.q_cmd = np.zeros(N_JOINTS)
        self._apply_joints(self.q_cmd, reset=True)

        self.sub = self.create_subscription(Point, 'target_point', self.on_target, 10)
        self.timer = self.create_timer(1.0 / 240.0, self.step)
        self.get_logger().info('Ready (FIX mode). 엔드이펙터를 고정(지면과 평행)하고 IK.')

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

    def on_target(self, msg):
        target = np.array([msg.x, msg.y, msg.z], float)
        q_sol, err = ik_fix(target, q0=self.q_cmd)     # 지면 평행 제약 IK
        q_sol = np.clip(wrap_to_pi(q_sol), -3.14, 3.14)
        self.q_cmd = q_sol

        T = fk(q_sol)
        reach = T[:3, 3]
        z_axis = T[:3, 2]                                  # 수평이면 ~[0,0,-1]
        lvl_sum = np.degrees(q_sol[1] + q_sol[2] + q_sol[3])
        self.get_logger().info(
            'target=%s | q(deg)=%s | 잔차=%.2e | 도달=%s | 말단z축=%s | q2+q3+q4=%.2fdeg'
            % (target.round(4), np.degrees(q_sol).round(2), err,
               reach.round(4), z_axis.round(3), lvl_sum))
        if err > 1e-3:
            self.get_logger().warn('수렴 부족: 평행 제약 하에 도달 불가한 목표일 수 있음')

        p.resetBasePositionAndOrientation(self.marker, target.tolist(), [0, 0, 0, 1])
        # 말단 z축(접근축) 시각화 : 도달점에서 z축 방향으로 짧은 선
        end = (reach + 0.08 * z_axis).tolist()
        kw = dict(lineColorRGB=[0, 0.4, 1], lineWidth=3)
        if self.axis_line is None:
            self.axis_line = p.addUserDebugLine(reach.tolist(), end, **kw)
        else:
            self.axis_line = p.addUserDebugLine(reach.tolist(), end,
                                                replaceItemUniqueId=self.axis_line, **kw)

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
    node = IKFixPyBulletNode()
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
