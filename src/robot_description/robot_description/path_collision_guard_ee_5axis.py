"""
robot_description / path_collision_guard_ee_5axis.py
----------------------------------------------------
엔드이펙터 포함 로봇의 경로 충돌 검사(별도 DIRECT 서버).
 - 바닥 충돌(로봇 vs plane)
 - 비인접(부모-자식 아님) 링크끼리 충돌  (인접 링크는 관절 한계각으로 제한)
 - home 에서 이미 닿은 쌍은 무시(형상 근사 오탐 방지)
위치 IK 는 q1~q4, joint5(그리퍼)는 0 으로 두고 검사한다.
"""
import os, re
import numpy as np
import pybullet as p
import pybullet_data


class CollisionChecker:
    def __init__(self, urdf_src, pkg_dir, n_pos_joints=4, mesh_dir=None):
        self.mesh_dir = mesh_dir
        self.n_pos = n_pos_joints
        self.cid = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        self.plane = p.loadURDF('plane.urdf', physicsClientId=self.cid)
        self.robot = p.loadURDF(self._prep(urdf_src, pkg_dir), [0, 0, 0], useFixedBase=True,
                                flags=p.URDF_USE_SELF_COLLISION, physicsClientId=self.cid)
        self.n_links = p.getNumJoints(self.robot, physicsClientId=self.cid)
        self.rev = [j for j in range(self.n_links)
                    if p.getJointInfo(self.robot, j, physicsClientId=self.cid)[2] == p.JOINT_REVOLUTE]
        for a in range(-1, self.n_links):
            for b in range(a + 1, self.n_links):
                p.setCollisionFilterPair(self.robot, self.robot, a, b,
                                         1 if abs(a - b) > 1 else 0, physicsClientId=self.cid)
        self._set_q(np.zeros(self.n_pos))
        self.ignore = self._selfpairs()

    def _prep(self, src, pkg_dir):
        with open(src) as f:
            txt = f.read()
        if self.mesh_dir:
            txt = txt.replace('robot_meshes/', self.mesh_dir.rstrip('/') + '/')
        txt = txt.replace('package://robot_description/', pkg_dir + '/')
        def addc(m):
            inner = re.search(r'<visual>(.*?)</visual>', m.group(0), re.S).group(1)
            return "<visual>" + inner + "</visual>\n<collision>" + inner + "</collision>"
        txt = re.sub(r'<visual>.*?</visual>', addc, txt, flags=re.S)
        out = '/tmp/ee_collision.urdf'
        with open(out, 'w') as f:
            f.write(txt)
        return out

    def _set_q(self, q):
        # q1~q4 세팅, 나머지 회전관절(q5=그리퍼)은 0
        full = list(q) + [0.0] * (len(self.rev) - self.n_pos)
        for k, jid in enumerate(self.rev):
            p.resetJointState(self.robot, jid, float(full[k]), physicsClientId=self.cid)
        p.performCollisionDetection(physicsClientId=self.cid)

    def _selfpairs(self):
        s = set()
        for cp in p.getContactPoints(self.robot, self.robot, physicsClientId=self.cid):
            if cp[3] != cp[4] and abs(cp[3] - cp[4]) > 1:
                s.add(tuple(sorted((cp[3], cp[4]))))
        return s

    def _floor(self):
        return len(p.getContactPoints(self.robot, self.plane, physicsClientId=self.cid)) > 0

    def check_config(self, q):
        self._set_q(q)
        if self._floor():
            return False, 'floor'
        new = self._selfpairs() - self.ignore
        if new:
            return False, 'self:%s' % new
        return True, ''

    def check_path(self, traj):
        for i in range(len(traj)):
            ok, reason = self.check_config(traj[i])
            if not ok:
                return False, i, reason
        return True, -1, ''

    def close(self):
        p.disconnect(physicsClientId=self.cid)
