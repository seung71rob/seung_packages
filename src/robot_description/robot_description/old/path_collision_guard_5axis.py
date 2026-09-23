"""
robot_description / path_collision_guard_5axis.py
--------------------------------------------
경로(관절 궤적)가 확정됐을 때, 실행 전에 충돌을 검사한다.
 - 바닥 충돌 : 로봇 vs plane
 - 링크 충돌 : 인접(부모-자식)하지 않은 링크쌍끼리
   (인접 링크는 관절 한계각으로 따로 제한하므로 여기선 제외)

PyBullet 서버(별도 DIRECT 인스턴스)를 하나 띄워 검사 전용으로 쓴다.
메인 GUI 시뮬과 분리해서, 검사 때문에 화면이 흔들리지 않게 한다.
"""

import os
import re
import numpy as np
import pybullet as p
import pybullet_data


class CollisionChecker:
    def __init__(self, urdf_src, pkg_dir, n_joints):
        self.n_joints = n_joints
        self.cid = p.connect(p.DIRECT)                 # 검사 전용(비표시) 서버
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        self.plane = p.loadURDF('plane.urdf', physicsClientId=self.cid)
        self.robot = p.loadURDF(self._prep(urdf_src, pkg_dir), [0, 0, 0],
                                useFixedBase=True,
                                flags=p.URDF_USE_SELF_COLLISION,
                                physicsClientId=self.cid)
        self.n_links = p.getNumJoints(self.robot, physicsClientId=self.cid)
        self.joint_ids = [j for j in range(self.n_links)
                          if p.getJointInfo(self.robot, j, physicsClientId=self.cid)[2] == p.JOINT_REVOLUTE]

        # 비인접(=abs(a-b)>1) 링크쌍만 검사 대상
        for a in range(-1, self.n_links):
            for b in range(a + 1, self.n_links):
                enable = 1 if abs(a - b) > 1 else 0
                p.setCollisionFilterPair(self.robot, self.robot, a, b, enable,
                                         physicsClientId=self.cid)

        # home 에서 이미 닿은 비인접 쌍 = 기준선(무시)
        self._set_q(np.zeros(self.n_joints))
        self.ignore = self._selfpairs()

    def _prep(self, src, pkg_dir):
        with open(src) as f:
            txt = f.read().replace('package://robot_description/', pkg_dir + '/')
        def addc(m):
            inner = re.search(r'<visual>(.*?)</visual>', m.group(0), re.S).group(1)
            return "<visual>" + inner + "</visual>\n<collision>" + inner + "</collision>"
        txt = re.sub(r'<visual>.*?</visual>', addc, txt, flags=re.S)
        out = '/tmp/robot_arm_5axis_collision.urdf'
        with open(out, 'w') as f:
            f.write(txt)
        return out

    def _set_q(self, q):
        for k, jid in enumerate(self.joint_ids[:self.n_joints]):
            p.resetJointState(self.robot, jid, float(q[k]), physicsClientId=self.cid)
        p.performCollisionDetection(physicsClientId=self.cid)

    def _selfpairs(self):
        s = set()
        for cp in p.getContactPoints(self.robot, self.robot, physicsClientId=self.cid):
            if cp[3] != cp[4] and abs(cp[3] - cp[4]) > 1:
                s.add(tuple(sorted((cp[3], cp[4]))))
        return s

    def _floor_hit(self):
        return len(p.getContactPoints(self.robot, self.plane, physicsClientId=self.cid)) > 0

    def check_config(self, q):
        """한 자세 검사 -> (ok, reason). ok=False 면 reason 에 사유."""
        self._set_q(q)
        if self._floor_hit():
            return False, 'floor'
        newself = self._selfpairs() - self.ignore
        if newself:
            return False, 'self:%s' % newself
        return True, ''

    def check_path(self, traj):
        """궤적 전체 검사 -> (ok, idx, reason). ok=False 면 idx 스텝에서 reason."""
        for i in range(len(traj)):
            ok, reason = self.check_config(traj[i])
            if not ok:
                return False, i, reason
        return True, -1, ''

    def close(self):
        p.disconnect(physicsClientId=self.cid)


# ---- 단독 테스트 ----
if __name__ == '__main__':
    import kinematics_5axis as K
    from trajectory_5axis import plan_joint_path
    PKG = os.path.expanduser('~/robot_sim/src/robot_description')
    URDF = os.path.join(PKG, 'urdf', 'robot_arm_5axis.urdf')
    cc = CollisionChecker(URDF, PKG, K.N_JOINTS)
    print('home 무시 자기충돌 쌍:', cc.ignore)
    for tgt in [[-0.25, 0.05, 0.35], [-0.1, 0.0, 0.02], [-0.3, 0.0, 0.6]]:
        q, _ = K.ik(np.array(tgt), q0=np.zeros(4))
        traj, _, _ = plan_joint_path(np.zeros(4), q)
        ok, idx, reason = cc.check_path(traj)
        print(f'목표{tgt}: {"통과" if ok else "충돌"} (idx={idx}, {reason})')
    cc.close()
