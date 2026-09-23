"""
joint_limit_check_5axis.py  (단독 실행, 진단 출력 버전)
------------------------------------------------------
슬라이더로 관절 조절 -> 링크 접촉 시 빨간 CONTACT.
끝단(link4-link5)이 안 잡히는 문제를 확실히 잡기 위해:
  - URDF_USE_SELF_COLLISION_INCLUDE_PARENT 로 부모-자식 충돌까지 포함
  - home 무시쌍에서 '끝단 쌍'은 제외(항상 감시)
  - 링크 인덱스/이름, home 무시쌍, 실시간 접촉쌍을 터미널에 출력
"""

import os
import re
import time
import numpy as np
import pybullet as p
import pybullet_data

PKG = os.path.expanduser('~/robot_sim/src/robot_description')
URDF_SRC = os.path.join(PKG, 'urdf', 'robot_arm_5axis.urdf')


def prepare_urdf():
    with open(URDF_SRC) as f:
        txt = f.read().replace('package://robot_description/', PKG + '/')

    def add_collision(m):
        inner = re.search(r'<visual>(.*?)</visual>', m.group(0), re.S).group(1)
        return "<visual>" + inner + "</visual>\n    <collision>" + inner + "</collision>"

    txt = re.sub(r'<visual>.*?</visual>', add_collision, txt, flags=re.S)
    out = '/tmp/robot_arm_5axis_collision.urdf'
    with open(out, 'w') as f:
        f.write(txt)
    return out


def raw_pairs(robot):
    p.performCollisionDetection()
    return set(tuple(sorted((cp[3], cp[4]))) for cp in p.getContactPoints(robot, robot)
               if cp[3] != cp[4])


def main():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)
    p.resetDebugVisualizerCamera(1.1, 50, -25, [0.05, 0, 0.35])

    # 부모-자식(인접) 충돌까지 포함하는 플래그
    robot = p.loadURDF(prepare_urdf(), [0, 0, 0], useFixedBase=True,
                       flags=p.URDF_USE_SELF_COLLISION |
                             p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)

    n_links = p.getNumJoints(robot)
    # 모든 링크쌍 충돌검사 ON (부모-자식 포함)
    for a in range(-1, n_links):
        for b in range(a + 1, n_links):
            p.setCollisionFilterPair(robot, robot, a, b, 1)

    # 링크 인덱스/이름 출력
    print('=== 링크 인덱스 ===')
    print('  -1 : base(link1)')
    for j in range(n_links):
        info = p.getJointInfo(robot, j)
        print(f'  {j:2d} : link={info[12].decode()}  joint={info[1].decode()}  type={info[2]}')

    joint_ids = [j for j in range(n_links) if p.getJointInfo(robot, j)[2] == p.JOINT_REVOLUTE]
    names = [p.getJointInfo(robot, j)[1].decode() for j in joint_ids]

    # 끝단 쌍 = 마지막 두 링크 인덱스
    tip_pair = tuple(sorted((n_links - 1, n_links - 2)))
    print('끝단 쌍(항상 감시):', tip_pair)

    for jid in joint_ids:
        p.resetJointState(robot, jid, 0.0)
    ignore = raw_pairs(robot)
    ignore.discard(tip_pair)                 # 끝단 쌍은 무시하지 않음
    print('home 접촉(무시) 쌍:', ignore if ignore else '없음')

    sliders = [p.addUserDebugParameter(f'{names[k]} (deg)', -180, 180, 0)
               for k in range(len(joint_ids))]

    txt_id = None
    prev = None
    while True:
        for k, jid in enumerate(joint_ids):
            p.resetJointState(robot, jid, np.radians(p.readUserDebugParameter(sliders[k])))

        cur = raw_pairs(robot)
        new_contact = cur - ignore

        if cur != prev:                       # 접촉쌍이 바뀔 때마다 raw 출력
            print('raw 접촉쌍:', cur if cur else '없음')
            prev = cur

        if txt_id is not None:
            p.removeUserDebugItem(txt_id); txt_id = None
        if new_contact:
            txt_id = p.addUserDebugText('CONTACT', [0, 0, 0.95],
                                        textColorRGB=[1, 0, 0], textSize=1.5)

        time.sleep(1 / 60.0)


if __name__ == '__main__':
    main()
