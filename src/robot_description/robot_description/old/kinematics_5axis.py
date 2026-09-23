"""
robot_description / kinematics_5axis.py
---------------------------------------
새 로봇팔(링크 5개 / 회전관절 4개, 고정 말단 없음)용 운동학.

DH 는 사용자가 작성한 표(modified/Craig) 그대로 사용한다:
  i | alpha_{i-1} |  a_{i-1} |   d_i   | theta_i
  1 |     0       | -0.0945  | -0.195  |  θ1        (joint1, 축 Z)
  2 |   -90 deg   |   0      |   0     |  θ2        (joint2, 축 Y)
  3 |     0       |  0.22    |   0     |  θ3        (joint3, 축 Y)
  4 |     0       |  0.23    |   0     |  θ4        (joint4, 축 Y)
  * joint1 프레임을 joint1~joint2 교차점(높이 0.195=0.105+0.09)으로 두어 d1=-0.195.

이 DH 는 URDF(joint2 origin z=0.09)와 전체 자세 오차 ~1e-15 로 일치한다.
검증: fk(q) == Tbase @ fk_dh(q + theta_offset) @ Ctool == fk_urdf(q)
따라서 fk/ik 의 q 는 곧 PyBullet 관절 명령과 같다.
"""

import numpy as np

DEG = np.pi / 180.0

# ── 사용자 작성 modified DH : [alpha_{i-1}, a_{i-1}, d_i, theta_offset_i] ──
#   theta_offset 은 DH 영점 == URDF 영점이 되도록 검증으로 찾은 상수.
DH = np.array([
    [   0.0,     -0.0945, -0.195,   0.0      ],   # joint1 (축 Z)
    [ -90.0*DEG,  0.0,     0.0,   -90.0 * DEG],   # joint2 (축 Y)
    [   0.0,      0.22,    0.0,     0.0      ],   # joint3 (축 Y)
    [   0.0,      0.23,    0.0,     0.0      ],   # joint4 (축 Y)
])
N_JOINTS = 4

# 베이스 프레임 변환 (DH base -> URDF/world base) : 회전 없음 + 평행이동만
Tbase = np.array([[1.0, 0.0, 0.0, 0.0 ],
                  [0.0, 1.0, 0.0, 0.0 ],
                  [0.0, 0.0, 1.0, 0.39],
                  [0.0, 0.0, 0.0, 1.0 ]])

# 툴 프레임 상수 회전 (DH 말단 방향 -> URDF link5 방향)
Ctool = np.array([[0, 0, 1, 0],
                  [1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1]])


def mdh(al, a, th, d):
    ct, st, ca, sa = np.cos(th), np.sin(th), np.cos(al), np.sin(al)
    return np.array([[ct,    -st,    0,   a],
                     [st*ca,  ct*ca, -sa, -sa*d],
                     [st*sa,  ct*sa,  ca,  ca*d],
                     [0,      0,      0,   1]])


def fk(q):
    """q: 길이 4 (구동 관절각, == PyBullet 관절 명령). 반환: 4x4 말단 자세(world)."""
    q = np.asarray(q, float).ravel()
    T = Tbase.copy()
    for k in range(N_JOINTS):
        al, a, d, off = DH[k]
        T = T @ mdh(al, a, q[k] + off, d)
    return T @ Ctool


def ik(target_pos, q0=None, iters=300, tol=1e-8, lam=0.05, step_max=0.4):
    """
    target_pos : 목표 위치 [x,y,z] (world/PyBullet 좌표)
    q0         : 초기 추정값(길이4). 현재 자세를 주면 가까운 해로 수렴
    반환 : (q[4], 최종 오차 노름)
    """
    q = np.zeros(N_JOINTS) if q0 is None else np.array(q0, float).ravel()
    eps = 1e-6
    for _ in range(iters):
        T = fk(q)
        p = T[:3, 3]
        e = target_pos - p
        if np.linalg.norm(e) < tol:
            break
        J = np.zeros((3, N_JOINTS))
        for i in range(N_JOINTS):
            dq = np.zeros(N_JOINTS); dq[i] = eps
            J[:, i] = (fk(q + dq)[:3, 3] - p) / eps
        dq = J.T @ np.linalg.solve(J @ J.T + (lam**2) * np.eye(3), e)
        nd = np.linalg.norm(dq)
        if nd > step_max:
            dq *= step_max / nd
        q = q + dq
    return q, float(np.linalg.norm(e))


def wrap_to_pi(a):
    return (np.asarray(a) + np.pi) % (2*np.pi) - np.pi
