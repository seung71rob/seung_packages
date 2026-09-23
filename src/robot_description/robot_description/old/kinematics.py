"""
robot_description / kinematics.py
---------------------------------
손으로 만든 modified(Craig) DH 파라미터를 그대로 사용하되,
URDF(PyBullet) 좌표계 및 관절 규약과 정확히 일치하도록 상수 변환
(관절각 오프셋 + 베이스 프레임 + 툴 프레임)을 함께 적용한다.

검증 결과(앞 단계):
  fk_urdf(q) == Tbase @ fk_dh(q + theta_offset) @ Ctool   (오차 ~1e-6 m)
따라서 아래 fk(q) 의 q 는 곧 PyBullet 관절 명령과 같다.
즉 ik(...) 의 출력 q 를 그대로 PyBullet 에 보내면 된다.
"""

import numpy as np

DEG = np.pi / 180.0

# ── modified DH 표 : [alpha_{i-1}, a_{i-1}, d_i, theta_offset_i] ──
#   theta_offset 에 검증으로 찾은 상수를 넣어 DH 영점 == URDF 영점이 되게 함.
#   (5번째 행은 고정 말단: joint5_fixed, 0.0945)
DH = np.array([
    [   0.0,     0.0,              -0.171,   180.0 * DEG],   # joint1 (회전)
    [ -90.0*DEG, 0.0,               0.0,     -90.0 * DEG],   # joint2 (회전)
    [   0.0,     0.175,             0.0,      90.0 * DEG],   # joint3 (회전)
    [   0.0,     0.23323854913922,  0.0,      90.0 * DEG],   # joint4 (회전)
    [   0.0,     0.0945,            0.0,       0.0       ],   # 고정 말단(툴)
])
N_JOINTS = 4   # 실제 구동 관절 수

# 베이스 프레임 변환 (DH base -> URDF/world base) : Rz(180deg) + 평행이동
Tbase = np.array([[-1, 0, 0, 0.109],
                  [ 0,-1, 0, 0.0  ],
                  [ 0, 0, 1, 0.342],
                  [ 0, 0, 0, 1.0  ]])

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
    th = [q[0], q[1], q[2], q[3], 0.0]
    T = Tbase.copy()
    for k in range(5):
        al, a, d, off = DH[k]
        T = T @ mdh(al, a, th[k] + off, d)
    return T @ Ctool


def _numeric_jacobian(q, use_axis):
    """유한차분 자코비안. use_axis=True 면 [위치3 + 말단z축정렬2/3] 형태."""
    eps = 1e-6
    f0 = _task(q, use_axis)
    J = np.zeros((f0.size, len(q)))
    for i in range(len(q)):
        dq = np.zeros(len(q)); dq[i] = eps
        J[:, i] = (_task(q + dq, use_axis) - f0) / eps
    return J


def _task(q, use_axis):
    T = fk(q)
    if not use_axis:
        return T[:3, 3]
    # 위치 + 말단 z축 (use_axis=목표 z방향) 정렬 오차용 현재 z축
    return np.concatenate([T[:3, 3], T[:3, 2]])


def ik(target_pos, q0=None, target_z_axis=None,
       iters=300, tol=1e-7, lam=0.05, step_max=0.4):
    """
    target_pos    : 목표 위치 [x,y,z] (world/PyBullet 좌표)
    q0            : 초기 추정값(길이4). 현재 자세를 주면 가까운 해로 수렴
    target_z_axis : 말단 z축을 향하게 할 방향(예: 연직 아래 [0,0,-1]).
                    None 이면 위치만 맞춤(자유도 4 ⇒ 위치 IK 권장).
    반환 : (q[4], 최종 오차 노름)
    """
    q = np.zeros(N_JOINTS) if q0 is None else np.array(q0, float).ravel()
    use_axis = target_z_axis is not None
    if use_axis:
        zd = np.asarray(target_z_axis, float)
        zd = zd / np.linalg.norm(zd)

    for _ in range(iters):
        T = fk(q)
        e = target_pos - T[:3, 3]
        if use_axis:
            eo = np.cross(T[:3, 2], zd)          # z축 정렬 오차
            e = np.concatenate([e, eo])
        if np.linalg.norm(e) < tol:
            break
        J = _numeric_jacobian(q, use_axis)
        m = J.shape[0]
        dq = J.T @ np.linalg.solve(J @ J.T + (lam**2) * np.eye(m), e)
        nd = np.linalg.norm(dq)
        if nd > step_max:
            dq *= step_max / nd
        q = q + dq

    return q, float(np.linalg.norm(e))


def wrap_to_pi(a):
    return (np.asarray(a) + np.pi) % (2*np.pi) - np.pi


def ik_fix(target_pos, q0=None, iters=400, tol=1e-8, lam=0.03, step_max=0.4):
    """
    엔드이펙터를 고정(지면과 평행, 초기 자세의 방향 = 말단 z축이 연직)한 채로
    위치 IK 를 푼다.
    이 팔에서 '지면 평행' 조건은 q2+q3+q4 = 0 (검증 완료, q1 과 무관).
    => 위치3 + 수평1 = 4 제약, 자유도 4 와 정확히 일치.
    반환 : (q[4], 최종 오차 노름)
    """
    q = np.zeros(N_JOINTS) if q0 is None else np.array(q0, float).ravel()
    eps = 1e-6
    for _ in range(iters):
        T = fk(q)
        p = T[:3, 3]
        e_pos = target_pos - p
        e_lvl = -(q[1] + q[2] + q[3])              # 목표: q2+q3+q4 = 0
        e = np.array([e_pos[0], e_pos[1], e_pos[2], e_lvl])
        if np.linalg.norm(e) < tol:
            break
        # position Jacobian (유한차분) 3x4
        Jp = np.zeros((3, 4))
        for i in range(4):
            dq = np.zeros(4); dq[i] = eps
            Jp[:, i] = (fk(q + dq)[:3, 3] - p) / eps
        J = np.vstack([Jp, np.array([[0.0, 1.0, 1.0, 1.0]])])   # level 제약 행
        dq = J.T @ np.linalg.solve(J @ J.T + (lam**2) * np.eye(4), e)
        nd = np.linalg.norm(dq)
        if nd > step_max:
            dq *= step_max / nd
        q = q + dq
    return q, float(np.linalg.norm(e))


# 이전 이름 호환용 별칭
ik_level = ik_fix
