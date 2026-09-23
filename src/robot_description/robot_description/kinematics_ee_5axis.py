#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kinematics_ee_5axis.py  (v2)

5축 로봇팔 운동학 (centric / suction 공용).
좌표축 설정은 사용자 수기 도면 <좌표축 설정> 기준, modified(Craig) DH.

v2 정정사항
-----------
1) 링크 길이:  joint2->joint3  0.22 -> 0.195
               joint3->joint4  0.24 -> 0.165        (URDF 실측값)
2) DH row1  d1 : -0.195 -> +0.195
3) DH row6  : a5=-0.2115 (x 방향) -> d6=+0.2115 (z 방향)
              TCP는 joint5 회전축 위에 있으므로 z 방향 오프셋이 맞다.
              a 칸에 넣으면 q5를 돌릴 때 TCP 위치가 움직여버린다(URDF는 0).
4) theta 오프셋 명시 : th2 = q2 - 90deg,  th4 = q4 + 90deg

검증: URDF 체인과 랜덤 5000자세 4x4 전체 비교 -> 최대오차 ~1e-16
     (이 파일을 직접 실행하면 self-test 가 돈다:  python3 kinematics_ee_5axis.py)
"""

import numpy as np

# ---------------------------------------------------------------- 치수 (m)
BASE_X = -0.0945   # link1 -> joint1  x 오프셋
D1     =  0.195    # 바닥 -> 어깨(joint2)  = 0.105 + 0.09
L2     =  0.195    # joint2 -> joint3      (URDF joint3 origin z)
L3     =  0.165    # joint3 -> joint4      (URDF joint4 origin z)
_TCP_Z =  0.2115   # joint4/5 축 -> TCP    = 0.087 + 0.1245

N_JOINTS       = 4   # 위치 IK 자유도 (q1~q4).  q5는 TCP 위치에 영향 없음
N_MOTOR_JOINTS = 5   # 실제 모터 관절 수

LEVEL_SUM = np.pi    # fix 모드: q2+q3+q4 = pi 이면 TCP z축이 [0,0,-1] (지면 향함)

SHOULDER = np.array([BASE_X, 0.0, D1])   # joint2 원점 (q1과 무관하게 고정)
MAX_REACH = L2 + L3 + _TCP_Z             # 0.5715 m
MIN_REACH = 0.0                          # 최장변 < 나머지 합 이므로 0

# ------------------------------------------------- 관절 한계 (아직 미확정)
# URDF <limit> / 여기 / 경로검사 3곳이 항상 같아야 한다.
# 값이 정해지면 아래 두 배열만 고치고 URDF <limit>도 같은 값으로 맞출 것.
Q_MIN = np.array([-np.pi, -np.pi, -np.pi, -np.pi, -np.pi])
Q_MAX = np.array([ np.pi,  np.pi,  np.pi,  np.pi,  np.pi])


# ---------------------------------------------------------------- DH 테이블
# (alpha_{i-1},  a_{i-1},  d_i,  theta offset)
# T_{i-1,i} = Rx(alpha) @ Tx(a) @ Rz(theta) @ Tz(d)
_D90 = np.pi / 2.0
DH_TABLE = (
    (0.0,   BASE_X,  D1,      0.0),    # row1  th1 = q1
    (-_D90, 0.0,     0.0,   -_D90),    # row2  th2 = q2 - 90deg
    (0.0,   L2,      0.0,     0.0),    # row3  th3 = q3
    (0.0,   L3,      0.0,   +_D90),    # row4  th4 = q4 + 90deg
    (+_D90, 0.0,     0.0,     0.0),    # row5  th5 = q5
    (0.0,   0.0,   _TCP_Z,    0.0),    # row6  고정 TCP
)


def wrap_to_pi(a):
    """각도를 (-pi, pi] 로 정규화. 스칼라/배열 모두 가능."""
    return (np.asarray(a, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def _rx(t):
    c, s = np.cos(t), np.sin(t)
    T = np.eye(4); T[1, 1] = c; T[1, 2] = -s; T[2, 1] = s; T[2, 2] = c
    return T


def _rz(t):
    c, s = np.cos(t), np.sin(t)
    T = np.eye(4); T[0, 0] = c; T[0, 1] = -s; T[1, 0] = s; T[1, 1] = c
    return T


def _tx(a):
    T = np.eye(4); T[0, 3] = a
    return T


def _tz(d):
    T = np.eye(4); T[2, 3] = d
    return T


def _pad5(q):
    """q 를 길이 5로 맞춘다 (q5 미지정이면 0)."""
    q = np.asarray(q, dtype=float).ravel()
    if q.size == 4:
        return np.concatenate([q, [0.0]])
    if q.size == 5:
        return q.copy()
    raise ValueError("q must have 4 or 5 elements, got %d" % q.size)


# ---------------------------------------------------------------- 정기구학
def fk(q):
    """
    TCP pose 4x4 반환.  q: (4,) 또는 (5,)
    DH 테이블을 그대로 곱한다 (수기 도면과 1:1 대응).
    """
    qq = _pad5(q)
    th = [qq[0], qq[1], qq[2], qq[3], qq[4], 0.0]
    T = np.eye(4)
    for i, (alpha, a, d, off) in enumerate(DH_TABLE):
        T = T @ _rx(alpha) @ _tx(a) @ _rz(th[i] + off) @ _tz(d)
    return T


def fk_pos(q):
    """
    TCP 위치만 (3,) 반환. 폐형식이라 fk()보다 훨씬 빠르다 (IK 내부 루프용).
      u2 = q2,  u3 = q2+q3,  u4 = q2+q3+q4
      x' = L2 sin u2 + L3 sin u3 + TCP sin u4
      z' = L2 cos u2 + L3 cos u3 + TCP cos u4
    """
    qq = _pad5(q)
    q1, q2, q3, q4 = qq[0], qq[1], qq[2], qq[3]
    u2 = q2
    u3 = q2 + q3
    u4 = q2 + q3 + q4
    xp = L2 * np.sin(u2) + L3 * np.sin(u3) + _TCP_Z * np.sin(u4)
    zp = L2 * np.cos(u2) + L3 * np.cos(u3) + _TCP_Z * np.cos(u4)
    return np.array([BASE_X + np.cos(q1) * xp,
                     np.sin(q1) * xp,
                     D1 + zp])


def jacobian_pos(q):
    """TCP 위치에 대한 3x4 야코비안 (q1~q4). DLS IK와 힘추정(J^T)에 사용."""
    qq = _pad5(q)
    q1, q2, q3, q4 = qq[0], qq[1], qq[2], qq[3]
    c1, s1 = np.cos(q1), np.sin(q1)
    u2 = q2
    u3 = q2 + q3
    u4 = q2 + q3 + q4

    A2, B2 = L2 * np.sin(u2), L2 * np.cos(u2)
    A3, B3 = L3 * np.sin(u3), L3 * np.cos(u3)
    A4, B4 = _TCP_Z * np.sin(u4), _TCP_Z * np.cos(u4)

    xp = A2 + A3 + A4
    zp = B2 + B3 + B4

    dxp = np.array([zp,        B3 + B4,   B4])          # d x' / d(q2,q3,q4)
    dzp = np.array([-xp,      -(A3 + A4), -A4])         # d z' / d(q2,q3,q4)

    J = np.zeros((3, 4))
    J[:, 0] = [-s1 * xp, c1 * xp, 0.0]
    J[0, 1:] = c1 * dxp
    J[1, 1:] = s1 * dxp
    J[2, 1:] = dzp
    return J


# ------------------------------------------------------------ 한계각 체크
def in_limits(q, margin=0.0):
    """q(4 또는 5)가 Q_MIN/Q_MAX 안에 있는지."""
    qq = _pad5(q)
    return bool(np.all(qq >= Q_MIN + margin) and np.all(qq <= Q_MAX - margin))


def clamp_to_limits(q):
    qq = _pad5(q)
    return np.clip(qq, Q_MIN, Q_MAX)


def reachable(target):
    """어깨 기준 도달반경 안인지 (빠른 사전판정)."""
    d = np.linalg.norm(np.asarray(target, dtype=float).ravel()[:3] - SHOULDER)
    return (MIN_REACH - 1e-9) <= d <= (MAX_REACH + 1e-9)


# ------------------------------------------------------- 위치 IK (q1~q4)
def _dls_solve(p_des, q, tol, max_iter, damping):
    """damped least squares 반복. (q, err) 반환."""
    for _ in range(max_iter):
        e = p_des - fk_pos(q)
        if np.linalg.norm(e) < tol:
            break
        J = jacobian_pos(q)
        JJt = J @ J.T + (damping ** 2) * np.eye(3)
        dq = J.T @ np.linalg.solve(JJt, e)
        n = np.linalg.norm(dq)
        if n > 0.3:                     # 스텝 제한 (발산 방지)
            dq *= 0.3 / n
        q = q + dq
    q = wrap_to_pi(q)
    return q, float(np.linalg.norm(p_des - fk_pos(q)))


def ik(target, q0=None, tol=1e-6, max_iter=300, damping=1e-3):
    """
    위치 IK (q1~q4). 3식 4미지수이므로 q0에서 가장 가까운 해로 수렴시킨다.

    반환: (q, err)
        q   : (4,) ndarray.  **항상 배열을 반환한다 (None 아님).**
              도달 불가한 목표면 가장 가까이 뻗은 자세를 돌려준다.
        err : 목표점과 실제 TCP 위치의 거리(m).
              호출부에서 err > 1e-3 이면 작업영역 밖/특이점으로 판정하면 된다.

    특이점 대응: q=0(팔이 완전히 곧게 선 자세)은 특이점이라 그 축 근처
    목표점에서 제자리에 갇힌다. 1차 시도가 실패하면 팔꿈치를 굽힌 시드로
    재시작해서 가장 좋은 해를 고른다.
    """
    p_des = np.asarray(target, dtype=float).ravel()[:3]
    q_seed = np.zeros(4) if q0 is None else _pad5(q0)[:4].copy()

    q_best, err_best = _dls_solve(p_des, q_seed.copy(), tol, max_iter, damping)
    if err_best < 1e-4:
        return q_best, err_best

    # 재시작 시드: q1은 목표 방향으로 맞추고 팔꿈치를 여러 각도로 굽힌다
    q1_hint = float(np.arctan2(p_des[1], p_des[0] - BASE_X))
    seeds = []
    for q1 in (q1_hint, float(wrap_to_pi(q1_hint + np.pi))):
        for bend in (0.6, 1.2, -0.6, -1.2, 2.0, -2.0):
            seeds.append(np.array([q1, bend, -2.0 * bend, bend]))
    seeds.append(np.array([q1_hint, 0.5, 1.0, 1.5]))
    seeds.append(np.array([q1_hint, -0.5, -1.0, -1.5]))

    for sd in seeds:
        q_try, err_try = _dls_solve(p_des, sd, tol, max_iter, damping)
        if err_try < err_best:
            q_best, err_best = q_try, err_try
            if err_best < 1e-4:
                break
    return q_best, err_best


# --------------------------------------- fix 모드 IK (말단이 지면을 향함)
def _ik_fix_solutions(target, allow_approx=False):
    """
    q2+q3+q4 = LEVEL_SUM 구속 하의 해석해(closed form).
    q1 두 갈래 x 팔꿈치 두 갈래 = 최대 4개.

    allow_approx=False : 정확히 도달하는 해만 (없으면 빈 리스트)
    allow_approx=True  : 도달 불가여도 손목 거리를 클램프해서 가장 가까운 자세를 만든다

    반환: list of (q(4,), err)
    """
    p_des = np.asarray(target, dtype=float).ravel()[:3]
    dx = p_des[0] - BASE_X
    dy = p_des[1]
    dz = p_des[2] - D1

    r = np.hypot(dx, dy)
    sols = []

    # u4 = pi 이므로 마지막 링크는 (x', z') 평면에서 (0, -TCP) 기여
    #  -> 손목점(joint4 축)은 목표점에서 +TCP 만큼 위
    for branch in (0, 1):
        if branch == 0:
            q1 = np.arctan2(dy, dx)
            wx = r
        else:
            q1 = float(wrap_to_pi(np.arctan2(dy, dx) + np.pi))
            wx = -r
        wz = dz + _TCP_Z

        d = np.hypot(wx, wz)
        d_max = L2 + L3
        d_min = abs(L2 - L3)

        if d > d_max + 1e-12 or d < d_min - 1e-12:
            if not allow_approx:
                continue
            # 도달 가능한 반경으로 클램프 (방향은 유지)
            d_c = float(np.clip(d, d_min, d_max))
            if d < 1e-12:
                wx, wz = d_c, 0.0
            else:
                wx, wz = wx * d_c / d, wz * d_c / d
            d = d_c

        cos_e = (d * d - L2 * L2 - L3 * L3) / (2.0 * L2 * L3)
        cos_e = float(np.clip(cos_e, -1.0, 1.0))

        for sign in (+1.0, -1.0):
            q3 = sign * np.arccos(cos_e)
            k1 = L2 + L3 * np.cos(q3)
            k2 = L3 * np.sin(q3)
            q2 = np.arctan2(wx, wz) - np.arctan2(k2, k1)
            q4 = LEVEL_SUM - q2 - q3

            q = wrap_to_pi(np.array([q1, q2, q3, q4]))
            err = float(np.linalg.norm(fk_pos(q) - p_des))

            # 중복 제거
            if any(np.max(np.abs(wrap_to_pi(q - s[0]))) < 1e-6 for s in sols):
                continue
            sols.append((q, err))

    return sols


def ik_fix(target, q0=None):
    """
    위치 + 말단 지면 향함(TCP z = [0,0,-1], 즉 q2+q3+q4 = pi).

    반환: (q, err)
        q   : (4,) ndarray. **항상 배열을 반환한다 (None 아님).**
        err : 목표점과 실제 TCP 위치의 거리(m).
    """
    p_des = np.asarray(target, dtype=float).ravel()[:3]

    sols = _ik_fix_solutions(target, allow_approx=False)
    if not sols:
        sols = _ik_fix_solutions(target, allow_approx=True)
    if not sols:
        # 여기까지 오는 일은 사실상 없지만 안전장치
        q = np.zeros(4) if q0 is None else _pad5(q0)[:4].copy()
        return wrap_to_pi(q), float(np.linalg.norm(fk_pos(q) - p_des))

    if q0 is not None:
        qref = _pad5(q0)[:4]
        sols.sort(key=lambda s: (s[1] > 1e-6,
                                 np.linalg.norm(wrap_to_pi(s[0] - qref))))
    else:
        sols.sort(key=lambda s: s[1])
    return sols[0]


def ik_fix_candidates(target, q0=None, n=4):
    """
    fix 모드 해 후보들 (충돌 회피용). 해석해라 최대 4개가 전부다.
    정확히 도달하는 해가 없으면 빈 리스트를 반환한다
    (호출부에서 ik_fix 폴백으로 처리).

    반환: list of (q(4,), err)   q0에서 가까운 순 정렬
    """
    sols = [s for s in _ik_fix_solutions(target, allow_approx=False) if s[1] < 1e-6]
    if not sols:
        return []
    if q0 is not None:
        qref = _pad5(q0)[:4]
        sols.sort(key=lambda s: np.linalg.norm(wrap_to_pi(s[0] - qref)))
    return sols[:max(0, int(n))]


# ---------------------------------------------------------------- self test
def _urdf_chain(q):
    """검증용: URDF 관절 체인을 그대로 곱한 참값."""
    def ry(t):
        c, s = np.cos(t), np.sin(t)
        T = np.eye(4); T[0, 0] = c; T[0, 2] = s; T[2, 0] = -s; T[2, 2] = c
        return T
    def tr(x, y, z):
        T = np.eye(4); T[:3, 3] = (x, y, z)
        return T
    q1, q2, q3, q4, q5 = _pad5(q)
    T = tr(-0.0945, 0, 0.105) @ _rz(q1)
    T = T @ tr(0, 0, 0.09)   @ ry(q2)
    T = T @ tr(0, 0, 0.195)  @ ry(q3)
    T = T @ tr(0, 0, 0.165)  @ ry(q4)
    T = T @ tr(0, 0, 0.087)  @ _rz(q5)
    T = T @ tr(0, 0, 0.1245)
    return T


def _self_test():
    rng = np.random.default_rng(0)
    np.set_printoptions(precision=6, suppress=True)

    # 1) DH FK  vs  URDF 체인
    e = 0.0
    for _ in range(5000):
        q = rng.uniform(-np.pi, np.pi, 5)
        e = max(e, np.abs(fk(q) - _urdf_chain(q)).max())
    print("[1] fk(DH) vs URDF chain   max err = %.3e" % e)

    # 2) fk_pos 폐형식 vs fk
    e = 0.0
    for _ in range(5000):
        q = rng.uniform(-np.pi, np.pi, 5)
        e = max(e, np.abs(fk_pos(q) - fk(q)[:3, 3]).max())
    print("[2] fk_pos vs fk           max err = %.3e" % e)

    # 3) 야코비안 vs 수치미분
    e = 0.0
    for _ in range(200):
        q = rng.uniform(-2, 2, 4)
        J = jacobian_pos(q)
        Jn = np.zeros((3, 4))
        for k in range(4):
            dq = np.zeros(4); dq[k] = 1e-6
            Jn[:, k] = (fk_pos(q + dq) - fk_pos(q - dq)) / 2e-6
        e = max(e, np.abs(J - Jn).max())
    print("[3] jacobian vs numeric    max err = %.3e" % e)

    # 4) q5는 TCP 위치에 영향 없어야 함
    e = 0.0
    for _ in range(500):
        q = rng.uniform(-np.pi, np.pi, 5)
        q2 = q.copy(); q2[4] = rng.uniform(-np.pi, np.pi)
        e = max(e, np.linalg.norm(fk(q)[:3, 3] - fk(q2)[:3, 3]))
    print("[4] q5 -> TCP position     max err = %.3e" % e)

    # 5) 위치 IK 왕복
    ok = 0; worst = 0.0; tried = 0
    for _ in range(500):
        qt = rng.uniform(-2.0, 2.0, 4)
        p = fk_pos(qt)
        tried += 1
        q, _e = ik(p, q0=qt + rng.uniform(-0.3, 0.3, 4))
        if q is not None:
            err = np.linalg.norm(fk_pos(q) - p)
            worst = max(worst, err)
            if err < 1e-5:
                ok += 1
    print("[5] ik round-trip          %d/%d ok,  max err = %.3e" % (ok, tried, worst))

    # 6) fix IK: 위치 + 말단 방향
    ok = 0; tried = 0; worst_p = 0.0; worst_z = 0.0
    for _ in range(500):
        qt = rng.uniform(-1.5, 1.5, 3)
        q4 = LEVEL_SUM - qt[1] - qt[2]
        qfull = np.array([qt[0], qt[1], qt[2], q4])
        p = fk_pos(qfull)
        tried += 1
        cands = ik_fix_candidates(p, q0=qfull, n=4)
        if cands:
            q, _e = cands[0]
            worst_p = max(worst_p, np.linalg.norm(fk_pos(q) - p))
            worst_z = max(worst_z, np.linalg.norm(fk(q)[:3, 2] - np.array([0, 0, -1.0])))
            ok += 1
    print("[6] ik_fix                 %d/%d ok,  pos err = %.3e,  z-axis err = %.3e"
          % (ok, tried, worst_p, worst_z))

    # 7) 기준 수치
    print()
    print("home TCP          =", fk(np.zeros(5))[:3, 3])
    print("home TCP z-axis   =", fk(np.zeros(5))[:3, 2])
    print("shoulder          =", SHOULDER)
    print("max reach         = %.4f m" % MAX_REACH)


if __name__ == "__main__":
    _self_test()
