"""
robot_description / vertical_path_5axis.py
--------------------------------------------------
수직자세(말단이 지면을 향함) 유지 경로 생성.

용도
  1차점(z 높음) -> 수직자세로 도달 -> 정지(dwell) -> 2차점(z 낮음)으로 하강.
  하강 구간은 데카르트 직선을 잘게 나누고 각 점마다 ik_fix 를 풀기 때문에
  경로 '전체'에서 q2+q3+q4 = pi (TCP z = [0,0,-1]) 가 유지된다.

  관절공간 보간만으로는 이 구속이 보장되지 않는다.
  (trajectory_5axis.plan_joint_path 는 중간점을 자동으로 넣기 때문에
   양 끝점이 수직조건을 만족해도 중간에서 벗어날 수 있다.)

주요 함수
  plan_level_descent(q_start, p_start, p_end, hz, lin_speed) -> (traj, T, msg)
  make_hold(q, hz, sec)                                      -> traj
  sort_by_height(p_a, p_b)                                   -> (높은점, 낮은점)
"""

import numpy as np

from .kinematics_ee_5axis import (ik_fix, fk, fk_pos, jacobian_pos, wrap_to_pi,
                                  LEVEL_SUM, N_JOINTS, BASE_X, D1, L2, L3, _TCP_Z)


def sort_by_height(p_a, p_b):
    """z 가 큰 점을 1차점, 작은 점을 2차점으로 돌려준다."""
    p_a = np.asarray(p_a, float).ravel()[:3]
    p_b = np.asarray(p_b, float).ravel()[:3]
    if p_a[2] >= p_b[2]:
        return p_a, p_b
    return p_b, p_a


def _smootherstep(u):
    """6u^5 - 15u^4 + 10u^3 : 시작/끝 속도와 가속도가 0."""
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * u * (u * (u * 6.0 - 15.0) + 10.0)


def make_hold(q, hz, sec):
    """같은 자세를 sec 초 동안 유지하는 구간 (정지)."""
    n = max(1, int(round(float(hz) * float(sec))))
    return np.tile(np.asarray(q, float).ravel()[:N_JOINTS], (n, 1))


def level_error(q):
    """수직조건에서 얼마나 벗어났는지 (rad). 0 이면 완벽히 수직."""
    q = np.asarray(q, float).ravel()
    return float(abs(wrap_to_pi(q[1] + q[2] + q[3] - LEVEL_SUM)))


def plan_level_descent(q_start, p_start, p_end, hz=240.0, lin_speed=0.05,
                       pos_tol=1e-4, level_tol=1e-6, max_joint_step=0.25):
    """
    p_start -> p_end 를 데카르트 직선으로 이동하되 매 점에서 수직자세를 유지.

    q_start   : 출발 관절각 (4,). p_start 에 이미 수직자세로 서 있어야 한다.
    lin_speed : 말단 직선속도 (m/s)
    max_joint_step : 인접 샘플 간 허용 관절 변화(rad). 넘으면 해가 튄 것으로 보고 실패.

    반환 (traj, T, msg)
      traj : (N,4) 관절 경로. 실패 시 None
      T    : 소요시간(초)
      msg  : 실패 사유 (성공이면 '')
    """
    q0 = np.asarray(q_start, float).ravel()[:N_JOINTS].copy()
    p0 = np.asarray(p_start, float).ravel()[:3]
    p1 = np.asarray(p_end, float).ravel()[:3]

    dist = float(np.linalg.norm(p1 - p0))
    if dist < 1e-6:
        return np.array([q0]), 0.0, ''

    lin_speed = max(1e-3, float(lin_speed))
    # smootherstep 은 평균속도가 최고속도의 1/2 이므로 시간을 2배로 잡는다
    T = 2.0 * dist / lin_speed
    n = max(2, int(np.ceil(T * float(hz))))

    traj = np.zeros((n + 1, N_JOINTS))
    traj[0] = q0
    q_prev = q0

    for i in range(1, n + 1):
        s = _smootherstep(i / float(n))
        p_i = p0 + s * (p1 - p0)

        q_i, err = ik_fix(p_i, q0=q_prev)
        if q_i is None:
            return None, 0.0, '하강 %d/%d 지점에서 IK 실패 (%.3f, %.3f, %.3f)' % \
                   (i, n, p_i[0], p_i[1], p_i[2])
        if err > pos_tol:
            return None, 0.0, \
                '하강 %d/%d 지점 도달 불가 (잔차 %.4f m). fix 모드 작업영역 밖.' % (i, n, err)

        lv = level_error(q_i)
        if lv > 1e-3:
            return None, 0.0, '하강 %d/%d 지점에서 수직조건 이탈 (%.4f rad)' % (i, n, lv)

        # 해가 다른 가지로 튀는 것 방지 (팔꿈치 뒤집힘 등)
        jump = float(np.max(np.abs(wrap_to_pi(q_i - q_prev))))
        if jump > max_joint_step:
            return None, 0.0, \
                '하강 %d/%d 지점에서 자세가 급변 (%.3f rad). 특이점 근처일 수 있음.' % (i, n, jump)

        traj[i] = q_i
        q_prev = q_i

    return traj, T, ''


def plan_two_point_level(q_now, p_high, p_low, hz=240.0,
                         approach_planner=None, dwell_sec=0.5, lin_speed=0.05,
                         checker=None, q_candidates=None):
    """
    (선택) 전체 시퀀스를 한 번에 만드는 헬퍼.
    노드에서 직접 조립해도 되지만, 단독 테스트용으로 남겨둔다.

    approach_planner(q_from, q_to) -> (traj, T, n)   : 보통 plan_joint_path
    q_candidates                                     : 1차점 fix IK 후보 리스트

    반환 (traj, T, msg)
    """
    if q_candidates is None:
        q_a, err = ik_fix(p_high, q0=q_now)
        if q_a is None or err > 1e-4:
            return None, 0.0, '1차점 fix IK 실패 (잔차 %.4f m)' % (err if q_a is not None else -1)
        q_candidates = [q_a]

    for q_a in q_candidates:
        if checker is not None and not checker.check_config(q_a)[0]:
            continue

        seg1, T1, _n = approach_planner(q_now, q_a)
        if checker is not None and not checker.check_path(seg1)[0]:
            continue

        seg_hold = make_hold(q_a, hz, dwell_sec)

        seg2, T2, msg = plan_level_descent(q_a, p_high, p_low, hz=hz, lin_speed=lin_speed)
        if seg2 is None:
            return None, 0.0, msg
        if checker is not None and not checker.check_path(seg2)[0]:
            continue

        traj = np.vstack([seg1, seg_hold, seg2])
        return traj, T1 + dwell_sec + T2, ''

    return None, 0.0, '충돌 없는 1차점 자세를 찾지 못함'


# ---------------------------------------------------------------- 도달 진단
def fix_reach_report(p_arm):
    """
    fix 모드(말단 지면 향함) 도달 가능성 진단.

    수직자세에서는 TCP 가 joint5 축 위 _TCP_Z 만큼 아래에 있으므로,
    손목점 W = p + [0, 0, _TCP_Z] 가 어깨에서 2링크 거리
    |L2-L3| ~ (L2+L3) 안에 들어와야 한다.

    반환 dict
      ok      : 도달 가능 여부
      d       : 어깨~손목점 거리 (m)
      d_min   : 최소 (m)
      d_max   : 최대 (m)
      r       : 어깨축 기준 수평거리 (m)
      z_max   : 같은 x,y 에서 도달 가능한 최대 z (없으면 None)
      msg     : 사람이 읽을 진단 문구
    """
    p = np.asarray(p_arm, float).ravel()[:3]
    dx = p[0] - BASE_X
    dy = p[1]
    r = float(np.hypot(dx, dy))
    wz = p[2] + _TCP_Z - D1            # 손목점의 어깨 기준 높이

    d = float(np.hypot(r, wz))
    d_max = L2 + L3
    d_min = abs(L2 - L3)
    ok = (d_min - 1e-9) <= d <= (d_max + 1e-9)

    z_max = None
    if r <= d_max:
        z_max = float(D1 - _TCP_Z + np.sqrt(max(0.0, d_max * d_max - r * r)))

    if ok:
        msg = '도달 가능 (어깨~손목 %.3f m, 한계 %.3f m)' % (d, d_max)
    elif d > d_max:
        if z_max is None:
            msg = ('수평거리 %.3f m 가 2링크 길이 %.3f m 를 넘어 어떤 z 로도 불가. '
                   '레일을 움직이거나 목표를 팔 쪽으로 당기세요.' % (r, d_max))
        else:
            msg = ('%.1f mm 초과. 이 x,y 에서는 z 를 %.4f 이하로 내려야 합니다 '
                   '(현재 %.4f, 팔 기준).' % ((d - d_max) * 1000.0, z_max, p[2]))
    else:
        msg = ('어깨에 너무 가까움 (%.3f m < %.3f m). 목표를 바깥으로 옮기세요.'
               % (d, d_min))

    return dict(ok=ok, d=d, d_min=d_min, d_max=d_max, r=r, z_max=z_max, msg=msg)


# ------------------------------------------------ 수직조건 완화 버전
def ik_level_soft(target, q0, k=1.0, iters=300, tol=1e-9, lam=0.02, step_max=0.3):
    """
    위치를 1순위, 수직조건을 2순위로 푸는 IK (널스페이스 우선순위).

    엄격한 ik_fix 는 q2+q3+q4 = pi 를 등식으로 걸어서, 손목점이 2링크
    범위(0.36 m) 밖이면 해 자체가 없다. 여기서는
      1순위 : 목표 위치에 정확히 도달        (3자유도 사용)
      2순위 : 남는 여유자유도 1개로 수직에 최대한 근접
    로 풀기 때문에, 도달 불가한 곳에서도 자세만 조금 기울여 위치를 맞춘다.

    반환 (q, pos_err, level_err)
    """
    p_des = np.asarray(target, float).ravel()[:3]
    q = np.asarray(q0, float).ravel()[:N_JOINTS].copy()
    grad = np.array([0.0, 1.0, 1.0, 1.0])
    I4 = np.eye(N_JOINTS)

    # 2순위 항은 뒤로 갈수록 줄여서, 마지막에는 순수 위치 수렴만 남긴다.
    # (널스페이스 항이 끝까지 살아 있으면 위치 잔차가 1e-3 수준에서 멈춘다)
    decay_until = 0.6 * iters

    for it in range(iters):
        e = p_des - fk_pos(q)
        J = jacobian_pos(q)
        Jp = J.T @ np.linalg.solve(J @ J.T + (lam ** 2) * np.eye(3), np.eye(3))

        dq = Jp @ e                                   # 1순위: 위치
        k_eff = k * max(0.0, 1.0 - it / decay_until)
        if k_eff > 0.0:
            e_lvl = float(wrap_to_pi(LEVEL_SUM - (q[1] + q[2] + q[3])))
            dq = dq + (I4 - Jp @ J) @ (k_eff * e_lvl * grad)   # 2순위: 수직

        n = np.linalg.norm(dq)
        if n < tol:
            break
        if n > step_max:
            dq *= step_max / n
        q = q + dq

    q = wrap_to_pi(q)
    return q, float(np.linalg.norm(p_des - fk_pos(q))), level_error(q)


def plan_level_descent_relaxed(q_start, p_start, p_end, hz=240.0, lin_speed=0.05,
                               level_tol=np.radians(25.0), pos_tol=1e-4,
                               k=1.0, max_joint_step=0.25):
    """
    plan_level_descent 의 완화 버전.

    각 샘플에서 먼저 엄격한 ik_fix 를 시도하고, 도달 불가하면
    ik_level_soft 로 넘어가 자세를 level_tol 이내에서 기울인다.
    위치는 pos_tol 안으로 맞춰야 하며, 못 맞추면 실패.

    반환 (traj, T, max_level_err, msg)
    """
    q0 = np.asarray(q_start, float).ravel()[:N_JOINTS].copy()
    p0 = np.asarray(p_start, float).ravel()[:3]
    p1 = np.asarray(p_end, float).ravel()[:3]

    dist = float(np.linalg.norm(p1 - p0))
    if dist < 1e-6:
        return np.array([q0]), 0.0, level_error(q0), ''

    lin_speed = max(1e-3, float(lin_speed))
    T = 2.0 * dist / lin_speed
    n = max(2, int(np.ceil(T * float(hz))))

    traj = np.zeros((n + 1, N_JOINTS))
    traj[0] = q0
    q_prev = q0
    worst_lv = level_error(q0)

    for i in range(1, n + 1):
        s = _smootherstep(i / float(n))
        p_i = p0 + s * (p1 - p0)

        q_i, err = ik_fix(p_i, q0=q_prev)
        lv = level_error(q_i) if q_i is not None else 9.9

        if q_i is None or err > pos_tol:
            # 엄격 해가 없으면 수직조건을 완화해서 위치를 우선 맞춘다
            q_i, err, lv = ik_level_soft(p_i, q_prev, k=k)
            if err > pos_tol:
                return None, 0.0, 0.0, \
                    '하강 %d/%d 지점 위치 도달 실패 (잔차 %.4f m).' % (i, n, err)
            if lv > level_tol:
                return None, 0.0, 0.0, \
                    '하강 %d/%d 지점 기울기 %.1f도 로 허용치 %.1f도 초과.' % \
                    (i, n, np.degrees(lv), np.degrees(level_tol))

        jump = float(np.max(np.abs(wrap_to_pi(q_i - q_prev))))
        if jump > max_joint_step:
            return None, 0.0, 0.0, \
                '하강 %d/%d 지점에서 자세 급변 (%.3f rad).' % (i, n, jump)

        traj[i] = q_i
        q_prev = q_i
        worst_lv = max(worst_lv, lv)

    return traj, T, worst_lv, ''


# ------------------------------------------- 관절 속도 한계 자동 맞춤
# Dynamixel PROFILE_VELOCITY 단위 = 0.229 rev/min
# 관절 최대 각속도[deg/s] = PV * 0.229 * 6 / gear
DXL_PV_UNIT_DPS = 0.229 * 360.0 / 60.0      # 1 단위당 모터 deg/s = 1.374

def joint_speed_limits(profile_velocity, gears):
    """
    profile_velocity : {모터ID: PV값}  (대표 모터 1개씩)
    gears            : [q1,q2,q3,q4] 기어비
    반환 : (4,) 관절 최대 각속도 [rad/s]
    """
    dps = np.array([profile_velocity[i] * DXL_PV_UNIT_DPS / g
                    for i, g in zip((1, 2, 3, 5), gears)], float)
    return np.radians(dps)


def path_joint_speed(traj, hz):
    """경로에서 관절별 최대 각속도 [rad/s] (4,)."""
    traj = np.asarray(traj, float)
    if len(traj) < 2:
        return np.zeros(N_JOINTS)
    return np.abs(np.diff(traj, axis=0)).max(axis=0) * float(hz)


def plan_level_descent_limited(q_start, p_start, p_end, hz=240.0, lin_speed=0.05,
                               vmax=None, margin=0.9, relaxed=False,
                               level_tol=np.radians(25.0), min_speed=0.005):
    """
    plan_level_descent(_relaxed) 를 돌린 뒤, 관절 속도가 vmax 를 넘으면
    넘지 않을 때까지 직선속도를 낮춰 다시 생성한다.

    모터가 낼 수 없는 속도를 명령하면 관절마다 뒤처지는 정도가 달라
    수직조건이 깨지고 움직임이 끊겨 보인다. 그걸 미리 막는다.

    vmax : (4,) 관절 최대 각속도 [rad/s]. None 이면 제한 없음.

    반환 (traj, T, used_speed, info)
      info : 사람이 읽을 요약 문구
    """
    def gen(spd):
        if relaxed:
            t, T, _mx, msg = plan_level_descent_relaxed(
                q_start, p_start, p_end, hz=hz, lin_speed=spd, level_tol=level_tol)
        else:
            t, T, msg = plan_level_descent(q_start, p_start, p_end,
                                           hz=hz, lin_speed=spd)
        return t, T, msg

    spd = float(lin_speed)
    traj, T, msg = gen(spd)
    if traj is None:
        return None, 0.0, spd, msg
    if vmax is None:
        return traj, T, spd, ''

    vmax = np.asarray(vmax, float).ravel()[:N_JOINTS]
    for _ in range(6):
        v = path_joint_speed(traj, hz)
        ratio = np.max(v / np.maximum(vmax, 1e-9))
        if ratio <= 1.0:
            break
        new_spd = max(min_speed, spd * margin / ratio)
        if new_spd >= spd - 1e-6:
            break
        spd = new_spd
        traj, T, msg = gen(spd)
        if traj is None:
            return None, 0.0, spd, msg

    v = path_joint_speed(traj, hz)
    worst = int(np.argmax(v / np.maximum(vmax, 1e-9)))
    info = ('관절속도 q1~q4 = %s deg/s (한계 %s), 최대부하 q%d'
            % (np.degrees(v).round(1), np.degrees(vmax).round(1), worst + 1))
    if abs(spd - lin_speed) > 1e-6:
        info = ('하강속도 %.3f -> %.3f m/s 로 자동 감속 (모터 속도한계). '
                % (lin_speed, spd)) + info
    return traj, T, spd, info


# ------------------------------------------------- 시간 재배분 (retiming)
def retime_path(traj, vmax, hz, margin=0.9, ramp_frac=0.15, ramp_scale=3.0,
                max_time=120.0):
    """
    경로의 '형상'은 그대로 두고 '시간'만 다시 배분한다.

    전체 속도를 균일하게 낮추면, 작업영역 경계처럼 야코비안이 나쁜 한
    구간 때문에 경로 전체가 느려진다. 여기서는 구간마다 필요한 시간을
    따로 계산해 어려운 곳만 느리게 지나간다.

      dt_i = max_j |dq_ij| / (vmax_j * margin)

    양 끝은 ramp_frac 구간에서 최대 ramp_scale 배까지 늘려 부드럽게
    출발/정지하게 한다.

    반환 (traj_out, T, info)
    """
    traj = np.asarray(traj, float)
    n = len(traj)
    if n < 2 or vmax is None:
        return traj, (n - 1) / float(hz), ''

    vmax = np.asarray(vmax, float).ravel()[:N_JOINTS] * float(margin)
    dq = np.abs(np.diff(traj, axis=0))
    dt = (dq / np.maximum(vmax, 1e-9)).max(axis=1)      # (n-1,)
    dt = np.maximum(dt, 1e-6)

    # 양 끝 완만하게
    m = max(1, int(round((n - 1) * ramp_frac)))
    for i in range(m):
        w = 1.0 - _smootherstep(i / float(m))           # 1 -> 0
        f = 1.0 + (ramp_scale - 1.0) * w
        dt[i] *= f
        dt[-(i + 1)] *= f

    t = np.concatenate([[0.0], np.cumsum(dt)])
    T = float(t[-1])
    if T > max_time:
        return None, 0.0, '재배분 후 소요시간 %.1f초 로 너무 김 (한계 %.0f초).' % (T, max_time)

    # 균일 시간 간격으로 재샘플링
    n_out = max(2, int(np.ceil(T * float(hz))) + 1)
    t_new = np.linspace(0.0, T, n_out)
    out = np.zeros((n_out, N_JOINTS))
    for j in range(N_JOINTS):
        out[:, j] = np.interp(t_new, t, traj[:, j])

    v = path_joint_speed(out, hz)
    info = ('시간 재배분: %.2f초, 관절속도 %s deg/s (한계 %s)'
            % (T, np.degrees(v).round(1), np.degrees(vmax / margin).round(1)))
    return out, T, info


def plan_level_descent_retimed(q_start, p_start, p_end, hz=240.0, lin_speed=0.05,
                               vmax=None, margin=0.9, relaxed=False,
                               level_tol=np.radians(25.0), max_time=120.0):
    """
    형상 생성(등간격) -> 속도한계에 맞춰 시간 재배분.

    plan_level_descent_limited 는 전체 속도를 균일하게 낮추기 때문에
    경계 근처 한 지점 때문에 전 구간이 느려진다. 이 함수는 구간별로
    시간을 다르게 줘서 필요한 곳만 느리게 간다.

    반환 (traj, T, info)
    """
    if relaxed:
        traj, _T, _mx, msg = plan_level_descent_relaxed(
            q_start, p_start, p_end, hz=hz, lin_speed=lin_speed, level_tol=level_tol)
    else:
        traj, _T, msg = plan_level_descent(q_start, p_start, p_end,
                                           hz=hz, lin_speed=lin_speed)
    if traj is None:
        return None, 0.0, msg

    if vmax is None:
        return traj, (len(traj) - 1) / float(hz), ''

    out, T, info = retime_path(traj, vmax, hz, margin=margin, max_time=max_time)
    if out is None:
        return None, 0.0, info
    return out, T, info


# ------------------------------------- 끝점 엄격 / 중간 완화
def _taper(u, mode):
    """u(0~1) 에서 허용 기울기 비율. 엄격해야 하는 끝에서 0."""
    if mode == 'both':
        return float(np.sin(np.pi * u))          # 양끝 0, 중앙 최대
    if mode == 'end':
        return float(1.0 - _smootherstep(u))     # 시작 최대, 끝 0
    return 1.0                                    # none: 제한 없음


def plan_level_descent_ends(q_start, p_start, p_end, hz=240.0, lin_speed=0.05,
                            mode='both', level_tol=np.radians(25.0),
                            pos_tol=1e-4, max_joint_step=0.25):
    """
    끝점 엄격 / 중간 완화 하강 경로.

    mode
      'both' : 1차점과 2차점 모두 정확히 수직이어야 한다. 중간만 완화.
               (수직 도달영역은 볼록한 공이라 양끝이 가능하면 중간도
                거의 항상 가능하므로, 실질적으로 전 구간 엄격과 같다)
      'end'  : 2차점(흡착 지점)만 정확히 수직. 1차점과 중간은 기울기 허용.
               접근 자세를 기울일 수 있어 사용 가능한 좌표가 넓어진다.
      'none' : 제한 없음 (완전 완화).

    허용 기울기는 엄격해야 하는 끝에서 0 이 되도록 서서히 줄어든다.

    반환 (traj, T, max_level, msg)
    """
    p0 = np.asarray(p_start, float).ravel()[:3]
    p1 = np.asarray(p_end, float).ravel()[:3]
    q0 = np.asarray(q_start, float).ravel()[:N_JOINTS].copy()

    # --- 끝점 검사 ---
    if mode in ('both', 'end'):
        q_e, e_e = ik_fix(p1, q0=q0)
        if q_e is None or e_e > 1e-6:
            r = fix_reach_report(p1)
            return None, 0.0, 0.0, '2차점이 정확한 수직자세로 도달 불가. %s' % r['msg']
    if mode == 'both':
        q_s, e_s = ik_fix(p0, q0=q0)
        if q_s is None or e_s > 1e-6:
            r = fix_reach_report(p0)
            return None, 0.0, 0.0, '1차점이 정확한 수직자세로 도달 불가. %s' % r['msg']
        if level_error(q0) > 1e-3:
            return None, 0.0, 0.0, '출발 자세가 수직이 아닙니다 (%.2f도).' % np.degrees(level_error(q0))

    dist = float(np.linalg.norm(p1 - p0))
    if dist < 1e-6:
        return np.array([q0]), 0.0, level_error(q0), ''

    T = 2.0 * max(1e-3, float(lin_speed)) ** -1 * dist
    n = max(2, int(np.ceil(T * float(hz))))

    traj = np.zeros((n + 1, N_JOINTS))
    traj[0] = q0
    q_prev = q0
    worst = level_error(q0)

    for i in range(1, n + 1):
        u = i / float(n)
        s = _smootherstep(u)
        p_i = p0 + s * (p1 - p0)
        allow = level_tol * _taper(u, mode)

        q_i, err = ik_fix(p_i, q0=q_prev)
        lv = level_error(q_i) if q_i is not None else 9.9

        if q_i is None or err > pos_tol:
            if allow <= 1e-6:
                return None, 0.0, 0.0, \
                    '%d/%d 지점: 엄격해야 하는 구간인데 수직 도달 불가 (잔차 %.4f m).' \
                    % (i, n, err if q_i is not None else -1)
            q_i, err, lv = ik_level_soft(p_i, q_prev)
            if err > pos_tol:
                return None, 0.0, 0.0, '%d/%d 지점 위치 도달 실패 (잔차 %.4f m).' % (i, n, err)
            if lv > allow:
                return None, 0.0, 0.0, \
                    '%d/%d 지점 기울기 %.1f도 > 허용 %.1f도 (끝점 근처는 더 엄격).' \
                    % (i, n, np.degrees(lv), np.degrees(allow))

        jump = float(np.max(np.abs(wrap_to_pi(q_i - q_prev))))
        if jump > max_joint_step:
            return None, 0.0, 0.0, '%d/%d 지점 자세 급변 (%.3f rad).' % (i, n, jump)

        traj[i] = q_i
        q_prev = q_i
        worst = max(worst, lv)

    return traj, T, worst, ''


def plan_descent_ends_retimed(q_start, p_start, p_end, hz=240.0, lin_speed=0.05,
                              mode='both', level_tol=np.radians(25.0),
                              vmax=None, margin=0.9, max_time=120.0):
    """plan_level_descent_ends 결과를 관절 속도한계에 맞춰 시간 재배분."""
    traj, T, mx, msg = plan_level_descent_ends(
        q_start, p_start, p_end, hz=hz, lin_speed=lin_speed,
        mode=mode, level_tol=level_tol)
    if traj is None:
        return None, 0.0, 0.0, msg
    if vmax is None:
        return traj, T, mx, ''
    out, T2, info = retime_path(traj, vmax, hz, margin=margin, max_time=max_time)
    if out is None:
        return None, 0.0, 0.0, info
    return out, T2, mx, info


# --------------------------------- 관절공간 하강 (직선 포기, 수직 유지)
def plan_joint_descent(q_start, p_start, p_end, hz=240.0, lin_speed=0.05,
                       pos_tol=1e-6, max_joint_step=0.25):
    """
    데카르트 직선 하강이 실패할 때의 대안.

    1차점과 2차점 각각에서 정확한 수직해(ik_fix)를 구하고,
    그 사이를 '모든 관절 공통 시간 프로파일'로 보간한다.

    핵심 성질:
      두 끝점이 q2+q3+q4 = pi 를 만족하고 s 를 공통으로 쓰면
        (1-s)*pi + s*pi = pi
      이므로 경로 전체에서 수직조건이 그대로 유지된다.
      잃는 것은 TCP 가 직선으로 내려가지 않는다는 점뿐이다.

    직선 하강은 중간 지점이 작업영역을 벗어나면 실패하지만,
    이 방식은 관절공간에서 두 유효 자세를 잇는 것이라 항상 성립한다.

    반환 (traj, T, max_level, msg)
    """
    q0 = np.asarray(q_start, float).ravel()[:N_JOINTS].copy()
    p0 = np.asarray(p_start, float).ravel()[:3]
    p1 = np.asarray(p_end, float).ravel()[:3]

    if level_error(q0) > 1e-3:
        return None, 0.0, 0.0, '출발 자세가 수직이 아닙니다 (%.2f도).' % np.degrees(level_error(q0))

    q1_, err = ik_fix(p1, q0=q0)
    if q1_ is None or err > pos_tol:
        r = fix_reach_report(p1)
        return None, 0.0, 0.0, '2차점 수직해 없음 (잔차 %.4f m). %s' % (err, r['msg'])

    dq = wrap_to_pi(q1_ - q0)
    if float(np.max(np.abs(dq))) > np.pi:
        return None, 0.0, 0.0, '2차점 자세가 너무 멀리 떨어져 있습니다.'

    dist = float(np.linalg.norm(p1 - p0))
    T = 2.0 * dist / max(1e-3, float(lin_speed))
    n = max(2, int(np.ceil(T * float(hz))))

    traj = np.zeros((n + 1, N_JOINTS))
    worst = 0.0
    for i in range(n + 1):
        s = _smootherstep(i / float(n))
        traj[i] = wrap_to_pi(q0 + s * dq)       # 공통 s -> 수직조건 보존
        worst = max(worst, level_error(traj[i]))

    jump = float(np.max(np.abs(np.diff(traj, axis=0))))
    if jump > max_joint_step:
        return None, 0.0, 0.0, '관절 변화가 큼 (%.3f rad).' % jump

    return traj, T, worst, ''


def plan_joint_descent_retimed(q_start, p_start, p_end, hz=240.0, lin_speed=0.05,
                               vmax=None, margin=0.9, max_time=120.0):
    """plan_joint_descent + 관절 속도한계 시간 재배분."""
    traj, T, mx, msg = plan_joint_descent(q_start, p_start, p_end,
                                          hz=hz, lin_speed=lin_speed)
    if traj is None:
        return None, 0.0, 0.0, msg
    if vmax is None:
        return traj, T, mx, ''
    out, T2, info = retime_path(traj, vmax, hz, margin=margin, max_time=max_time)
    if out is None:
        return None, 0.0, 0.0, info
    return out, T2, mx, info


# ------------------------------------ 수직 유지 관절 이동 (복귀 등)
def plan_sync_joint_move(q_from, q_to, hz=240.0, vmax=None, margin=0.9,
                         n_base=240, max_time=120.0):
    """
    두 자세를 '공통 시간 프로파일'로 잇는다.

    모든 관절에 같은 s(t) 를 쓰므로
        (q2+q3+q4)(s) = (1-s)*A + s*B
    가 되고, 양 끝이 모두 pi 면 경로 전체에서 pi 가 유지된다.
    한쪽만 수직이면 기울기가 그 값에서 0 으로 단조 감소한다.

    trajectory_5axis.plan_joint_path 는 중간점을 자동으로 넣기 때문에
    이 성질이 보장되지 않는다. 그래서 복귀 구간에는 이 함수를 쓴다.

    반환 (traj, T, max_level, info). 실패 시 (None, 0, 0, 사유)
    """
    q0 = np.asarray(q_from, float).ravel()[:N_JOINTS]
    q1 = np.asarray(q_to, float).ravel()[:N_JOINTS]
    dq = wrap_to_pi(q1 - q0)

    if float(np.max(np.abs(dq))) < 1e-9:
        return np.array([q0]), 0.0, level_error(q0), ''

    n = max(2, int(n_base))
    traj = np.zeros((n + 1, N_JOINTS))
    for i in range(n + 1):
        traj[i] = wrap_to_pi(q0 + _smootherstep(i / float(n)) * dq)

    if vmax is not None:
        out, T, info = retime_path(traj, vmax, hz, margin=margin, max_time=max_time)
        if out is None:
            return None, 0.0, 0.0, info
        traj = out
    else:
        T = n / float(hz)
        info = ''

    worst = max(level_error(q) for q in traj)
    return traj, T, worst, info
