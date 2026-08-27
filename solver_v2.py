"""
Quasi-steady-state minimum lap time solver, v2.

Changes from Layer 1
--------------------
1. LAP PERIODIC. The forward/backward passes are iterated with the lap
   treated as a closed loop, so v(0) == v(L). Layer 1 started the lap at an
   arbitrary speed and finished at a different one, which corrupted the
   opening sector and the lap time.

2. NO ARTIFICIAL SPEED CAP. Layer 1 clipped corner speed at 120 m/s, which
   meant reported top speed was a guard clause rather than a physical
   result. Top speed is now set by the drag/power balance.

3. SELECTABLE CORNER MODEL. 'point_mass' reproduces Layer 1. 'four_wheel'
   uses the Tier 2 load-transfer model. Both are kept so the cost of load
   transfer can be measured directly on the same track and car.

Method
------
Standard three-pass QSS:
    1. Grip-limited speed at every point (the corner speed limit).
    2. Backward pass: propagate braking capability upstream.
    3. Forward pass: propagate acceleration capability downstream.
    4. Element-wise minimum.

Wrapped in an outer loop that carries v(L) round to v(0) until the lap
closes.

Limitations
-----------
- Quasi-steady-state: no transient weight transfer, no yaw dynamics.
- Fixed racing line; the trajectory is an input, not an output.
- No tyre thermal model, no degradation.
- Aero state (Corner/Straight Mode) is a supplied boolean per point, not
  solved from the FIA Activation Zones.
"""

from __future__ import annotations
import math
import numpy as np

G = 9.80665


# ------------------------------------------------------------ corner speed

def corner_speed_point_mass(kappa_abs, car):
    """
    Layer 1 friction circle limit, retained for comparison.

        m v^2 kappa = mu (m g + 0.5 rho ClA v^2)

    Solved analytically for v. If downforce alone can carry the corner the
    result is unbounded, which is why Layer 1 needed a cap; here we return
    infinity and let the power/drag balance set top speed instead.
    """
    if kappa_abs < 1e-9:
        return math.inf
    A = car['m_kg'] * kappa_abs - 0.5 * car['rho_kgm3'] * car['ClA_m2'] * car['mu']
    B = car['mu'] * car['m_kg'] * G
    if A <= 0.0:
        return math.inf
    return math.sqrt(B / A)


def corner_speed_four_wheel(kappa, model, gradient=0.0, straight_mode=False):
    """Tier 2 limit: bisection inside the four-wheel load transfer model."""
    if abs(kappa) < 1e-9:
        return math.inf
    return model.max_corner_speed(kappa, gradient, straight_mode)


# ------------------------------------------------- longitudinal capability

def accel_capability(v, car, grade, mguk_fn, year, kappa=0.0,
                     fw_model=None, straight_mode=False):
    """
    Maximum forward acceleration (m/s^2) at speed v.

    With fw_model supplied, the traction limit comes from the friction
    ellipse on the driven axle, so cornering consumes traction capacity.
    Power limit is unchanged - it is a powertrain constraint, not a tyre one.
    """
    q = 0.5 * car['rho_kgm3'] * v * v
    F_df = q * car['ClA_m2']
    F_drag = q * car['CdA_m2']
    N = car['m_kg'] * G + F_df

    if fw_model is not None:
        F_traction = fw_model.max_longitudinal_acceleration(
            v, kappa, grade, straight_mode, braking=False) * car['m_kg']
    else:
        F_traction = car['mu'] * car.get('rear_load_frac', 0.55) * N
    P = min(car['P_ice_kW'] * 1000.0 + mguk_fn(v, year) * 1000.0,
            car['P_total_kW'] * 1000.0)
    F_power = P / max(v, 0.5)

    F_drive = min(F_traction, F_power)
    F_rr = car['Crr'] * N
    F_grade = car['m_kg'] * G * math.sin(math.atan(grade))
    return (F_drive - F_drag - F_rr - F_grade) / car['m_kg']


def brake_capability(v, car, grade, kappa=0.0, fw_model=None,
                     straight_mode=False):
    """
    Maximum deceleration magnitude (m/s^2) at speed v. Positive.

    With fw_model supplied, braking capacity comes from the friction ellipse
    across all four corners, so trail-braking into a corner is grip-limited
    rather than assuming full mu is available longitudinally.
    """
    q = 0.5 * car['rho_kgm3'] * v * v
    F_df = q * car['ClA_m2']
    F_drag = q * car['CdA_m2']
    N = car['m_kg'] * G + F_df

    if fw_model is not None:
        F_brake = fw_model.max_longitudinal_acceleration(
            v, kappa, grade, straight_mode, braking=True) * car['m_kg']
    else:
        F_brake = car['mu'] * N
    F_rr = car['Crr'] * N
    F_grade = car['m_kg'] * G * math.sin(math.atan(grade))
    return (F_brake + F_drag + F_rr + F_grade) / car['m_kg']


def terminal_speed(car, grade=0.0, mguk_fn=None, year=2026):
    """
    Speed at which drive force equals resistance. Replaces the hard cap.
    Found by bisection on accel_capability(v) = 0.
    """
    lo, hi = 10.0, 200.0
    if accel_capability(hi, car, grade, mguk_fn, year) > 0:
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if accel_capability(mid, car, grade, mguk_fn, year) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 0.05:
            break
    return 0.5 * (lo + hi)


# ------------------------------------------------------------------ solver

def solve_lap(s, kappa, grade, car, mguk_fn, year=2026, ds=10.0,
              corner_model='point_mass', fw_model=None,
              straight_mode=None, max_outer=25, tol=0.05, verbose=False):
    """
    Solve the lap-periodic velocity profile.

    Args:
        s, kappa, grade : (N,) track arrays. kappa is SIGNED.
        car             : parameter dict (point-mass path and longitudinal)
        mguk_fn         : callable(v_ms, year) -> MGU-K limit in kW
        corner_model    : 'point_mass' or 'four_wheel'
        fw_model        : FourWheelModel instance, required for 'four_wheel'
        straight_mode   : (N,) bool array of active-aero state, or None
        max_outer       : lap-closure iteration cap
        tol             : closure tolerance on v(0) (m/s)

    Returns dict with v, t, a, lap_time, v_corner, closure_error, iterations.
    """
    N = len(s)
    if straight_mode is None:
        straight_mode = np.zeros(N, dtype=bool)

    fw = fw_model if corner_model == 'four_wheel' else None
    v_term = terminal_speed(car, 0.0, mguk_fn, year)

    # --- pass 1: grip-limited corner speed -----------------------------
    v_corner = np.empty(N)
    for i in range(N):
        if corner_model == 'four_wheel':
            if fw_model is None:
                raise ValueError("four_wheel corner model needs fw_model")
            vc = corner_speed_four_wheel(kappa[i], fw_model, grade[i],
                                         bool(straight_mode[i]))
        else:
            vc = corner_speed_point_mass(abs(kappa[i]), car)
        v_corner[i] = min(vc, v_term)

    # --- passes 2 and 3, iterated for lap closure ----------------------
    v_start = float(v_corner[0])
    closure_err = math.inf
    it = 0

    for it in range(1, max_outer + 1):
        # backward pass (braking), wrapping the end of the lap to the start
        v_b = v_corner.copy()
        v_b[-1] = min(v_b[-1], v_start)
        for i in range(N - 2, -1, -1):
            ab = brake_capability(v_b[i + 1], car, grade[i], kappa[i],
                                  fw, bool(straight_mode[i]))
            v_b[i] = min(v_b[i], math.sqrt(v_b[i + 1] ** 2 + 2.0 * ab * ds))
        # one extra wrap so the constraint at s=0 propagates from the tail
        v_b[0] = min(v_b[0], v_start)

        # forward pass (acceleration), starting from the wrapped speed
        v = v_b.copy()
        v[0] = min(v[0], v_start)
        for i in range(1, N):
            aa = accel_capability(v[i - 1], car, grade[i], mguk_fn, year,
                                  kappa[i], fw, bool(straight_mode[i]))
            v[i] = min(v[i], math.sqrt(max(v[i - 1] ** 2 + 2.0 * max(aa, 0.0) * ds, 0.0)))

        v_end = v[-1]
        # speed carried across the start/finish line into the next lap
        aa = accel_capability(v_end, car, grade[-1], mguk_fn, year,
                              kappa[-1], fw, bool(straight_mode[-1]))
        v_wrap = math.sqrt(max(v_end ** 2 + 2.0 * max(aa, 0.0) * ds, 0.0))
        v_wrap = min(v_wrap, v_corner[0])

        closure_err = abs(v_wrap - v_start)
        if verbose:
            print(f"  iter {it}: v(0)={v_start:6.2f}  v_wrap={v_wrap:6.2f}  err={closure_err:.4f}")
        if closure_err < tol:
            v_start = v_wrap
            break
        v_start = v_wrap

    # --- integrate ------------------------------------------------------
    dt = np.zeros(N)
    for i in range(N - 1):
        dt[i] = ds / max(0.5 * (v[i] + v[i + 1]), 0.1)
    dt[-1] = ds / max(0.5 * (v[-1] + v[0]), 0.1)   # close the loop

    t = np.concatenate(([0.0], np.cumsum(dt)[:-1]))

    a = np.zeros(N)
    for i in range(N - 1):
        a[i] = (v[i + 1] ** 2 - v[i] ** 2) / (2.0 * ds)
    a[-1] = (v[0] ** 2 - v[-1] ** 2) / (2.0 * ds)

    return {
        'v': v, 't': t, 'a': a,
        'lap_time': float(np.sum(dt)),
        'v_corner': v_corner,
        'closure_error': closure_err,
        'iterations': it,
        'terminal_speed': v_term,
        'corner_model': corner_model,
    }
