"""
Layer 2 - MGU-K deployment optimisation by dynamic programming.

Objective
---------
Find the MGU-K power schedule u(s) that minimises lap time subject to the
2026 FIA energy regulations.

Why this is not trivial
-----------------------
Layer 1 deployed at the regulatory ceiling wherever the car was accelerating.
That produced ~10 MJ of deployment against a battery that stores 4 MJ, so the
schedule was not merely suboptimal - it was infeasible.

The real problem has structure Layer 1 ignored entirely:

  - The energy store holds 4 MJ (C5.2.9) but up to 8.5 MJ may be recharged
    per lap (C5.2.10). The battery therefore cycles roughly twice per lap.
    State of charge is a continuously integrated state, not a budget that is
    decremented once.

  - Harvest is not a passive by-product of braking. Superclipping - loading
    the MGU-K while the driver is at full throttle - is an active control
    decision that costs straight-line speed to buy energy. It is why 2026
    energy management is hard and why drivers complain about it.

  - A joule is worth more in some places than others. Deployed out of a slow
    corner onto a long straight it buys a lot of time; deployed into an
    already speed-limited section it buys almost none. The optimiser has to
    discover that distribution rather than be told it.

Formulation
-----------
State:   battery state of charge E, discretised over [0, delta_soc_max].
Control: u = MGU-K DC power, -350 kW <= u <= deployment_limit(v, mode).
             u > 0 deploy, u = 0 clip, u < 0 superclip / harvest.
Stage:   track segment of length ds.
Cost:    segment traversal time.

TERMINAL CONDITION - two modes, and the choice is not cosmetic
--------------------------------------------------------------
  'periodic'  SoC(finish) = SoC(start). Correct for a RACE stint, where the
              next lap must also be possible. Forces deployment to equal
              harvest over the lap.

  'deplete'   SoC(start) = full window, SoC(finish) = 0. Correct for a
              QUALIFYING flying lap: the car arrives from an out-lap with a
              full store and any charge remaining at the flag is lap time
              thrown away. The energy budget is then the 4 MJ carried in
              PLUS everything harvested during the lap, subject to the
              per-lap Recharge cap - a far larger budget than periodicity
              allows.

Using 'periodic' to model a qualifying lap understates deployment badly and
was the reason an early version reported only 2.2 MJ deployed on a flying
lap.

Two constraints are handled differently:
  - The 4 MJ window is a hard state bound, enforced by the SoC grid.
  - The 8.5 MJ per-lap Recharge cap is a path constraint on cumulative
    harvest. Rather than lifting it into the state (which would square the
    grid), it is enforced by a Lagrangian penalty on harvest, with the
    multiplier found by bisection so the cap is met with equality.

Not modelled
------------
C5.12 imposes ramp-rate and monotonicity constraints on driver power demand
once the car enters a "power limited" state: a first step down of at most
150 kW held for one second (C5.12.4), demand that cannot increase except via
Boost (C5.12.5), and a reduction rate of 50 or 100 kW/s (C5.12.6). The
trigger condition for that state is specified in FIA-F1-DOC-058, which is not
public. This solver treats u as independently selectable per segment, so
predicted lap times are optimistic by the time lost to the mandated ramp.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

from energy.regs_2026 import deployment_limit_kw, EnergyConfig

G = 9.80665
PERIODIC_PENALTY = 1e-4    # s per J of SoC mismatch at the finish line
DEPLETE_PENALTY  = 1e-3    # s per J left unspent at the flag.
                           # Larger than PERIODIC_PENALTY because leaving
                           # charge on a qualifying lap is a pure loss, and a
                           # soft penalty left ~0.9 MJ unspent.


@dataclass
class DPResult:
    v: np.ndarray                 # optimised velocity profile (m/s)
    u_kW: np.ndarray              # MGU-K power schedule (kW), +deploy -harvest
    soc_MJ: np.ndarray            # battery state of charge (MJ)
    lap_time: float
    energy_deployed_MJ: float
    energy_harvested_MJ: float
    soc_closure_MJ: float         # |SoC(end) - SoC(start)|
    marginal_value: np.ndarray    # d(lap time)/d(energy) per segment, s/MJ
    binding: np.ndarray           # 0 traction, 1 ceiling, 2 energy, 3 corner


def _drag_N(v, car, straight_mode=False):
    CdA = car.get('CdA_straight_m2', car['CdA_m2']) if straight_mode \
        else car['CdA_m2']
    return 0.5 * car['rho_kgm3'] * CdA * v * v


def _downforce_N(v, car, straight_mode=False):
    ClA = car.get('ClA_straight_m2', car['ClA_m2']) if straight_mode \
        else car['ClA_m2']
    return 0.5 * car['rho_kgm3'] * ClA * v * v


def optimise_deployment(s, kappa, grade, v_corner, car, cfg: EnergyConfig,
                        straight_mode=None, ds=10.0,
                        n_soc=41, n_ctrl=9, n_speed=44,
                        fw_model=None, verbose=False) -> DPResult:
    """
    JOINT-STATE dynamic program over (speed, state of charge).

    Why speed must be a state
    -------------------------
    An earlier version kept only SoC as the state and froze speeds from the
    previous rollout. Pricing a control then saw the energy leaving and a
    one-segment time change - but not the speed carried into every following
    segment, which is where deployment's value actually lives. The optimiser
    systematically undervalued deployment (near-zero MGU-K on the main
    straight), and refining the grid made the lap WORSE, because a finer
    grid only resolves the mis-specified problem more precisely. With speed
    in the state, the value of a joule IS the compounded time saved by the
    speed it buys, by construction.

    States   : v on a per-point feasible grid, E on [0, delta_soc_max]
    Controls : propulsive MGU-K power u in [0, C5.2.8 profile], or
               braking to a lower speed with regeneration credited at
               min(350 kW, rear-axle share) per C5.2.7
    Terminal : 'deplete' rewards crossing the line empty (flying lap);
               'periodic' penalises SoC mismatch with the start (race lap)
    Speed at the line is closed (v(0) ~ v(end)) by outer iteration.
    """
    N = len(s)
    if straight_mode is None:
        straight_mode = np.zeros(N, dtype=bool)
    m = car['m_kg']
    E_max = cfg.delta_soc_max_MJ * 1e6
    soc_grid = np.linspace(0.0, E_max, n_soc)
    dE_soc = soc_grid[1] - soc_grid[0]
    overtake = cfg.overtake_active()
    P_ice = cfg.ice_power_kW * 1e3

    v_cap = np.minimum(np.asarray(v_corner, float), 120.0)

    # ---- braking-feasibility ceiling: max speed from which every later
    # corner can still be made. Any state at or below it is reachable.
    v_brk = v_cap.copy()
    for _ in range(2):                     # wrap twice so the lap closes
        for i in range(N - 1, -1, -1):
            j = (i + 1) % N
            vi = max(v_brk[j], 1.0)
            if fw_model is not None:
                ab = fw_model.max_longitudinal_acceleration(
                    vi, kappa[i], grade[i], bool(straight_mode[i]),
                    braking=True)
            else:
                q = 0.5 * car['rho_kgm3'] * vi * vi
                ab = car['mu'] * (m * G + q * car['ClA_m2']) / m
            ab += _drag_N(vi, car, bool(straight_mode[i])) / m
            v_brk[i] = min(v_brk[i], math.sqrt(v_brk[j] ** 2 + 2 * ab * ds))

    v_lo = 12.0
    v_hi = float(v_brk.max())
    vg = np.linspace(v_lo, v_hi, n_speed)          # global speed grid

    def vidx(v, mode='floor'):
        """Speed -> grid index. 'floor' for acceleration results: rounding
        to NEAREST let half the accel steps snap ~1 m/s of unearned speed,
        which compounded cell-by-cell along straights and made the DP lap
        ~5 s optimistic against the QSS solving identical physics. Never
        grant speed the dynamics did not earn."""
        x = (v - v_lo) / (vg[1] - vg[0])
        i = math.floor(x + 1e-9) if mode == 'floor' else round(x)
        return int(np.clip(i, 0, n_speed - 1))

    feas = vg[None, :] <= (v_brk[:, None] + 1e-9)   # (N, n_speed) mask

    def accel_next(vi, u_W, i):
        sm = bool(straight_mode[i])
        P = P_ice + u_W
        F = P / max(vi, 1.0)
        if fw_model is not None:
            F = min(F, fw_model.max_longitudinal_acceleration(
                vi, kappa[i], grade[i], sm, braking=False) * m)
        F -= _drag_N(vi, car, sm)
        F -= car['Crr'] * (m * G + _downforce_N(vi, car, sm))
        F -= m * G * math.sin(math.atan(grade[i]))
        v2 = vi * vi + 2.0 * (F / m) * ds
        return math.sqrt(max(v2, 1.0))

    def solve(lam, k_start, v_start_idx):
        BIG = 1e6
        V = np.full((N + 1, n_speed, n_soc), BIG)
        Uu = np.zeros((N, n_speed, n_soc))          # chosen MGU-K power (W)
        Uv = np.zeros((N, n_speed, n_soc), dtype=int)

        if cfg.terminal_mode == 'deplete':
            V[N, :, :] = DEPLETE_PENALTY * soc_grid[None, :]
        else:
            V[N, :, :] = PERIODIC_PENALTY * np.abs(
                soc_grid[None, :] - soc_grid[k_start])
        # speed closure: only end states near the chosen start speed are free
        pen_v = 0.05 * np.abs(np.arange(n_speed) - v_start_idx)[:, None]
        V[N, :, :] += pen_v

        for i in range(N - 1, -1, -1):
            for a_i in range(n_speed):
                if not feas[i, a_i]:
                    continue
                vi = vg[a_i]
                dt_base = None
                best = None
                # -- propulsive controls ------------------------------
                P_max = deployment_limit_kw(vi, overtake) * 1e3
                for u_W in np.linspace(0.0, P_max, max(2, n_ctrl // 2)):
                    vn = accel_next(vi, u_W, i)
                    vn = min(vn, v_brk[(i + 1) % N])
                    b_i = vidx(vn)
                    if not feas[(i + 1) % N, b_i]:
                        b_i = int(np.max(np.nonzero(feas[(i + 1) % N])[0]))
                    dt = ds / max(0.5 * (vi + vg[b_i]), 1.0)
                    dE = u_W * dt                    # energy OUT of store
                    kshift = dE / dE_soc
                    k0 = np.arange(n_soc) - kshift
                    kf = np.clip(np.round(k0).astype(int), -1, n_soc - 1)
                    ok = k0 >= -0.5
                    cost = np.full(n_soc, BIG)
                    cost[ok] = dt + V[i + 1, b_i, kf[ok]]
                    if best is None:
                        best = cost
                        Uu[i, a_i, :] = u_W
                        Uv[i, a_i, :] = b_i
                    else:
                        better = cost < best
                        best = np.where(better, cost, best)
                        Uu[i, a_i, better] = u_W
                        Uv[i, a_i, better] = b_i
                # -- braking / lift controls (regen credited) ---------
                for b_i in range(a_i - 1, -1, -max(1, n_speed // 8)):
                    vn = vg[b_i]
                    a_req = (vn * vn - vi * vi) / (2 * ds)   # negative
                    if fw_model is not None:
                        a_max = fw_model.max_longitudinal_acceleration(
                            vi, kappa[i], grade[i],
                            bool(straight_mode[i]), braking=True)
                    else:
                        a_max = car['mu'] * G * 1.5
                    if -a_req > a_max + 5.0:
                        break                          # cannot brake harder
                    dt = ds / max(0.5 * (vi + vn), 1.0)
                    P_h = min(350e3, 0.45 * m * (-a_req) * vi)
                    dE_in = P_h * dt                   # energy INTO store
                    kshift = dE_in / dE_soc
                    kf = np.clip(np.floor(np.arange(n_soc) + kshift
                                          ).astype(int), 0, n_soc - 1)
                    cost = dt + lam * dE_in + V[i + 1, vidx(vn),
                                                kf.clip(0, n_soc - 1)]
                    # <=: regeneration wins ties against the propulsive
                    # branch's cap-snap, which otherwise replicates braking
                    # without crediting the harvest.
                    better = cost <= best
                    best = np.where(better, cost, best)
                    Uu[i, a_i, better] = -P_h
                    Uv[i, a_i, better] = vidx(vn)
                V[i, a_i, :] = best

        # rollout
        u = np.zeros(N); v = np.zeros(N); soc = np.zeros(N + 1)
        dt = np.zeros(N)
        a_i = v_start_idx; E = soc_grid[k_start]; soc[0] = E
        for i in range(N):
            k = int(np.clip(round(E / dE_soc), 0, n_soc - 1))
            u_W = Uu[i, a_i, k]; b_i = Uv[i, a_i, k]
            vi, vn = vg[a_i], vg[b_i]
            dt[i] = ds / max(0.5 * (vi + vn), 1.0)
            if u_W >= 0:
                E = max(E - u_W * dt[i], 0.0)
            else:
                E = min(E - u_W * dt[i], E_max)
            u[i] = u_W / 1e3
            v[i] = vi
            soc[i + 1] = E
            a_i = b_i
        return u, v, soc, dt, a_i

    # start speed: iterate to close the lap in speed
    k_start = n_soc - 1 if cfg.terminal_mode == 'deplete' else n_soc // 2
    a0 = vidx(min(v_brk[0], v_hi))
    cap_J = cfg.recharge_max_MJ * 1e6
    best_pack = None
    for _outer in range(3):
        # lam = 0 first: the recharge cap usually does not bind, and each
        # solve is expensive. Bisect only if harvest exceeds the cap.
        lam = 0.0
        u, v, soc, dt, a_end = solve(lam, k_start, a0)
        harv = float(np.sum(np.clip(-u, 0, None) * 1e3 * dt))
        best_pack = (u, v, soc, dt)
        if harv > cap_J:
            lo, hi = 0.0, 5e-7
            for _b in range(5):
                lam = 0.5 * (lo + hi)
                u, v, soc, dt, a_end = solve(lam, k_start, a0)
                harv = float(np.sum(np.clip(-u, 0, None) * 1e3 * dt))
                best_pack = (u, v, soc, dt)
                if harv > cap_J:
                    lo = lam
                else:
                    hi = lam
        if abs(a_end - a0) <= 1:
            break
        a0 = a_end

    u, v, soc, dt = best_pack
    dep = float(np.sum(np.clip(u, 0, None) * 1e3 * dt))
    harv = float(np.sum(np.clip(-u, 0, None) * 1e3 * dt))
    marg = np.zeros(N)
    binding = np.zeros(N, dtype=int)
    for i in range(N):
        lim = deployment_limit_kw(v[i], overtake)
        if abs(kappa[i]) > 1e-6 and v[i] >= v_cap[i] * 0.98:
            binding[i] = 3
        elif u[i] >= lim * 0.98 and lim > 0:
            binding[i] = 1
        elif u[i] < 0:
            binding[i] = 2

    return DPResult(
        v=v, u_kW=u, soc_MJ=soc[:N] / 1e6,
        lap_time=float(np.sum(dt)),
        energy_deployed_MJ=dep / 1e6,
        energy_harvested_MJ=harv / 1e6,
        soc_closure_MJ=float(soc[-1]) / 1e6 if cfg.terminal_mode == 'deplete'
            else abs(soc[-1] - soc[0]) / 1e6,
        marginal_value=marg, binding=binding,
    )
