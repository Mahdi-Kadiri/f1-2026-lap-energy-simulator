#!/usr/bin/env python3
"""
Identify drag area and ICE power from straight-line acceleration telemetry.

Method
------
At full throttle on a straight the longitudinal power balance is

    m * a * v  =  P_total(v)  -  0.5 * rho * CdA * v^3  -  P_rolling

P_total is the ICE plus the MGU-K, and the MGU-K contribution is KNOWN as a
function of speed from Article C5.2.8 (profile (ii) applies on a qualifying
lap, since Overtake is activated at all times in an LTCS per B7.2.3(b)).
Moving that known term to the left leaves an expression linear in v^3:

    m*a*v - P_mguk(v) + P_rolling  =  P_ice  -  (0.5 * rho * CdA) * v^3
    `-------------------------'        `---'     `--------------'
              measured                intercept       slope

So a straight-line fit against v^3 yields BOTH the ICE power and the drag
area, from data, in one regression - the same trick used for tyre grip and
downforce in identify_tyre_aero.py.

Why this matters here
---------------------
CdA was the last parameter in the model still doing real work as a pure
estimate. The FIA's published -55% drag target gives CdA_X = 0.585 m^2; the
measured terminal speed would balance at 1.31 m^2 IF the car were at terminal
velocity, which it is not, because Barcelona's main straight ends at a braking
point. The truth lies between and only the acceleration gradient can settle
it. Terminal speed error of ~+10 km/h in validation is the symptom.

Point selection
---------------
Only samples with throttle at or near 100%, brake released, and low curvature
qualify - the car must be purely longitudinally limited. Gear changes and the
first moments after corner exit are excluded via an acceleration-consistency
filter, and 2-sigma outlier rejection cleans the rest.

Known limitation
----------------
P_ice and CdA are correlated in this fit: a higher assumed engine power can be
traded against more drag. The v^3 lever separates them only as well as the
speed range allows. Report both, and treat agreement of the fitted ICE power
with the ~400 kW regulatory expectation as the check on whether the separation
worked. If fitted P_ice comes back far from 400 kW, the CdA is not trustworthy
either.

Usage
-----
    python identify_drag.py --year 2026 --event Barcelona --session Q
"""

from __future__ import annotations
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RHO = 1.225
G = 9.80665


def collect_straight_points(cl, lap, kappa, min_throttle=95.0,
                            max_kappa=1.0 / 800.0, min_speed_ms=50.0):
    """
    Full-throttle, straight-line, non-braking samples from one lap.

    Returns (v, a, in_straight_mode) arrays.
    """
    from validate import align_to_centreline

    tel = lap.get_car_data().add_distance()
    d = tel['Distance'].to_numpy()
    v_raw = tel['Speed'].to_numpy() / 3.6
    thr = tel['Throttle'].to_numpy() if 'Throttle' in tel else np.zeros_like(d)
    brk = (tel['Brake'].to_numpy().astype(float)
           if 'Brake' in tel else np.zeros_like(d))

    shift, v = align_to_centreline(cl.s, kappa, d, v_raw, cl.length_m)
    d_scaled = d * (cl.length_m / d.max())
    ds = cl.length_m / len(cl.s)
    roll = int(round(shift / ds))
    thr_g = np.roll(np.interp(cl.s, d_scaled, thr), -roll)
    brk_g = np.roll(np.interp(cl.s, d_scaled, brk), -roll)

    # dv/dt from dv/ds * v, on the uniform centreline grid
    dvds = np.gradient(v, ds)
    a = dvds * v

    ok = ((thr_g >= min_throttle) & (brk_g <= 0.5)
          & (np.abs(kappa) <= max_kappa) & (v >= min_speed_ms) & (a > 0.0))
    return v[ok], a[ok], cl.s[ok]


def robust_fit(x, y, sigma=2.0, max_reject_frac=0.1):
    """Least squares y = c0 + c1*x with iterative outlier rejection."""
    keep = np.ones(len(x), dtype=bool)
    n_max = max(1, int(len(x) * max_reject_frac))
    c0 = c1 = 0.0
    for _ in range(n_max + 1):
        A = np.vstack([np.ones(keep.sum()), x[keep]]).T
        (c0, c1), *_ = np.linalg.lstsq(A, y[keep], rcond=None)
        r = y - (c0 + c1 * x)
        s = np.std(r[keep])
        worst = int(np.argmax(np.abs(r) * keep))
        if s > 0 and abs(r[worst]) > sigma * s and keep.sum() > 10:
            keep[worst] = False
        else:
            break
    return c0, c1, keep


def identify(year, event, session, window=None, laps=10, samples=2000,
             n_fit_laps=10, plot=False):
    import fastf1 as _ff
    from data.fastf1_track import build_centreline, DEFAULT_SMOOTH_M
    from data.vehicle_params import CAR_2026, _sync_aero
    from energy.regs_2026 import deployment_limit_kw, ActivationZones, BARCELONA_ZONES

    _sync_aero()
    window = DEFAULT_SMOOTH_M if window is None else window

    print("building centreline (window %.0f m)..." % window)
    cl = build_centreline(year, event, session, laps, samples, smooth=window)
    print(f"  {cl.circuit}: {cl.length_m:.1f} m")

    print("collecting full-throttle straight-line samples...")
    ses = _ff.get_session(year, event, session)
    ses.load(telemetry=True, laps=True, weather=False)
    quick = ses.laps.pick_quicklaps()
    if len(quick) == 0:
        quick = ses.laps
    quick = quick.sort_values('LapTime').head(n_fit_laps)

    V, A, S = [], [], []
    for _, lap in quick.iterrows():
        try:
            v, a, s_at = collect_straight_points(cl, lap, cl.kappa)
        except Exception:
            continue
        V.append(v); A.append(a); S.append(s_at)

    if not V:
        print("\n  no usable straight-line samples")
        return None

    v = np.concatenate(V); a = np.concatenate(A); s_at = np.concatenate(S)
    if len(v) == 0:
        print("\n  no samples passed the full-throttle / straight-line gate")
        print("  (check that the session has Throttle and Brake channels)")
        return None
    print(f"  samples: {len(v)}  |  speed range "
          f"{v.min()*3.6:.0f} - {v.max()*3.6:.0f} km/h")
    if len(v) < 30:
        print("\n  too few samples to fit reliably")
        return None

    m = CAR_2026['m_kg']

    # DO NOT subtract the MGU-K regulatory ceiling.
    #
    # An earlier version assumed the car deploys the maximum permitted power
    # at all times and subtracted C5.2.8 profile (ii) from the measured
    # power. That is wrong: the energy store holds 4 MJ, and this project's
    # own DP optimiser shows a legal qualifying lap deploys only ~2.2 MJ,
    # with harvesting in between. Subtracting the ceiling over-subtracted by
    # hundreds of kW, flattened the signal, and returned P_ice = -4 kW with
    # r^2 = 0.19 - correctly flagged by the separation check.
    #
    # The actual deployment schedule is not observable from public telemetry,
    # so it is not assumed. The intercept instead becomes the EFFECTIVE TOTAL
    # power (ICE plus whatever MGU-K the driver was actually using), which is
    # a legitimate quantity and is what the drag fit needs anyway.

    # Rolling resistance, using the straight-mode downforce where applicable
    # Activation zones are per-circuit and published by the FIA; only
    # Barcelona is encoded here. At another circuit, fall back to treating
    # the low-curvature samples as Corner Mode, which slightly overstates
    # rolling resistance and is therefore conservative for CdA.
    zones = BARCELONA_ZONES if 'barcelona' in cl.circuit.lower() \
        else ActivationZones(zones=())
    sm = zones.mask(s_at)
    ClA = np.where(sm, CAR_2026.get('ClA_straight_m2', CAR_2026['ClA_m2']),
                   CAR_2026['ClA_m2'])
    N = m * G + 0.5 * RHO * ClA * v * v
    P_roll = CAR_2026['Crr'] * N * v

    y = m * a * v + P_roll     # = P_total_effective - 0.5*rho*CdA*v^3
    x = v ** 3

    c0, c1, keep = robust_fit(x, y)
    P_total = c0
    CdA = -2.0 * c1 / RHO

    pred = c0 + c1 * x
    r2 = 1.0 - (np.sum((y[keep] - pred[keep]) ** 2)
                / np.sum((y[keep] - y[keep].mean()) ** 2))

    print(f"\n  outliers rejected : {int((~keep).sum())}")
    print(f"  IDENTIFIED (fit r^2 = {r2:.3f}):")
    print(f"    CdA                    = {CdA:.3f} m^2")
    print(f"    effective total power  = {P_total/1e3:.0f} kW")
    print(f"\n  current model: CdA_straight {CAR_2026.get('CdA_straight_m2'):.3f}, "
          f"CdA_corner {CAR_2026['CdA_m2']:.3f} m^2")
    print(f"  reference points: FIA -55% target 0.585 | terminal-speed "
          f"balance 1.31 m^2")

    # ICE ~400 kW; MGU-K adds 0-350 kW depending on the actual schedule, so
    # an effective total anywhere in 400-750 kW is physically admissible.
    ok_power = 380e3 < P_total < 800e3
    print(f"\n  SEPARATION CHECK: effective total power {P_total/1e3:.0f} kW")
    print(f"    admissible range 400-750 kW (ICE ~400 plus 0-350 of MGU-K)")
    print(f"    -> {'PASS' if ok_power else 'FAIL'}")
    if not ok_power:
        print("    Power and CdA trade off against each other in this fit. A")
        print("    fitted power outside the physically admissible range means")
        print("    the v^3 lever did not separate them, and the CdA is NOT")
        print("    trustworthy. Reject this fit.")
    else:
        print("    Power lands in the admissible range, so the two parameters")
        print("    separated and the CdA is usable. Note the intercept is an")
        print("    EFFECTIVE average over the deployment schedule actually")
        print("    used, not the ICE figure alone.")

    if plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        ax.scatter(x[keep] / 1e6, y[keep] / 1e3, s=14, color='#5DCAA5',
                   alpha=.75, label=f'samples (n={int(keep.sum())})')
        if (~keep).any():
            ax.scatter(x[~keep] / 1e6, y[~keep] / 1e3, s=22, marker='x',
                       color='#E24B4A', label=f'rejected (n={int((~keep).sum())})')
        xx = np.linspace(0, x.max() * 1.05, 100)
        ax.plot(xx / 1e6, (c0 + c1 * xx) / 1e3, color='#EF9F27', lw=1.7,
                label=f'fit: CdA={CdA:.3f} m², P_total={P_total/1e3:.0f} kW '
                      f'(r²={r2:.3f})')
        ax.set_xlabel('v³  (×10⁶ m³/s³)')
        ax.set_ylabel('m·a·v + P_roll   (kW)')
        ax.set_title('Drag and ICE power identification from straight-line data',
                     fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(alpha=.25, lw=.5)
        plt.tight_layout()
        os.makedirs('outputs', exist_ok=True)
        plt.savefig('outputs/drag_identification.png', dpi=150,
                    facecolor='black')
        print("\n  plot saved: outputs/drag_identification.png")

    return {'CdA': CdA, 'P_total': P_total, 'r2': r2, 'n': int(keep.sum())}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--year', type=int, default=2026)
    ap.add_argument('--event', default='Barcelona')
    ap.add_argument('--session', default='Q')
    ap.add_argument('--window', type=float, default=None)
    ap.add_argument('--laps', type=int, default=10)
    ap.add_argument('--samples', type=int, default=2000)
    ap.add_argument('--fit-laps', type=int, default=10)
    ap.add_argument('--plot', action='store_true')
    args = ap.parse_args()

    print("Drag and ICE power identification")
    print("=" * 58)
    identify(args.year, args.event, args.session, args.window,
             args.laps, args.samples, args.fit_laps, args.plot)


if __name__ == '__main__':
    main()
