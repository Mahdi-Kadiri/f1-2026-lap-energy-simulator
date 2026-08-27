#!/usr/bin/env python3
"""
Identify effective tyre grip and downforce from telemetry.

Method
------
At a grip-limited corner apex the car's lateral acceleration equals its
lateral capability:

    a_y = mu * (g + q * SCz / m),   q = 0.5 * rho * v^2

which is LINEAR in v^2:

    a_y = mu*g + (mu * rho * SCz / (2 m)) * v^2
          `----'  `--------------------'
        intercept          slope

So a straight-line fit of apex lateral acceleration against apex speed
squared yields both parameters at once:

    mu  = intercept / g
    SCz = 2 * m * slope / (mu * rho)

This is the standard trackside method for backing tyre grip and downforce
out of corner-speed data. It became usable here only after two things were
certified: the curvature scale (heading integral = -1.000, exact for a
closed circuit) and the distance-axis alignment between telemetry and
centreline (cross-correlation, consistent -22 to -25 m).

Selection of grip-limited points
--------------------------------
Only apex points are usable - elsewhere the car is below its lateral limit
and the equation is an inequality. Apexes are taken as local minima of the
aligned speed trace that coincide with significant curvature. Low-curvature
minima (lifts, traffic) are rejected.

What the fitted values mean
---------------------------
mu here is an EFFECTIVE, load-averaged friction coefficient - the tyre model
in this project makes mu fall with load, so the fitted constant is a mean
over the apex load range, not mu0 at the reference load. SCz likewise
absorbs any speed dependence of the aero platform. Both are exactly what a
lap time simulator needs, which is why the method is standard.

Honest limitation: the fit assumes the driver is at the limit at every
selected apex. A cautious lap under-reports grip; qualifying laps are used
for precisely this reason.

Usage
-----
    python identify_tyre_aero.py --year 2026 --event Barcelona --session Q
"""

from __future__ import annotations
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

G = 9.80665
RHO = 1.225


def find_apexes(s, v, kappa, min_kappa=1.0 / 400.0, min_gap_m=60.0):
    """
    Grip-limited apex points: local speed minima with meaningful curvature.

    Returns indices into the arrays.
    """
    n = len(v)
    w = max(3, n // 150)
    idx = []
    for i in range(w, n - w):
        if v[i] <= v[i - w:i + w + 1].min() and abs(kappa[i]) > min_kappa:
            if not idx or (s[i] - s[idx[-1]]) > min_gap_m:
                idx.append(i)
            elif v[i] < v[idx[-1]]:
                idx[-1] = i
    return np.array(idx, dtype=int)


def collect_lap_apexes(cl, lap, kappa, throttle_max=30.0,
                       fast_corner_ms=45.0):
    """
    Apex points from one lap, gated on the pedal channels.

    Gating logic, and why it is speed-dependent:

    - Brake must be released at every accepted apex. Trail-braking points are
      combined-slip, not pure lateral, and belong off this fit.
    - Below fast_corner_ms, throttle must also be low. A slow apex taken with
      substantial throttle is the driver accelerating through a compromise,
      not cornering at the limit (observed: a 228 km/h, 1.5 g "apex" at high
      throttle collapsed a 9-point fit to r^2 = 0.27).
    - At or above fast_corner_ms, throttle is NOT gated. Fast corners are
      driven at sustained throttle while fully laterally grip-limited - that
      is what a fast corner is. Gating them out was found to delete every
      apex above 150 km/h, leaving the fit's slope (which is what determines
      SCz) extrapolated from a narrow low-speed cluster: r^2 = 0.77 and
      SCz = 6.9 m^2, double any plausible value. The cost of admitting them
      is small and conservative: combined slip places a driven fast corner
      slightly BELOW the pure-lateral line, so SCz is under- rather than
      over-estimated.
    """
    from validate import align_to_centreline
    tel = lap.get_car_data().add_distance()
    d = tel['Distance'].to_numpy()
    v_raw = tel['Speed'].to_numpy() / 3.6
    thr = tel['Throttle'].to_numpy() if 'Throttle' in tel else np.zeros_like(d)
    brk = tel['Brake'].to_numpy() if 'Brake' in tel else np.zeros_like(d)

    shift, v = align_to_centreline(cl.s, kappa, d, v_raw, cl.length_m)
    d_scaled = d * (cl.length_m / d.max())
    ds = cl.length_m / len(cl.s)
    roll = int(round(shift / ds))
    thr_g = np.roll(np.interp(cl.s, d_scaled, thr), -roll)
    brk_g = np.roll(np.interp(cl.s, d_scaled, brk.astype(float)), -roll)

    idx = find_apexes(cl.s, v, kappa)
    keep = [i for i in idx
            if brk_g[i] <= 0.5 and (thr_g[i] <= throttle_max
                                    or v[i] >= fast_corner_ms)]
    return np.array(keep, dtype=int), v, shift


def robust_fit(v_ap, ay_ap, n_reject=None, sigma=2.0):
    """Least squares with iterative outlier rejection (max ~10% of points)."""
    if n_reject is None:
        n_reject = max(1, len(v_ap) // 10)
    keep = np.ones(len(v_ap), dtype=bool)
    c0 = c1 = 0.0
    for _ in range(n_reject + 1):
        A = np.vstack([np.ones(keep.sum()), v_ap[keep] ** 2]).T
        (c0, c1), *_ = np.linalg.lstsq(A, ay_ap[keep], rcond=None)
        r = ay_ap - (c0 + c1 * v_ap ** 2)
        s = np.std(r[keep])
        worst = np.argmax(np.abs(r) * keep)
        if abs(r[worst]) > sigma * s and keep.sum() > 6:
            keep[worst] = False
        else:
            break
    return c0, c1, keep


def identify(year, event, session, window=None, laps=10, samples=2000,
             plot=False, n_fit_laps=10):
    import fastf1 as _ff
    from data.fastf1_track import build_centreline
    from validate import load_reference_lap

    from data.fastf1_track import DEFAULT_SMOOTH_M
    window = DEFAULT_SMOOTH_M if window is None else window
    print("building centreline (window %.0f m)..." % window)
    cl = build_centreline(year, event, session, laps, samples, smooth=window)

    ds_arr = np.gradient(cl.s)
    turn = float(np.sum(cl.kappa * ds_arr) / (2 * np.pi))
    print(f"  heading integral {turn:+.3f}  (must be ±1 for a closed lap)")

    print("collecting apexes from the %d fastest laps..." % n_fit_laps)
    ses = _ff.get_session(year, event, session)
    ses.load(telemetry=True, laps=True, weather=False)
    quick = ses.laps.pick_quicklaps()
    if len(quick) == 0:
        quick = ses.laps
    quick = quick.sort_values('LapTime').head(n_fit_laps)

    v_pts, ay_pts, shifts = [], [], []
    n_gated = 0
    for _, lap in quick.iterrows():
        try:
            keep, v, shift = collect_lap_apexes(cl, lap, cl.kappa)
        except Exception:
            continue
        shifts.append(shift)
        all_idx = find_apexes(cl.s, v, cl.kappa)
        n_gated += len(all_idx) - len(keep)
        for i in keep:
            v_pts.append(v[i])
            ay_pts.append(v[i] ** 2 * abs(cl.kappa[i]))

    v_ap = np.array(v_pts)
    ay_ap = np.array(ay_pts)
    print(f"  apex candidates kept: {len(v_ap)}  "
          f"(pedal gate removed {n_gated})")
    if shifts:
        print(f"  axis offsets: {min(shifts):+.0f} to {max(shifts):+.0f} m "
              f"(consistency check)")
    if len(v_ap) < 8:
        print("\n  too few grip-limited apexes found - cannot fit")
        return None

    c0, c1, keep = robust_fit(v_ap, ay_ap)
    n_out = int((~keep).sum())
    if n_out:
        print(f"  outliers rejected: {n_out}")
    v_fit, ay_fit = v_ap[keep], ay_ap[keep]
    A = np.vstack([np.ones_like(v_fit), v_fit ** 2]).T

    m_car = 768.0
    mu = c0 / G
    scz = 2.0 * m_car * c1 / (mu * RHO) if mu > 0 else float('nan')

    pred = A @ np.array([c0, c1])
    r2 = 1.0 - np.sum((ay_fit - pred) ** 2) / np.sum((ay_fit - ay_fit.mean()) ** 2)

    print(f"\n  apexes used        : {int(keep.sum())} of {len(v_ap)}")
    print(f"  apex speed range   : {v_fit.min()*3.6:.0f} - {v_fit.max()*3.6:.0f} km/h")
    print(f"  apex a_y range     : {ay_fit.min()/G:.2f} - {ay_fit.max()/G:.2f} g")
    print(f"\n  IDENTIFIED (fit r^2 = {r2:.3f}):")
    print(f"    mu  (effective, load-averaged) = {mu:.3f}")
    print(f"    SCz (ClA equivalent)           = {scz:.2f} m^2")
    print(f"\n  current model values:")
    print(f"    mu0 1.60 at 2 kN, k 0.1215     (load-averaged ~1.45-1.55)")
    print(f"    ClA_corner 3.51")
    print(f"\n  Interpretation: if the identified values sit above the model's,")
    print(f"  the model's grip capability is too low by that ratio, and the")
    print(f"  residual 'physically impossible' fraction in tune_curvature is")
    print(f"  explained. Update f1_2026_params.py with these, tagged [IDENT],")
    print(f"  citing this fit - session, driver, r^2 - as the source.")
    print(f"\n  Caution: identified from ONE session. Before trusting them,")
    print(f"  re-run on a second session (FP3, or another driver's lap) and")
    print(f"  check they agree. Identification validated on the data that")
    print(f"  produced it proves nothing.")

    if plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(8, 5.5))
        vv = np.linspace(0, max(v_fit.max() * 1.1, 5), 100)
        ax.scatter(v_fit ** 2, ay_fit / G, color='#5DCAA5', s=28, zorder=3,
                   label=f'apex points (n={int(keep.sum())})')
        if n_out:
            ax.scatter(v_ap[~keep] ** 2, ay_ap[~keep] / G, color='#E24B4A',
                       s=34, marker='x', zorder=3,
                       label=f'rejected outliers (n={n_out})')
        ax.plot(vv ** 2, (c0 + c1 * vv ** 2) / G, color='#EF9F27', lw=1.6,
                label=f'fit: mu={mu:.3f}, SCz={scz:.2f} m²  (r²={r2:.3f})')
        ax.set_xlabel('v² (m²/s²)')
        ax.set_ylabel('Lateral acceleration (g)')
        ax.set_title('Tyre and aero identification from grip-limited apexes',
                     fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(alpha=.25, lw=.5)
        plt.tight_layout()
        os.makedirs('outputs', exist_ok=True)
        os.makedirs('outputs', exist_ok=True)
        plt.savefig('outputs/tyre_aero_identification.png', dpi=150,
                    facecolor='black')
        print("\n  plot saved: outputs/tyre_aero_identification.png")

    return {'mu': mu, 'SCz': scz, 'r2': r2, 'n': int(keep.sum())}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--year', type=int, default=2026)
    ap.add_argument('--event', default='Barcelona')
    ap.add_argument('--session', default='Q')
    ap.add_argument('--window', type=float, default=None,
                    help='curvature filter window (m)')
    ap.add_argument('--laps', type=int, default=10)
    ap.add_argument('--samples', type=int, default=2000)
    ap.add_argument('--plot', action='store_true')
    ap.add_argument('--fit-laps', type=int, default=10,
                    help='number of fastest laps to pool apexes from')
    args = ap.parse_args()

    print("Tyre and aero identification")
    print("=" * 58)
    identify(args.year, args.event, args.session, args.window,
             args.laps, args.samples, args.plot, args.fit_laps)


if __name__ == '__main__':
    main()
