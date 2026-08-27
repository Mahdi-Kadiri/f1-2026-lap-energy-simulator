#!/usr/bin/env python3
"""
Select the curvature filter window by physical consistency.

The problem
-----------
Curvature from GPS is a second derivative, so it is limited by the position
scatter. Lateral line variation d across a filter window L produces an
apparent curvature of roughly 8d/L^2, i.e. an apparent radius of L^2/(8d):

    window   d = 1 m   d = 2 m   d = 3 m
     20 m      50 m      25 m      17 m
     30 m     112 m      56 m      38 m
     50 m     312 m     156 m     104 m
     80 m     800 m     400 m     267 m

With a 30 m window and 2 m of lap-to-lap scatter the noise floor is R ~ 56 m,
which is the same order as a real F1 corner. A short window cannot separate
the two. A long window rounds real corners off. The window must therefore be
chosen, and choosing it by eye - or by whichever value makes lap time match -
is tuning, not engineering.

The criterion
-------------
Curvature must be consistent with the measured speed trace. At every point
the car's actual lateral acceleration is

    a_y = v^2 * kappa

with v MEASURED and kappa derived. That a_y cannot exceed what the car can
physically produce at that speed, which the four-wheel model computes
independently from mass, aero and tyre data.

So: sweep the window, and for each one ask what fraction of the lap demands
more lateral grip than the car has. Over-resolved curvature shows up
immediately as impossible lateral g. The right window is the SHORTEST one
that is physically achievable - shortest because that preserves the most
genuine corner detail.

Note this uses no lap-time information at all, so it cannot be accused of
being tuned to produce a flattering answer.

Usage
-----
    python tune_curvature.py --year 2026 --event Barcelona --session Q
"""

from __future__ import annotations
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA_HOME = os.path.join(os.path.expanduser('~'), 'f1sim_data')


def sweep(year, event, session, windows, laps=10, samples=2000, plot=False):
    from data.fastf1_track import build_centreline
    from data.vehicle_params import CAR_2026, _sync_aero
    from vehicle.four_wheel import FourWheelModel
    from vehicle.f1_2026_params import GEO_2026, AERO_2026, TYRE_2026
    from validate import load_reference_lap

    _sync_aero()
    fw = FourWheelModel(GEO_2026, AERO_2026, TYRE_2026)

    print("fetching reference lap...")
    ref = load_reference_lap(year, event, session)
    print(f"  {ref['driver']}  {ref['lap_time']:.3f} s\n")

    from validate import align_to_centreline

    rows = []
    for w in windows:
        cl = build_centreline(year, event, session, laps, samples, smooth=float(w))
        s, kap = cl.s, cl.kappa

        # Closed-curve check: for any simple closed circuit the heading
        # integral is exactly +/- 2 pi. A different value means the curvature
        # magnitude carries a systematic scale error.
        ds = np.gradient(s)
        turn = float(np.sum(kap * ds) / (2.0 * np.pi))
        # Align the telemetry Distance axis to the centreline arc length,
        # then use the aligned measured speed.
        shift, v = align_to_centreline(s, kap, ref['s'], ref['v'], cl.length_m)

        a_y = v * v * np.abs(kap)                       # demanded, m/s^2
        cap = np.array([fw.max_lateral_acceleration(max(vi, 5.0), 1e-3)
                        for vi in v])                   # available, m/s^2

        excess = a_y - cap
        frac_over = float(np.mean(excess > 0))
        p99 = float(np.percentile(a_y, 99) / 9.80665)
        R = np.abs(1.0 / np.where(np.abs(kap) > 1e-5, kap, np.nan))
        Rmin = float(np.nanmin(R))

        rows.append((w, Rmin, p99, 100 * frac_over))
        print(f"  window {w:3.0f} m | tightest R {Rmin:6.1f} m | "
              f"99th pct {p99:4.2f} g | impossible {100*frac_over:5.1f}% | "
              f"align {shift:+4.0f} m | heading integral {turn:+.3f} (want ±1)")

    print()
    ok = [r for r in rows if r[3] < 2.0]
    if ok:
        best = min(ok, key=lambda r: r[0])
        print(f"  RECOMMENDED window: {best[0]:.0f} m")
        print(f"    tightest resolved radius {best[1]:.1f} m")
        print(f"    demands more grip than available for only {best[3]:.1f}% of the lap")
        print(f"\n  Set this in extract_telemetry.py with --smooth {best[0]:.0f}")
    else:
        print("  No window tested is physically consistent.")
        print("  Either extend the sweep to longer windows, or the vehicle")
        print("  model's grip capability is too low (check mu0 and ClA).")

    if plot:
        _plot(rows)
    return rows


def _plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.style.use('dark_background')
    w = [r[0] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].plot(w, [r[1] for r in rows], 'o-', color='#5DCAA5')
    ax[0].set_xlabel('Filter window (m)'); ax[0].set_ylabel('Tightest radius (m)')
    ax[0].set_title('Resolved corner radius', fontsize=11)
    ax[1].plot(w, [r[2] for r in rows], 'o-', color='#EF9F27')
    ax[1].set_xlabel('Filter window (m)'); ax[1].set_ylabel('99th pct lateral demand (g)')
    ax[1].set_title('Implied lateral acceleration', fontsize=11)
    ax[2].plot(w, [r[3] for r in rows], 'o-', color='#E24B4A')
    ax[2].axhline(2.0, ls='--', lw=.8, color='#888780', label='2% threshold')
    ax[2].set_xlabel('Filter window (m)'); ax[2].set_ylabel('% of lap physically impossible')
    ax[2].set_title('Consistency check', fontsize=11); ax[2].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=.25, lw=.5)
    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/curvature_window_sweep.png', dpi=150, facecolor='black')
    print("\n  plot saved: outputs/curvature_window_sweep.png")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--year', type=int, default=2026)
    ap.add_argument('--event', default='Barcelona')
    ap.add_argument('--session', default='Q')
    ap.add_argument('--laps', type=int, default=10)
    ap.add_argument('--samples', type=int, default=2000)
    ap.add_argument('--windows', default='25,35,45,55,70,90',
                    help='comma-separated filter windows in metres')
    ap.add_argument('--plot', action='store_true')
    args = ap.parse_args()

    windows = [float(x) for x in args.windows.split(',')]
    print("Curvature window selection by physical consistency")
    print("=" * 58)
    sweep(args.year, args.event, args.session, windows,
          args.laps, args.samples, args.plot)


if __name__ == '__main__':
    main()
