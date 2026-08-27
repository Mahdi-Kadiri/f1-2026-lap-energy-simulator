#!/usr/bin/env python3
"""
Run the full analysis. No network required.

This is the entry point for everything the simulator produces offline:

    1. 2025 vs 2026 regulatory comparison  (each car on its own parameters)
    2. Point mass vs four-wheel            (the cost of load transfer)
    3. Deployment profile comparison       (Overtake vs baseline, LTCS)
    4. LLTD and load-sensitivity sweep     (uncertainty band)
    5. DP energy optimisation              (the legal lap)

Usage
-----
    python run_analysis.py                 # everything
    python run_analysis.py --quick         # skip the sweep and the DP
    python run_analysis.py --ds 20         # coarser discretisation, faster

Plots are written to outputs/. Nothing here touches the internet - for the
telemetry validation see extract_telemetry.py and validate.py.
"""

from __future__ import annotations
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.barcelona_track import get_track_segments, TOTAL_LAP_M
from data.vehicle_params import CAR_2025, CAR_2026, _sync_aero
from solver_v2 import solve_lap, corner_speed_four_wheel
from vehicle.four_wheel import FourWheelModel
from vehicle.f1_2026_params import (GEO_2026, AERO_2026, TYRE_2026,
                                    GEO_2025, AERO_2025, TYRE_2025)
from energy.regs_2026 import (deployment_limit_kw, BARCELONA_ZONES,
                              EnergyConfig, Session)

OUT = 'outputs'


def banner(t):
    print("\n" + "=" * 64)
    print("  " + t)
    print("=" * 64)


def mguk_2025(v_ms, year=2025):
    """2025 MGU-K: flat 120 kW."""
    return 120.0


def mguk_2026_overtake(v_ms, year=2026):
    """
    2026 qualifying. Qualifying is an LTCS, and in an LTCS Overtake is
    enabled and activated at all times (B7.2.3(b)), so profile (ii) applies.
    """
    return deployment_limit_kw(v_ms, overtake_active=True)


def mguk_2026_baseline(v_ms, year=2026):
    """2026 profile (i), Overtake not active."""
    return deployment_limit_kw(v_ms, overtake_active=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ds', type=float, default=10.0,
                    help='track discretisation in metres (default 10)')
    ap.add_argument('--quick', action='store_true',
                    help='skip the parameter sweep and the DP optimiser')
    ap.add_argument('--no-plots', action='store_true')
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    t_start = time.time()

    # ---------------------------------------------------------- track ----
    s, kappa, grade, sector = get_track_segments(args.ds)
    print(f"\nBarcelona: {TOTAL_LAP_M} m, {len(s)} points at {args.ds} m spacing")
    print(f"  corners: {(kappa < 0).sum()} right, {(kappa > 0).sum()} left samples")

    # 2026 has two-state active aero; 2025 does not (DRS is not modelled).
    sm_2026 = BARCELONA_ZONES.mask(s)
    sm_2025 = np.zeros(len(s), dtype=bool)
    _sync_aero()   # keep CAR_* dicts consistent with the AeroModel objects
    print(f"  2026 Straight Mode active over {100*sm_2026.mean():.0f}% of the lap")

    fw25 = FourWheelModel(GEO_2025, AERO_2025, TYRE_2025)
    fw26 = FourWheelModel(GEO_2026, AERO_2026, TYRE_2026)

    # ------------------------------------------- 1. regulatory delta ----
    banner("1. 2025 vs 2026 regulatory comparison")
    r25 = solve_lap(s, kappa, grade, CAR_2025, mguk_2025, 2025, args.ds,
                    'four_wheel', fw_model=fw25, straight_mode=sm_2025)
    r26 = solve_lap(s, kappa, grade, CAR_2026, mguk_2026_overtake, 2026,
                    args.ds, 'four_wheel', fw_model=fw26, straight_mode=sm_2026)

    print(f"  2025 : {r25['lap_time']:7.3f} s | top {r25['v'].max()*3.6:6.1f} "
          f"| min {r25['v'].min()*3.6:5.1f} km/h")
    print(f"  2026 : {r26['lap_time']:7.3f} s | top {r26['v'].max()*3.6:6.1f} "
          f"| min {r26['v'].min()*3.6:5.1f} km/h")
    d = r26['lap_time'] - r25['lap_time']
    print(f"  delta: {d:+7.3f} s  -> 2026 {'slower' if d > 0 else 'FASTER (check!)'}")
    print(f"\n  mechanism, R = 120 m corner:")
    print(f"    2025 {fw25.max_corner_speed(1/120.)*3.6:.1f} km/h  vs  "
          f"2026 {fw26.max_corner_speed(1/120.)*3.6:.1f} km/h")
    print(f"  each car uses its OWN aero and geometry; tyre model identical,")
    print(f"  so the comparison isolates the regulation change.")

    print("\n  sector splits (2026):")
    for k in (1, 2, 3):
        m = sector == k
        print(f"    S{k} : {r26['t'][m][-1] - r26['t'][m][0]:6.3f} s")

    # --------------------------------------- 2. cost of load transfer ----
    banner("2. Point mass vs four-wheel (cost of load transfer)")
    pm = solve_lap(s, kappa, grade, CAR_2026, mguk_2026_overtake, 2026,
                   args.ds, 'point_mass', straight_mode=sm_2026)
    print(f"  point mass : {pm['lap_time']:7.3f} s")
    print(f"  four wheel : {r26['lap_time']:7.3f} s")
    print(f"  load transfer costs {r26['lap_time'] - pm['lap_time']:+.3f} s "
          f"at k = {TYRE_2026.load_sensitivity_k}")

    # ------------------------------------- 3. deployment profile ---------
    banner("3. Deployment profile (regulatory detail)")
    rb = solve_lap(s, kappa, grade, CAR_2026, mguk_2026_baseline, 2026,
                   args.ds, 'four_wheel', fw_model=fw26, straight_mode=sm_2026)
    print(f"  profile (i)  baseline      : {rb['lap_time']:7.3f} s")
    print(f"  profile (ii) Overtake, LTCS: {r26['lap_time']:7.3f} s")
    print(f"  qualifying correction      : {r26['lap_time'] - rb['lap_time']:+.3f} s")
    print("  (a qualifying lap is an LTCS; Overtake is active at all times)")

    # ---------------------------------------------- 4. sweep -------------
    sweep = None
    if not args.quick:
        banner("4. LLTD and load-sensitivity sweep")
        import copy
        ks = [0.05, 0.10, 0.15, 0.20]
        lltds = np.linspace(0.40, 0.70, 7)
        Ktot = GEO_2026.roll_stiffness_front + GEO_2026.roll_stiffness_rear
        ds_s = max(args.ds, 25.0)
        s2, k2, g2, _ = get_track_segments(ds_s)
        sm2 = BARCELONA_ZONES.mask(s2)
        pm2 = solve_lap(s2, k2, g2, CAR_2026, mguk_2026_overtake, 2026, ds_s,
                        'point_mass', straight_mode=sm2)['lap_time']

        grid = np.zeros((len(ks), len(lltds)))
        for i, kk in enumerate(ks):
            for j, L in enumerate(lltds):
                g = copy.deepcopy(GEO_2026)
                g.roll_stiffness_front = Ktot * L
                g.roll_stiffness_rear = Ktot * (1 - L)
                t = copy.deepcopy(TYRE_2026)
                t.load_sensitivity_k = kk
                grid[i, j] = solve_lap(
                    s2, k2, g2, CAR_2026, mguk_2026_overtake, 2026, ds_s,
                    'four_wheel', fw_model=FourWheelModel(g, AERO_2026, t),
                    straight_mode=sm2)['lap_time']
            print(f"  k = {kk:.2f} done")

        print(f"\n  {'k':>5} {'best LLTD':>11} {'lap (s)':>9} {'cost (s)':>10}")
        for i, kk in enumerate(ks):
            j = int(np.argmin(grid[i]))
            print(f"  {kk:5.2f} {100*lltds[j]:10.0f}% {grid[i, j]:9.3f} "
                  f"{grid[i, j]-pm2:+10.3f}")
        print("\n  optimal LLTD moves with the tyre parameter, so it cannot be")
        print("  quoted for an F1 car without declaring the assumption behind it.")
        sweep = (ks, lltds, grid, pm2)

    # ------------------------------------------------- 5. DP -------------
    dp = None
    if not args.quick:
        banner("5. DP energy optimisation")
        from energy.deployment_dp import optimise_deployment
        ds_dp = max(args.ds, 20.0)
        s3, k3, g3, _ = get_track_segments(ds_dp)
        sm3 = BARCELONA_ZONES.mask(s3)
        vc = np.minimum([corner_speed_four_wheel(kk, fw26, gg, bool(mm))
                         if abs(kk) > 1e-9 else 120.0
                         for kk, gg, mm in zip(k3, g3, sm3)], 120.0)
        print("  solving both terminal conditions (~40 s)...")
        runs = {}
        for mode, label in (('deplete', 'qualifying flying lap'),
                            ('periodic', 'race stint lap')):
            cfg = EnergyConfig(session=Session.LTCS, terminal_mode=mode)
            runs[mode] = optimise_deployment(s3, k3, g3, vc, CAR_2026, cfg,
                                             sm3, ds_dp, n_soc=41, n_ctrl=11)
            d = runs[mode]
            print(f"    {label:22s}: {d.lap_time:7.3f} s | deployed "
                  f"{d.energy_deployed_MJ:5.2f} MJ | harvested "
                  f"{d.energy_harvested_MJ:5.2f} MJ | SoC "
                  f"{d.soc_MJ[0]:.2f} -> {d.soc_MJ[-1]:.2f}")
        dp = runs['deplete']
        print(f"\n  unconstrained (illegal) : {r26['lap_time']:7.3f} s")
        print(f"  qualifying (deplete)    : {runs['deplete'].lap_time:7.3f} s"
              f"   -> energy constraint costs "
              f"{runs['deplete'].lap_time - r26['lap_time']:+.3f} s")
        print(f"  race stint (periodic)   : {runs['periodic'].lap_time:7.3f} s"
              f"   -> having to break even costs "
              f"{runs['periodic'].lap_time - runs['deplete'].lap_time:+.3f} s")
        print("\n  A flying lap arrives full and crosses the line empty, so it")
        print("  spends the store plus everything harvested. A race lap must")
        print("  finish where it started. C5.2.9's 4 MJ is a state-of-charge")
        print("  WINDOW, not a per-lap deployment budget - the per-lap limit")
        print("  is on Recharge (8.5 MJ, C5.2.10).")

    # ---------------------------------------------- plots ----------------
    if not args.no_plots:
        make_plots(s, sector, kappa, r25, r26, pm, sweep, dp)

    banner(f"done in {time.time() - t_start:.1f} s  -  plots in {OUT}/")


def make_plots(s, sector, kappa, r25, r26, pm, sweep, dp):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.style.use('dark_background')

    # --- 2025 vs 2026 ---
    fig, ax = plt.subplots(2, 1, figsize=(15, 7), sharex=True,
                           gridspec_kw={'height_ratios': [3, 1.4]})
    ax[0].plot(s, r25['v']*3.6, color='#E24B4A', lw=1.4,
               label=f"2025  {r25['lap_time']:.3f} s")
    ax[0].plot(s, r26['v']*3.6, color='#5DCAA5', lw=1.4,
               label=f"2026  {r26['lap_time']:.3f} s")
    ax[0].set_ylabel('Speed (km/h)'); ax[0].legend(fontsize=9, loc='lower right')
    ax[0].grid(alpha=.2, lw=.5)
    ax[0].set_title('Barcelona — 2025 vs 2026, four-wheel model', fontsize=12)
    dv = (r26['v'] - r25['v'])*3.6
    ax[1].fill_between(s, dv, 0, where=dv >= 0, color='#5DCAA5', alpha=.65,
                       label='2026 faster')
    ax[1].fill_between(s, dv, 0, where=dv < 0, color='#E24B4A', alpha=.65,
                       label='2025 faster')
    ax[1].axhline(0, color='#888780', lw=.6)
    ax[1].set_ylabel('Δv (km/h)'); ax[1].set_xlabel('Lap distance (m)')
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.2, lw=.5)
    plt.tight_layout()
    plt.savefig(f'{OUT}/1_2025_vs_2026.png', dpi=150, facecolor='black')
    plt.close()

    # --- point mass vs four wheel ---
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(s, pm['v']*3.6, color='#888780', lw=1.2,
            label=f"point mass  {pm['lap_time']:.3f} s")
    ax.plot(s, r26['v']*3.6, color='#5DCAA5', lw=1.5,
            label=f"four wheel  {r26['lap_time']:.3f} s")
    ax.set_ylabel('Speed (km/h)'); ax.set_xlabel('Lap distance (m)')
    ax.legend(fontsize=9, loc='lower right'); ax.grid(alpha=.2, lw=.5)
    ax.set_title('Cost of lateral load transfer', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{OUT}/2_load_transfer.png', dpi=150, facecolor='black')
    plt.close()

    # --- sweep ---
    if sweep:
        ks, lltds, grid, pm2 = sweep
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
        cols = ['#5DCAA5', '#378ADD', '#EF9F27', '#E24B4A']
        for i, (kk, c) in enumerate(zip(ks, cols)):
            ax[0].plot(lltds*100, grid[i], color=c, lw=1.8, marker='o', ms=4,
                       label=f'k={kk:.2f}')
            j = int(np.argmin(grid[i]))
            ax[0].plot(lltds[j]*100, grid[i, j], '*', color=c, ms=14)
        ax[0].axhline(pm2, color='#888780', ls='--', lw=.9, label='point mass')
        ax[0].set_xlabel('Elastic LLTD (% front)'); ax[0].set_ylabel('Lap time (s)')
        ax[0].set_title('Lap time vs LLTD', fontsize=11)
        ax[0].legend(fontsize=8); ax[0].grid(alpha=.25, lw=.5)
        ax[1].plot(ks, [grid[i].min()-pm2 for i in range(len(ks))],
                   color='#5DCAA5', lw=2, marker='o')
        ax[1].set_xlabel('Load sensitivity k')
        ax[1].set_ylabel('Cost of load transfer (s)')
        ax[1].set_title('Uncertainty band', fontsize=11)
        ax[1].grid(alpha=.25, lw=.5)
        plt.tight_layout()
        plt.savefig(f'{OUT}/3_lltd_sweep.png', dpi=150, facecolor='black')
        plt.close()

    # --- DP ---
    if dp is not None:
        n = len(dp.v)
        sd = np.linspace(0, TOTAL_LAP_M, n, endpoint=False)
        fig, ax = plt.subplots(3, 1, figsize=(15, 9), sharex=True,
                               gridspec_kw={'height_ratios': [2.2, 1.6, 1.4]})
        ax[0].plot(sd, dp.v*3.6, color='#5DCAA5', lw=1.5)
        ax[0].set_ylabel('Speed (km/h)'); ax[0].grid(alpha=.2, lw=.5)
        ax[0].set_title(f'DP-optimised energy deployment — {dp.lap_time:.3f} s',
                        fontsize=12)
        ax[1].fill_between(sd, np.clip(dp.u_kW, 0, None), 0, color='#5DCAA5',
                           alpha=.75, label='deploy')
        ax[1].fill_between(sd, np.clip(dp.u_kW, None, 0), 0, color='#E24B4A',
                           alpha=.75, label='superclip / harvest')
        ax[1].axhline(0, color='#888780', lw=.6); ax[1].set_ylabel('MGU-K (kW)')
        ax[1].legend(fontsize=8, loc='upper right'); ax[1].grid(alpha=.2, lw=.5)
        ax[2].plot(sd, dp.soc_MJ, color='#EF9F27', lw=1.5)
        ax[2].axhline(4.0, color='#E24B4A', ls='--', lw=.8,
                      label='4 MJ window (C5.2.9)')
        ax[2].axhline(0, color='#E24B4A', ls='--', lw=.8)
        ax[2].set_ylabel('SoC (MJ)'); ax[2].set_xlabel('Lap distance (m)')
        ax[2].set_ylim(-.3, 4.4); ax[2].legend(fontsize=8, loc='lower right')
        ax[2].grid(alpha=.2, lw=.5)
        plt.tight_layout()
        plt.savefig(f'{OUT}/4_dp_deployment.png', dpi=150, facecolor='black')
        plt.close()

    print(f"\n  plots written to {OUT}/")


if __name__ == '__main__':
    main()
