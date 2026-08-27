#!/usr/bin/env python3
"""
Extract F1 telemetry and track geometry from FastF1.

Run this on a machine with internet access. It writes CSV and NPZ files that
the simulator and validation harness read, so nothing downstream needs network
access afterwards.

What it pulls
-------------
  speed / throttle / brake / gear / rpm / DRS   vs lap distance
  X / Y position                                for centreline reconstruction
  lap and sector times

Usage
-----
    # everything for one session
    python extract_telemetry.py --year 2026 --event Barcelona --session Q

    # a specific driver instead of the session's fastest lap
    python extract_telemetry.py --year 2026 --event Barcelona --driver VER

    # list what sessions are available
    python extract_telemetry.py --list --year 2026

Outputs (into --outdir, default 'telemetry/')
    <event>_<session>_fastest.csv        channel data vs distance
    <event>_<session>_position.csv       X/Y/Z position samples
    <event>_<session>_laps.csv           all laps with times and compounds
    centreline_<event>_<year>.npz/.json  measured curvature for the simulator

Note on 2026: the calendar has BOTH a Barcelona Grand Prix (14 June) and a
Spanish Grand Prix at Madrid (13 September). Use 'Barcelona' for the Circuit
de Barcelona-Catalunya, not 'Spain'.
"""

from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import pandas as pd

# Stable data home, OUTSIDE the project folder, so cache and telemetry
# survive re-downloading or moving the project.
DATA_HOME = os.path.join(os.path.expanduser('~'), 'f1sim_data')


def setup_cache(cache_dir=None):
    """FastF1 cache in a stable location (~/f1sim_data/cache) unless overridden."""
    import fastf1
    cache_dir = cache_dir or os.path.join(DATA_HOME, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)
    return fastf1


def list_events(year: int):
    fastf1 = setup_cache()
    sch = fastf1.get_event_schedule(year, include_testing=False)
    today = pd.Timestamp.now()
    print(f"\n{year} calendar\n" + "-" * 52)
    for _, r in sch.iterrows():
        held = "held " if r['EventDate'] < today else "upcoming"
        print(f"  {held}  {r['EventDate'].date()}  {r['EventName']}")
    print("\nSessions: FP1 FP2 FP3 Q SQ S R")


def extract_session(year, event, session='Q', driver=None,
                    outdir=None, cache_dir=None):
    """Pull channel data, position data and lap times for one session."""
    fastf1 = setup_cache(cache_dir)
    outdir = outdir or os.path.join(DATA_HOME, 'telemetry')
    os.makedirs(outdir, exist_ok=True)

    print(f"loading {year} {event} {session} ...")
    ses = fastf1.get_session(year, event, session)
    ses.load(telemetry=True, laps=True, weather=True)

    name = str(ses.event['EventName']).replace(' ', '_')
    stem = os.path.join(outdir, f"{name}_{session}")

    # ---- all laps ------------------------------------------------------
    laps = ses.laps
    if len(laps) == 0:
        raise RuntimeError("no lap data in this session")

    cols = [c for c in ['Driver', 'LapNumber', 'LapTime', 'Sector1Time',
                        'Sector2Time', 'Sector3Time', 'Compound', 'TyreLife',
                        'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST',
                        'IsAccurate'] if c in laps.columns]
    laps[cols].to_csv(f"{stem}_laps.csv", index=False)
    print(f"  laps            -> {stem}_laps.csv  ({len(laps)} laps)")

    # ---- reference lap -------------------------------------------------
    if driver:
        dl = laps.pick_drivers(driver)
        if len(dl) == 0:
            raise RuntimeError(f"no laps for driver {driver}")
        lap = dl.pick_fastest()
    else:
        lap = laps.pick_fastest()

    lt = lap['LapTime'].total_seconds()
    print(f"  reference lap   : {lap['Driver']} {lt:.3f} s")

    car = lap.get_car_data().add_distance()
    keep = [c for c in ['Distance', 'Speed', 'Throttle', 'Brake', 'nGear',
                        'RPM', 'DRS', 'Time'] if c in car.columns]
    car[keep].to_csv(f"{stem}_fastest.csv", index=False)
    print(f"  channels        -> {stem}_fastest.csv  ({len(car)} samples)")

    pos = lap.get_pos_data()
    pkeep = [c for c in ['X', 'Y', 'Z', 'Time'] if c in pos.columns]
    pos[pkeep].to_csv(f"{stem}_position.csv", index=False)
    print(f"  position        -> {stem}_position.csv  ({len(pos)} samples)")

    # ---- summary -------------------------------------------------------
    print(f"\n  lap time        : {lt:.3f} s")
    for i, k in enumerate(['Sector1Time', 'Sector2Time', 'Sector3Time'], 1):
        val = lap.get(k)
        if val is not None and hasattr(val, 'total_seconds'):
            print(f"  sector {i}        : {val.total_seconds():.3f} s")
    print(f"  top speed       : {car['Speed'].max():.1f} km/h")
    print(f"  min speed       : {car['Speed'].min():.1f} km/h")
    if 'Compound' in lap:
        print(f"  compound        : {lap['Compound']}")

    return ses, lap


def build_centreline_from_session(year, event, session='Q', n_laps=10,
                                  n_samples=2000, smooth=None,
                                  outdir=None, cache_dir=None):
    """
    Reconstruct a measured centreline with curvature.

    Averages the fastest clean laps, resamples to uniform arc length, fits a
    periodic spline and differentiates it analytically. See
    data/fastf1_track.py for why finite differencing of raw GPS is unusable.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data.fastf1_track import build_centreline

    outdir = outdir or os.path.join(DATA_HOME, 'telemetry')
    os.makedirs(outdir, exist_ok=True)
    cl = build_centreline(year, event, session, n_laps, n_samples, smooth,
                          cache_dir)
    name = cl.circuit.replace(' ', '_')
    # Session in the filename: centrelines built from different sessions are
    # DIFFERENT GEOMETRY ESTIMATES (FP3 laps are scrappier, with more line
    # variation reading as spurious curvature). Overwriting one with another
    # silently changed the simulator's own lap time by 2.6 s and confounded
    # an out-of-sample validation. Qualifying gives the cleanest lines and is
    # the geometry source of record.
    path = os.path.join(outdir, f"centreline_{name}_{year}_{session}")
    cl.save(path)

    R = np.where(np.abs(cl.kappa) > 1e-4,
                 1.0 / np.maximum(np.abs(cl.kappa), 1e-9), np.inf)
    finite = R[np.isfinite(R)]
    print(f"\n  centreline      -> {path}.npz / .json")
    print(f"  laps averaged   : {cl.n_laps_used}")
    print(f"  lap length      : {cl.length_m:.1f} m")
    print(f"  kappa range     : {cl.kappa.min():+.5f} to {cl.kappa.max():+.5f} 1/m")
    if len(finite):
        Rmin = finite.min()
        print(f"  tightest corner : R = {Rmin:.1f} m")
        from data.fastf1_track import DEFAULT_SMOOTH_M
        print(f"  filter window   : {smooth or DEFAULT_SMOOTH_M:.0f} m")
        if Rmin < 25.0:
            print("  WARNING: an F1 racing line rarely goes below ~40 m radius.")
            print("           A tighter value usually means the curvature filter")
            print("           window is too short and is following GPS noise.")
            print("           Try --smooth 40 and compare.")
    print(f"  left / right    : {(cl.kappa > 1e-4).sum()} / "
          f"{(cl.kappa < -1e-4).sum()} samples")
    return cl


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--year', type=int, default=2026)
    ap.add_argument('--event', default='Barcelona',
                    help="event name; use 'Barcelona' not 'Spain' for Catalunya")
    ap.add_argument('--session', default='Q', help='FP1 FP2 FP3 Q SQ S R')
    ap.add_argument('--driver', default=None, help='e.g. VER, NOR')
    ap.add_argument('--laps', type=int, default=10,
                    help='laps to average for the centreline')
    ap.add_argument('--samples', type=int, default=2000)
    ap.add_argument('--smooth', type=float, default=None,
                    help='curvature filter window in METRES (default 30); '
                         'larger suppresses noise but rounds off tight corners')
    ap.add_argument('--outdir', default=None,
                    help='defaults to ~/f1sim_data/telemetry (stable across project folders)')
    ap.add_argument('--list', action='store_true', help='list the calendar')
    ap.add_argument('--no-centreline', action='store_true')
    args = ap.parse_args()

    try:
        import fastf1  # noqa: F401
    except ImportError:
        print("fastf1 not installed.  pip install fastf1")
        sys.exit(1)

    if args.outdir is None:
        args.outdir = os.path.join(DATA_HOME, 'telemetry')

    if args.list:
        list_events(args.year)
        return

    try:
        extract_session(args.year, args.event, args.session, args.driver,
                        args.outdir)
        if not args.no_centreline:
            build_centreline_from_session(args.year, args.event, args.session,
                                          args.laps, args.samples,
                                          args.smooth, args.outdir)
        print("\ndone. next:")
        print("  python validate.py --year %d --event %s --session %s --plot"
              % (args.year, args.event, args.session))
    except Exception as e:
        print(f"\nFAILED: {type(e).__name__}: {e}")
        print("\nCommon causes:")
        print("  - no network access to livetiming.formula1.com")
        print("  - session has not happened yet (check --list)")
        print("  - wrong event name (Barcelona vs Spain in 2026)")
        sys.exit(1)


if __name__ == '__main__':
    main()
