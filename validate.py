"""
Validation against measured telemetry.

Objective
---------
Compare the simulator's velocity profile, sector splits and lap time against
real car telemetry from a 2026 session, on the same circuit, using the same
measured track geometry.

What counts as validation
-------------------------
Landing near a published pole time is NOT validation. A model with a dozen
estimated parameters can arrive at a plausible lap time for entirely wrong
reasons - an optimistic tyre coefficient cancelling a pessimistic aero map,
for instance. Agreement on a single scalar tells you almost nothing.

Validation means comparing DISTRIBUTED quantities that isolate different
parts of the model:

  1. Corner minimum speeds       -> tests grip and load transfer
  2. Straight-line terminal speed -> tests drag and deployment power
  3. Acceleration out of corners -> tests traction and the energy schedule
  4. Braking distances           -> tests combined slip and downforce
  5. Sector splits               -> tests the balance of the above

A model can match lap time while failing every one of these. Reporting them
separately is what makes the exercise diagnostic rather than decorative.

Interpreting disagreement
-------------------------
The simulator computes a THEORETICAL minimum lap time: no driver error, no
tyre thermal state, no fuel burn, a fixed racing line. Real telemetry
includes all of those. The simulator should therefore be FASTER than reality.
If it is slower, something in the model is wrong - that is a stronger signal
than closeness.

Usage
-----
    python validate.py --year 2026 --event Barcelona --session Q
"""

from __future__ import annotations
import argparse
import os
import sys
import numpy as np
G_STD = 9.80665


def align_to_centreline(s_cl, kappa, ref_s, ref_v, length_m):
    """
    Find the circular shift (m) between the telemetry Distance axis and the
    centreline arc length, by cross-correlation.

    The two axes have independent zeros: telemetry Distance starts wherever
    FastF1 slices the lap, the centreline starts at its first GPS sample. A
    30-80 m offset puts measured straight-line speeds on top of corners,
    which shows up as physically impossible lateral demand at every corner
    edge - a roughly constant fraction of the lap, insensitive to curvature
    smoothing. (Same failure class as a timing skew between two sensors;
    same cure.)

    Speed dips and |kappa| peaks must coincide, so correlate -v with |kappa|
    over all circular shifts and take the maximum. FFT-based, exact for the
    periodic lap.

    Returns (shift_m, ref_v_aligned_on_centreline_grid).
    """
    ref_scaled = ref_s * (length_m / ref_s.max())
    v_grid = np.interp(s_cl, ref_scaled, ref_v)

    a = -(v_grid - v_grid.mean())
    b = np.abs(kappa) - np.abs(kappa).mean()
    corr = np.fft.irfft(np.fft.rfft(a) * np.conj(np.fft.rfft(b)), n=len(a))
    shift_idx = int(np.argmax(corr))
    ds = length_m / len(s_cl)
    shift_m = shift_idx * ds
    if shift_m > length_m / 2:
        shift_m -= length_m

    v_aligned = np.roll(v_grid, -shift_idx)
    return shift_m, v_aligned


def find_centreline(event, year, session=None, search_dirs=None):
    """
    Locate a saved centreline for THIS event and year.

    Two failure modes this guards against, both observed:

    1. Cross-event fallback. An earlier version fell back to "any centreline
       for this year" when no exact match was found. A legacy Barcelona file
       was silently loaded for a Silverstone validation - 4599 m of the wrong
       circuit against a 5805 m lap, r = 0.33, and a nonsensical +1889 m
       alignment. The fallback now NEVER crosses events; if the event does
       not match, it returns None and the caller tells the user to build it.

    2. Event aliases. FastF1 resolves "Silverstone" to "British Grand Prix",
       so the saved filename shares no words with what the user typed.
       Common circuit-to-event aliases are resolved here.
    """
    import glob

    ALIASES = {
        'silverstone': 'british', 'monza': 'italian', 'spa': 'belgian',
        'suzuka': 'japanese', 'interlagos': 'brazilian', 'sao': 'brazilian',
        'zandvoort': 'dutch', 'imola': 'emilia', 'catalunya': 'barcelona',
        'spielberg': 'austrian', 'austria': 'austrian', 'baku': 'azerbaijan',
        'jeddah': 'saudi', 'yas': 'abu', 'cota': 'united', 'austin': 'united',
        'hungaroring': 'hungarian', 'monaco': 'monaco', 'miami': 'miami',
        'montreal': 'canadian', 'canada': 'canadian', 'mexico': 'mexico',
        'vegas': 'las', 'lusail': 'qatar', 'shanghai': 'chinese',
        'china': 'chinese', 'melbourne': 'australian',
    }

    if search_dirs is None:
        home = os.path.join(os.path.expanduser('~'), 'f1sim_data', 'telemetry')
        search_dirs = (home, 'telemetry', 'data', '.')

    raw = str(event).split()[0].lower()
    keys = {raw}
    if raw in ALIASES:
        keys.add(ALIASES[raw])
    want_session = str(session).lower() if session else None

    candidates = []
    for d in search_dirs:
        for f in glob.glob(os.path.join(d, 'centreline_*.npz')):
            stem = f[:-4]
            name = os.path.basename(stem).lower()
            if str(year) not in name:
                continue
            if not any(k in name for k in keys):
                continue            # NEVER fall back across events
            has_session = want_session and name.endswith('_' + want_session)
            candidates.append((0 if has_session else 1, stem))

    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


def load_reference_lap(year, event, session='Q', cache_dir=None):
    """
    Fetch the fastest lap of a session with speed against lap distance.

    Returns dict with s (m), v (m/s), lap_time (s), driver, sectors.
    """
    import fastf1

    cache_dir = cache_dir or os.path.join(os.path.expanduser('~'),
                                          'f1sim_data', 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)

    ses = fastf1.get_session(year, event, session)
    ses.load(telemetry=True, laps=True, weather=False)

    lap = ses.laps.pick_fastest()
    tel = lap.get_car_data().add_distance()

    sectors = {}
    for i, key in enumerate(['Sector1Time', 'Sector2Time', 'Sector3Time'], 1):
        val = lap.get(key)
        if val is not None and not isinstance(val, float):
            try:
                sectors[f'S{i}'] = val.total_seconds()
            except AttributeError:
                pass

    return {
        's': tel['Distance'].to_numpy(),
        'v': tel['Speed'].to_numpy() / 3.6,       # km/h -> m/s
        'throttle': tel['Throttle'].to_numpy() if 'Throttle' in tel else None,
        'brake': tel['Brake'].to_numpy() if 'Brake' in tel else None,
        'lap_time': lap['LapTime'].total_seconds(),
        'driver': str(lap['Driver']),
        'sectors': sectors,
        'circuit': str(ses.event['EventName']),
    }


# ------------------------------------------------------------ comparisons

def find_corner_minima(s, v, kappa=None, depth_frac=0.06, min_gap_m=60.0):
    """
    Locate corner speed minima.

    Local-extremum tests fail on this signal because a QSS profile holds a
    CONSTANT speed through the apex - the corner floor is a plateau, not a
    point, so any fixed comparison window that does not clear the plateau
    finds nothing.

    Instead: identify contiguous regions where speed drops below a rolling
    baseline by at least depth_frac, then take the minimum of each region.
    This handles both flat simulated floors and rounded measured ones.

    Args:
        depth_frac : depth below the rolling baseline that defines a corner
        min_gap_m  : merge minima closer than this

    Returns (s_at_minima, v_at_minima).
    """
    s = np.asarray(s, dtype=float)
    v = np.asarray(v, dtype=float)
    n = len(v)
    if n < 20:
        return np.array([]), np.array([])

    # Rolling baseline, wide enough to span a corner but not a whole sector.
    span_m = 400.0
    w = max(5, int(span_m / max(np.median(np.diff(s)), 1.0)))
    pad = np.concatenate([v[-w:], v, v[:w]])            # wrap: lap is closed
    kernel = np.ones(2 * w + 1) / (2 * w + 1)
    baseline = np.convolve(pad, kernel, mode='same')[w:w + n]

    below = v < baseline * (1.0 - depth_frac)
    if not below.any():
        return np.array([]), np.array([])

    # Group contiguous below-baseline regions, wrapping the lap.
    idx = np.where(below)[0]
    groups, cur = [], [idx[0]]
    for i in idx[1:]:
        if i == cur[-1] + 1:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    if len(groups) > 1 and groups[0][0] == 0 and groups[-1][-1] == n - 1:
        groups[0] = groups[-1] + groups[0]               # wrap start/finish
        groups.pop()

    mins = []
    for g in groups:
        j = g[int(np.argmin(v[g]))]
        mins.append(j)
    mins.sort()

    merged = [mins[0]]
    for j in mins[1:]:
        if s[j] - s[merged[-1]] < min_gap_m:
            if v[j] < v[merged[-1]]:
                merged[-1] = j
        else:
            merged.append(j)

    out = np.array(merged, dtype=int)
    return s[out], v[out]


def compare_profiles(sim_s, sim_v, ref_s, ref_v):
    """
    Resample both onto a common grid and compute agreement statistics.

    Returns dict of metrics. RMS and mean bias are in m/s; the correlation
    is on the speed trace itself.
    """
    lo = max(sim_s.min(), ref_s.min())
    hi = min(sim_s.max(), ref_s.max())
    grid = np.arange(lo, hi, 5.0)

    a = np.interp(grid, sim_s, sim_v)
    b = np.interp(grid, ref_s, ref_v)
    d = a - b

    return {
        'grid': grid, 'sim': a, 'ref': b, 'delta': d,
        'rms_ms': float(np.sqrt(np.mean(d ** 2))),
        'mean_bias_ms': float(np.mean(d)),
        'max_abs_ms': float(np.max(np.abs(d))),
        'correlation': float(np.corrcoef(a, b)[0, 1]),
        'sim_faster_frac': float(np.mean(d > 0)),
    }


def match_corner_speeds(sim_s, sim_v, ref_s, ref_v, tol_m=60.0):
    """
    Pair simulated and measured corner minima by proximity in lap distance.

    This is the single most diagnostic comparison available: corner minimum
    speed is set almost entirely by grip and lateral load transfer, with very
    little contribution from the powertrain or the energy schedule.
    """
    ss, sv = find_corner_minima(sim_s, sim_v)
    rs, rv = find_corner_minima(ref_s, ref_v)

    pairs = []
    for s_i, v_i in zip(ss, sv):
        if len(rs) == 0:
            break
        j = int(np.argmin(np.abs(rs - s_i)))
        if abs(rs[j] - s_i) <= tol_m:
            pairs.append((s_i, v_i, rv[j]))

    if not pairs:
        return {'n': 0}

    arr = np.array(pairs)
    err = arr[:, 1] - arr[:, 2]
    return {
        'n': len(pairs), 'pairs': arr,
        'rms_ms': float(np.sqrt(np.mean(err ** 2))),
        'mean_bias_ms': float(np.mean(err)),
        'mean_pct': float(np.mean(err / arr[:, 2]) * 100.0),
    }


def energy_audit(sim_result, s, car, cfg=None):
    """
    How much electrical energy would this lap actually need?

    validate.py solves with solve_lap, which deploys the MGU-K at the
    regulatory ceiling wherever the car is accelerating and does NOT track
    state of charge. That is an energy-UNCONSTRAINED lap. The audit below
    integrates the deployment it implies and compares it with the two limits
    that bind a real car:

        C5.2.9  energy store window   4 MJ
        C5.2.10 recharge per lap      8.5 MJ

    If the required deployment exceeds what can be harvested and stored, the
    simulated lap is not merely optimistic - it is infeasible, and the excess
    is concentrated on long straights where the car would in reality clip.
    """
    from energy.regs_2026 import deployment_limit_kw

    v = sim_result['v']
    a = sim_result['a']
    ds = float(np.median(np.diff(s))) if len(s) > 1 else 10.0

    E_dep = 0.0
    E_regen = 0.0
    for i in range(len(v)):
        vi = max(float(v[i]), 1.0)
        dt = ds / vi
        if a[i] > 0:
            # accelerating: MGU-K assumed at the ceiling, as solve_lap does
            P = deployment_limit_kw(vi, overtake_active=True) * 1e3
            E_dep += P * dt
        elif a[i] < 0:
            # braking: harvest bounded by the 350 kW absolute limit (C5.2.7)
            P = min(350e3, abs(a[i]) * car['m_kg'] * vi * 0.45)
            E_regen += P * dt
    return E_dep, E_regen


def report(sim_result, ref, sim_s=None, car=None):
    """Print a validation report."""
    sim_v = sim_result['v']
    if sim_s is None:
        sim_s = sim_result.get('s')

    print("=" * 62)
    print(f"  VALIDATION  {ref['circuit']}  vs  {ref['driver']}")
    print("=" * 62)

    print("\n  LAP TIME")
    print(f"    simulated : {sim_result['lap_time']:8.3f} s")
    print(f"    measured  : {ref['lap_time']:8.3f} s")
    d = sim_result['lap_time'] - ref['lap_time']
    print(f"    delta     : {d:+8.3f} s  ({100*d/ref['lap_time']:+.2f} %)")
    if d < 0:
        print("    -> simulator faster, as expected for a theoretical minimum lap")
    else:
        print("    -> SIMULATOR SLOWER THAN REALITY. A minimum-lap-time model")
        print("       should not be. Investigate before trusting any result.")

    prof = compare_profiles(sim_s, sim_v, ref['s'], ref['v'])
    print("\n  VELOCITY PROFILE")
    print(f"    RMS error      : {prof['rms_ms']*3.6:6.2f} km/h")
    print(f"    mean bias      : {prof['mean_bias_ms']*3.6:+6.2f} km/h")
    print(f"    max abs error  : {prof['max_abs_ms']*3.6:6.2f} km/h")
    print(f"    correlation    : {prof['correlation']:6.4f}")
    print(f"    sim faster for : {100*prof['sim_faster_frac']:5.1f} % of the lap")

    cs = match_corner_speeds(sim_s, sim_v, ref['s'], ref['v'])
    print("\n  CORNER MINIMUM SPEEDS  (tests grip and load transfer)")
    if cs['n'] == 0:
        print("    no corners paired")
    else:
        print(f"    corners paired : {cs['n']}")
        print(f"    RMS error      : {cs['rms_ms']*3.6:6.2f} km/h")
        print(f"    mean bias      : {cs['mean_bias_ms']*3.6:+6.2f} km/h "
              f"({cs['mean_pct']:+.1f} %)")
        if abs(cs['mean_pct']) > 8:
            hint = "mu0 too high" if cs['mean_pct'] > 0 else "mu0 too low"
            print(f"    -> systematic bias suggests {hint}, or an aero/CoG error")

    print("\n  TERMINAL SPEED  (tests drag and deployment)")
    print(f"    simulated : {sim_v.max()*3.6:6.1f} km/h")
    print(f"    measured  : {ref['v'].max()*3.6:6.1f} km/h")
    print(f"    delta     : {(sim_v.max()-ref['v'].max())*3.6:+6.1f} km/h")

    if ref['sectors']:
        print("\n  SECTOR SPLITS (measured)")
        for k, v in ref['sectors'].items():
            print(f"    {k} : {v:7.3f} s")
        print("    note: FIA sector boundaries are not published as lap")
        print("    distances, so simulated splits are not directly comparable.")
        print("    Use the velocity profile and corner speeds instead.")

    if car is not None and sim_s is not None:
        E_dep, E_regen = energy_audit(sim_result, sim_s, car)
        print("\n  ENERGY AUDIT  (this lap is solved UNCONSTRAINED)")
        print(f"    deployment implied : {E_dep/1e6:6.2f} MJ")
        print(f"    harvest available  : {E_regen/1e6:6.2f} MJ")
        print(f"    store window       :   4.00 MJ   (C5.2.9)")
        print(f"    recharge cap       :   8.50 MJ/lap (C5.2.10)")
        net = (E_dep - E_regen) / 1e6
        if net > 4.0:
            print(f"    -> INFEASIBLE: needs {net:.2f} MJ more than it harvests,")
            print(f"       against a 4 MJ store. The car would clip on the long")
            print(f"       straights; this lap time is optimistic by that amount.")
            print(f"       Run the DP optimiser for the legal lap.")
        else:
            print(f"    -> net {net:+.2f} MJ, within the store window")

    print("\n" + "=" * 62)
    return {'profile': prof, 'corners': cs, 'lap_delta': d}


# ------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--year', type=int, default=2026)
    ap.add_argument('--event', default='Barcelona')
    ap.add_argument('--session', default='Q')
    ap.add_argument('--centreline', default=None,
                    help='path stem of a saved centreline; auto-discovered '
                         'in telemetry/ if omitted')
    ap.add_argument('--geometry-session', default='Q',
                    help="session whose centreline supplies the GEOMETRY "
                         "(default Q - cleanest lines). Independent of "
                         "--session, which selects the reference lap. This "
                         "is what makes out-of-sample validation clean: "
                         "geometry fixed, reference lap swapped.")
    ap.add_argument('--ds', type=float, default=10.0)
    ap.add_argument('--plot', action='store_true')
    ap.add_argument('--energy', action='store_true',
                    help='solve the LEGAL lap with the DP energy optimiser '
                         '(qualifying: store full at the line, empty at the '
                         'flag) instead of the energy-unconstrained solver. '
                         'Slower (~30 s) but this is the lap a real car can '
                         'actually drive.')
    ap.add_argument('--ds-energy', type=float, default=15.0,
                    help='track discretisation for the DP (coarser = faster)')
    ap.add_argument('--dp-res', type=int, default=51,
                    help='SoC grid resolution for the DP. The default was '
                         'raised from 41 after a coarse grid left 0.55 MJ '
                         'unspent at the flag despite a heavy penalty - the '
                         'control quantisation simply could not place the '
                         'deployment. Numerical, not physical.')
    ap.add_argument('--dp-ctrl', type=int, default=11,
                    help='control grid resolution for the DP')
    args = ap.parse_args()

    import sys
    sys.path.insert(0, '.')
    from data.fastf1_track import Centreline
    from data.vehicle_params import CAR_2026, _sync_aero
    from solver_v2 import solve_lap
    from vehicle.four_wheel import FourWheelModel
    from vehicle.f1_2026_params import GEO_2026, AERO_2026, TYRE_2026
    from energy.regs_2026 import deployment_limit_kw

    print("loading measured centreline...")
    path = args.centreline or find_centreline(args.event, args.year,
                                              args.geometry_session)
    if path is None:
        print(f"\nNo centreline found for {args.event} {args.year}. Build it:")
        print(f"  python extract_telemetry.py --year {args.year} "
              f"--event {args.event} --session {args.session}")
        sys.exit(1)
    print(f"  {path}")
    cl = Centreline.load(path)
    s, kappa = cl.resample(args.ds)
    grade = np.zeros(len(s))          # FastF1 provides no elevation channel
    win = cl.metadata.get('curvature_filter_window_m')
    print(f"  {cl.circuit}: {cl.length_m:.1f} m, {cl.n_laps_used} laps averaged")
    if win is not None:
        from data.fastf1_track import DEFAULT_SMOOTH_M
        print(f"  curvature filter window: {win:.0f} m")
        if abs(win - DEFAULT_SMOOTH_M) > 1e-6:
            print(f"  WARNING: this centreline was built at {win:.0f} m but the")
            print(f"           canonical window is {DEFAULT_SMOOTH_M:.0f} m. Lap time is")
            print(f"           sensitive to this (~3 s across 30-55 m), so results")
            print(f"           are NOT comparable with runs at the canonical value.")
            print(f"           Rebuild: python extract_telemetry.py ... (no --smooth)")

    _sync_aero()   # keep CAR_* dicts consistent with the AeroModel objects
    print("solving lap...")
    fw = FourWheelModel(GEO_2026, AERO_2026, TYRE_2026)

    def mguk(v, y=2026):
        # Qualifying is an LTCS; Overtake is active at all times (B7.2.3(b))
        return deployment_limit_kw(v, overtake_active=True)

    sim = solve_lap(s, kappa, grade, CAR_2026, mguk, 2026, args.ds,
                    'four_wheel', fw_model=fw)

    if args.energy:
        from energy.deployment_dp import optimise_deployment
        from energy.regs_2026 import EnergyConfig, Session, BARCELONA_ZONES,\
            ActivationZones
        from solver_v2 import corner_speed_four_wheel

        print("solving the LEGAL lap with the DP energy optimiser...")
        ds_e = max(args.ds, args.ds_energy)
        s_e, kap_e = cl.resample(ds_e)
        grade_e = np.zeros(len(s_e))

        # Activation zones are per-circuit and published by the FIA; only
        # Barcelona is encoded. Elsewhere, fall back to Corner Mode
        # throughout, which understates straight-line speed and is therefore
        # conservative.
        zones = (BARCELONA_ZONES if 'barcelona' in cl.circuit.lower()
                 else ActivationZones(zones=()))
        sm_e = zones.mask(s_e)
        if not zones.zones:
            print("  note: no activation zones encoded for this circuit;")
            print("        running Corner Mode throughout (conservative).")

        vc = np.array([corner_speed_four_wheel(kk, fw, gg, bool(mm))
                       if abs(kk) > 1e-9 else 120.0
                       for kk, gg, mm in zip(kap_e, grade_e, sm_e)])
        vc = np.minimum(vc, 120.0)

        cfg = EnergyConfig(session=Session.LTCS, terminal_mode='deplete')
        dp = optimise_deployment(s_e, kap_e, grade_e, vc, CAR_2026, cfg,
                                 sm_e, ds_e, n_soc=args.dp_res,
                                 n_ctrl=args.dp_ctrl)
        print(f"  unconstrained : {sim['lap_time']:7.3f} s  (infeasible)")
        print(f"  legal lap     : {dp.lap_time:7.3f} s  | deployed "
              f"{dp.energy_deployed_MJ:.2f} MJ, harvested "
              f"{dp.energy_harvested_MJ:.2f} MJ, SoC "
              f"{dp.soc_MJ[0]:.2f} -> {dp.soc_MJ[-1]:.2f}")
        # ── REFINE: the DP's job is the ENERGY SCHEDULE; the QSS solver's
        # job is the TRACE. The DP runs on a coarse speed grid (~2 m/s
        # cells), which snaps corner minima down, never presses the drag
        # wall, and integrates lap time on quantised speeds - comparing
        # that trace to telemetry mixes solver quantisation into the
        # validation. So the DP's deployment schedule u(s) is mapped onto
        # the fine grid and the three-pass QSS profile is re-solved under
        # it: legality from the DP, dynamic fidelity from the QSS.
        from energy.deployment_dp import _drag_N, _downforce_N
        from energy.regs_2026 import deployment_limit_kw as _dl
        import math as _m

        u_fine = np.interp(s, s_e, dp.u_kW) * 1e3          # W, +ve deploy
        sm_f = zones.mask(s)
        m_kg = CAR_2026['m_kg']
        P_ice = cfg.ice_power_kW * 1e3
        vcap_f = np.array([corner_speed_four_wheel(kk, fw, 0.0, bool(mm))
                           if abs(kk) > 1e-9 else 120.0
                           for kk, mm in zip(kappa, sm_f)])
        vcap_f = np.minimum(vcap_f, 120.0)
        Nf = len(s)
        E0 = cfg.delta_soc_max_MJ * 1e6
        harv_cap = cfg.recharge_max_MJ * 1e6

        def run_refined(scale):
            """
            Three-pass QSS profile under the DP schedule scaled by `scale`,
            with a per-sample SoC ledger. The DP supplies the SHAPE of the
            schedule (where energy is worth most); the scale is bisected so
            the store actually EMPTIES at the flag - a flying lap crossing
            the line with charge left has thrown that time away, and the
            unscaled schedule (built by a deliberately conservative DP)
            left 2.8 MJ unspent. Deployment per sample is still capped by
            the C5.2.8 profile and by what the ledger holds.
            """
            v_start = float(min(vcap_f[0], 120.0))
            for _ in range(12):
                vb = vcap_f.copy()
                vb[-1] = min(vb[-1], v_start)
                for i in range(Nf - 2, -1, -1):
                    vi = max(vb[i + 1], 1.0)
                    ab = fw.max_longitudinal_acceleration(
                        vi, kappa[i], 0.0, bool(sm_f[i]), braking=True)
                    ab += _drag_N(vi, CAR_2026, bool(sm_f[i])) / m_kg
                    vb[i] = min(vb[i],
                                _m.sqrt(vb[i + 1] ** 2 + 2 * ab * args.ds))
                vb[0] = min(vb[0], v_start)
                vv = vb.copy(); vv[0] = min(vv[0], v_start)
                E_led = E0; harv = 0.0
                for i in range(Nf - 1):
                    vi = max(vv[i], 1.0)
                    dt_i = args.ds / vi
                    want = max(u_fine[i], 0.0) * scale
                    u_i = min(want, _dl(vi, True) * 1e3,
                              E_led / max(dt_i, 1e-6))
                    E_led = max(E_led - u_i * dt_i, 0.0)
                    if vb[i + 1] < vi:
                        a_b = (vb[i + 1] ** 2 - vi * vi) / (2 * args.ds)
                        P_h = min(350e3, 0.45 * m_kg * max(-a_b, 0.) * vi)
                        P_h = min(P_h, (harv_cap - harv) / max(dt_i, 1e-6))
                        add = min(P_h * dt_i, E0 - E_led)
                        E_led += add; harv += add
                    F = (P_ice + u_i) / vi
                    F = min(F, fw.max_longitudinal_acceleration(
                        vi, kappa[i], 0.0, bool(sm_f[i]), braking=False)
                        * m_kg)
                    F -= _drag_N(vi, CAR_2026, bool(sm_f[i]))
                    F -= CAR_2026['Crr'] * (m_kg * G_STD
                                            + _downforce_N(vi, CAR_2026,
                                                           bool(sm_f[i])))
                    v2 = vi * vi + 2.0 * (F / m_kg) * args.ds
                    vv[i + 1] = min(vv[i + 1], _m.sqrt(max(v2, 1.0)))
                if abs(vv[-1] - v_start) < 0.05:
                    v_start = vv[-1]; break
                v_start = vv[-1]
            dtf = args.ds / np.maximum(0.5 * (vv + np.roll(vv, -1)), 1.0)
            dep = E0 + harv - E_led
            return vv, float(np.sum(dtf)), E_led, dep, harv

        lo_s, hi_s = 1.0, 8.0
        vv, lap_f, E_end, dep_J, harv_J = run_refined(1.0)
        if E_end > 0.15e6:
            for _b in range(10):
                mid = 0.5 * (lo_s + hi_s)
                vv, lap_f, E_end, dep_J, harv_J = run_refined(mid)
                if E_end > 0.15e6:
                    lo_s = mid
                else:
                    hi_s = mid
        print(f"  refined trace : {lap_f:7.3f} s  (QSS under scaled DP "
              f"schedule: deployed {dep_J/1e6:.2f} MJ, harvested "
              f"{harv_J/1e6:.2f}, SoC end {E_end/1e6:.2f})")

        sim = {'v': vv, 'lap_time': lap_f,
               'a': np.gradient(vv ** 2) / (2 * args.ds),
               't': np.zeros(Nf)}

    print("fetching reference telemetry...")
    ref = load_reference_lap(args.year, args.event, args.session)

    # Align the telemetry Distance axis to the centreline arc length before
    # any comparison - their zeros are independent.
    s_grid, kap_grid = cl.resample(args.ds)
    shift_m, v_al = align_to_centreline(s_grid, kap_grid,
                                        ref['s'], ref['v'], cl.length_m)
    print(f"  distance-axis offset found: {shift_m:+.0f} m (corrected)")
    ref = dict(ref)
    ref['s'] = s_grid
    ref['v'] = v_al

    res = report(sim, ref, sim_s=s,
                 car=None if args.energy else CAR_2026)

    if args.plot:
        import matplotlib.pyplot as plt
        plt.style.use('dark_background')
        p = res['profile']
        fig, ax = plt.subplots(2, 1, figsize=(15, 7), sharex=True,
                               gridspec_kw={'height_ratios': [3, 1.4]})
        ax[0].plot(p['grid'], p['ref'] * 3.6, color='#EF9F27', lw=1.4,
                   label=f"measured — {ref['driver']}  {ref['lap_time']:.3f} s")
        ax[0].plot(p['grid'], p['sim'] * 3.6, color='#5DCAA5', lw=1.4,
                   label=f"simulated  {sim['lap_time']:.3f} s")
        ax[0].set_ylabel('Speed (km/h)')
        ax[0].legend(fontsize=9, loc='lower right')
        ax[0].grid(alpha=.2, lw=.5)
        ax[0].set_title(f"{ref['circuit']} — simulation vs measured telemetry\n"
                        f"RMS {p['rms_ms']*3.6:.1f} km/h, r = {p['correlation']:.4f}",
                        fontsize=12)
        d = p['delta'] * 3.6
        ax[1].fill_between(p['grid'], d, 0, where=d >= 0, color='#5DCAA5', alpha=.65)
        ax[1].fill_between(p['grid'], d, 0, where=d < 0, color='#E24B4A', alpha=.65)
        ax[1].axhline(0, color='#888780', lw=.6)
        ax[1].set_ylabel('Δv (km/h)\nsim − measured')
        ax[1].set_xlabel('Lap distance (m)')
        ax[1].grid(alpha=.2, lw=.5)
        plt.tight_layout()
        os.makedirs('outputs', exist_ok=True)
        plt.savefig('outputs/validation_vs_telemetry.png', dpi=150,
                    facecolor='black')
        print("plot saved: outputs/validation_vs_telemetry.png")


if __name__ == '__main__':
    main()
