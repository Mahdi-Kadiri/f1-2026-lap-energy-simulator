"""
Barcelona (Circuit de Barcelona-Catalunya) track geometry.

SIGN CONVENTION (SAE, Z up):
    kappa = 1/R,  positive = LEFT turn, negative = RIGHT turn.
    Sign is load-bearing for any model with axles: it determines which
    side of the car the lateral load transfers to. Do not strip it.

Source: published circuit map / racing-line geometry, cross-referenced
        with FIA circuit data. ESTIMATED — see limitations.

Limitations:
- Curvatures are estimates from published racing-line geometry, not
  measured. Replace with FastF1 GPS-derived curvature for validation work.
- Gradients from circuit elevation profile, max error ~0.3%.
- Track width, kerbs and banking are not modelled.
- Corner directions follow official FIA Barcelona numbering.
"""

import numpy as np

INF = 1e6  # straight

# (s_start_m, length_m, signed_radius_m, gradient, sector, label)
#   signed radius: +ve = LEFT, -ve = RIGHT, INF = straight
_SEGMENTS_RAW = [
    # --- Sector 1 ---
    (0,     150,  INF,   +0.010, 1, "Start/finish straight"),
    (150,   80,  -115,   +0.008, 1, "T1 braking (right)"),
    (230,   60,  -115,   +0.006, 1, "T1 apex (right)"),
    (290,   90,  INF,    +0.004, 1, "T1-T2 link"),
    (380,   70,   -80,   +0.002, 1, "T2 (right)"),
    (450,   60,   -80,    0.000, 1, "T2 apex (right)"),
    (510,  120,  INF,    -0.005, 1, "T2-T3 short straight"),
    (630,   80,  -100,   -0.010, 1, "T3 (right)"),
    (710,   60,  -100,   -0.012, 1, "T3 apex (right)"),
    (770,  180,  INF,    -0.015, 1, "T3-T4 straight"),
    (950,   90,  -110,   -0.010, 1, "T4 (right)"),
    (1040,  70,  -110,   -0.008, 1, "T4 apex (right)"),
    # --- Sector 2 ---
    (1110, 200,  INF,    -0.005, 2, "T4-T5 straight"),
    (1310,  80,  INF,     0.000, 2, "T5 approach"),
    (1390,  70,  -400,   +0.005, 2, "T5 (right, high speed)"),
    (1460, 150,  INF,    +0.008, 2, "T5-T6 straight"),
    (1610,  90,   +70,   +0.010, 2, "T6 (left)"),
    (1700,  60,   +70,   +0.008, 2, "T6 apex (left)"),
    (1760,  80,  +200,   +0.005, 2, "T7 (left)"),
    (1840,  70,  +200,   +0.003, 2, "T7 apex (left)"),
    (1910, 130,  INF,     0.000, 2, "T7-T8 straight"),
    (2040, 100,   -90,   -0.005, 2, "T8 (right)"),
    (2140,  70,   -90,   -0.008, 2, "T8 apex (right)"),
    (2210,  80,  +250,   -0.010, 2, "T9 entry (left)"),
    (2290, 120,  +250,   -0.008, 2, "T9 apex (left)"),
    (2410,  80,  +250,   -0.005, 2, "T9 exit (left)"),
    # --- Sector 3 ---
    (2490, 100,  INF,    -0.003, 3, "T9-T10 straight"),
    (2590,  90,  -130,   +0.002, 3, "T10 (right)"),
    (2680,  80,  -130,   +0.005, 3, "T10 apex (right)"),
    (2760, 100,  INF,    +0.010, 3, "T10-T11 straight"),
    (2860,  80,  +120,   +0.015, 3, "T11 (left)"),
    (2940,  60,  +120,   +0.012, 3, "T11 apex (left)"),
    (3000,  90,  INF,    +0.008, 3, "T11-T12 link"),
    (3090,  70,   -90,   +0.005, 3, "T12 (right)"),
    (3160,  80,   -90,   +0.003, 3, "T12 apex (right)"),
    (3240, 100,  INF,     0.000, 3, "T12-T13 link"),
    (3340,  80,  +200,   -0.002, 3, "T13 (left)"),
    (3420,  60,  +200,   -0.003, 3, "T13 apex (left)"),
    (3480, 130,  INF,    -0.005, 3, "T13-T14 straight"),
    (3610,  70,  -160,   -0.008, 3, "T14 (right)"),
    (3680,  60,  -160,   -0.008, 3, "T14 apex (right)"),
    (3740, 100,  INF,    -0.005, 3, "T14-T15 straight"),
    (3840,  80,   -80,   -0.003, 3, "T15 (right)"),
    (3920,  70,   -80,    0.000, 3, "T15 apex (right)"),
    (3990,  90,  INF,    +0.005, 3, "T15 exit"),
    (4080, 160,  INF,    +0.008, 3, "Final straight"),
    (4240, 180,  INF,    +0.010, 3, "Pit straight run"),
    (4420, 237,  INF,    +0.010, 3, "Start/finish completion"),
]

TOTAL_LAP_M = 4657


def get_track_segments(ds=10.0):
    """
    Discretise the track at resolution ds (m).

    Returns:
        s      : (N,) cumulative distance (m)
        kappa  : (N,) SIGNED curvature (1/m), +ve = left
        grade  : (N,) longitudinal gradient (m/m)
        sector : (N,) sector number
    """
    s_out = np.arange(0.0, TOTAL_LAP_M, ds)
    N = len(s_out)
    kappa_out = np.zeros(N)
    grade_out = np.zeros(N)
    sector_out = np.ones(N, dtype=int)

    starts = np.array([seg[0] for seg in _SEGMENTS_RAW], dtype=float)
    for i, s in enumerate(s_out):
        j = int(np.searchsorted(starts, s, side='right') - 1)
        _, _, R, g, sec, _ = _SEGMENTS_RAW[j]
        kappa_out[i] = (1.0 / R) if abs(R) < INF else 0.0
        grade_out[i] = g
        sector_out[i] = sec

    return s_out, kappa_out, grade_out, sector_out


def get_segment_properties():
    rows = []
    for s0, L, R, g, sec, label in _SEGMENTS_RAW:
        straight = abs(R) >= INF
        rows.append({
            's_start_m': s0, 'length_m': L,
            'radius_m': None if straight else R,
            'kappa_1pm': 0.0 if straight else round(1.0 / R, 6),
            'direction': 'straight' if straight else ('left' if R > 0 else 'right'),
            'gradient': g, 'sector': sec, 'label': label,
            'type': 'straight' if straight else 'corner',
        })
    return rows
