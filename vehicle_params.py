"""
F1 car parameters: 2025 baseline and 2026 regulatory estimates.

All values are estimates. Primary source: project summary doc.
Secondary sources: FIA 2026 Technical Regulations, published chassis
homologation data, academic references (Milliken & Milliken).

Notation follows Milliken & Milliken Chapter 8 (QSS point mass).
"""

# --- 2025 Baseline ---
CAR_2025 = {
    # Inertia & geometry
    'm_kg':        798.0,    # total mass inc. driver (FIA minimum + fuel)
    'CdA_m2':      0.95,     # drag coefficient × reference area (m²)
    'ClA_m2':      3.20,     # downforce coefficient × ref area (signed +ve down)
    'wheelbase_m': 3.60,     # front-to-rear axle (m)
    'track_m':     1.60,     # mean track width (m)
    'CoG_h_m':     0.29,     # centre of gravity height (m)
    'mu':          1.65,     # peak tyre-road friction coefficient (compound C3)
    
    # Powertrain
    'P_ice_kW':    550.0,    # internal combustion engine peak power (kW)
    'P_mguk_kW':   120.0,    # MGU-K peak power (kW)
    'P_total_kW':  670.0,    # combined ICE + MGU-K (kW)
    'E_deploy_MJ': 4.0,      # max battery energy deployable per lap (MJ)
    'E_harvest_MJ':2.0,      # max harvest per lap (MJ)
    
    # Aero reference
    'rho_kgm3':    1.225,    # air density (kg/m³) at sea level, 15°C
    # Note: Barcelona altitude ~100m → rho ≈ 1.210 kg/m³ — use 1.225 for simplicity
    
    # Tyre rolling resistance (first-order model)
    'Crr':         0.02,     # rolling resistance coefficient
}

# --- 2026 Regulatory Estimate ---
CAR_2026 = {
    'm_kg':        768.0,    # ~30 kg lighter (smaller ICE, smaller fuel tank)
    'CdA_m2':      0.70,     # ~26% drag reduction (active aero + smaller bodywork)
    'ClA_m2':      2.40,     # ~25% downforce reduction
    'wheelbase_m': 3.60,
    'track_m':     1.60,
    'CoG_h_m':     0.28,     # slightly lower (battery repositioning)
    'mu':          1.65,     # same compound assumption
    
    'P_ice_kW':    400.0,    # smaller ICE under 2026 regs
    'P_mguk_kW':   350.0,    # ~3x MGU-K power (50:50 power split target)
    'P_total_kW':  750.0,    # higher total installed power
    'E_deploy_MJ': 4.0,      # same usable energy per lap
    'E_harvest_MJ':8.5,      # 4.25x harvest cap vs 2025
    
    'rho_kgm3':    1.225,
    'Crr':         0.02,
    
    # 2026 MGU-K deployment rules (FIA 2026 Technical Regs, Article 5.x)
    # Below 340 km/h: P_deploy = min(350, 1800 - 5*v_kmh) kW
    # Above 340 km/h: P_deploy = max(0, 6900 - 20*v_kmh) kW
    # Zero above 345 km/h
    'mguk_rules': '2026_fia',
}

def mguk_limit_kw(v_ms, year=2026):
    """
    Returns MGU-K deployment power limit (kW) at speed v_ms (m/s).
    
    2026 FIA rules:
      v < 340 km/h: P = min(350, 1800 - 5*v_kmh)  [kW]
      v >= 340 km/h: P = max(0, 6900 - 20*v_kmh)  [kW]
    
    Args:
        v_ms   : speed (m/s)
        year   : 2025 or 2026
    
    Returns:
        P_kW : float
    """
    if year == 2025:
        return 120.0  # constant limit
    
    v_kmh = v_ms * 3.6
    if v_kmh < 340.0:
        return min(350.0, 1800.0 - 5.0 * v_kmh)
    else:
        return max(0.0,  6900.0 - 20.0 * v_kmh)


def aero_forces(v_ms, car):
    """
    Returns aerodynamic drag (N) and downforce (N) at speed v_ms.
    
    F_drag     = 0.5 * rho * CdA * v²
    F_downforce= 0.5 * rho * ClA * v²
    
    Assumptions:
    - Fixed CdA, ClA (no DRS, no aero map)
    - Symmetric aero (no cornering aero sensitivity)
    - DRS is not modelled — add as a toggle in future iteration
    """
    q = 0.5 * car['rho_kgm3'] * v_ms**2
    F_drag      = q * car['CdA_m2']
    F_downforce = q * car['ClA_m2']
    return F_drag, F_downforce


def normal_load(v_ms, car, kappa=0.0, gradient=0.0):
    """
    Returns total normal load (N) on all four tyres.
    
    N = m*g*cos(theta) + F_downforce + m*v²*kappa*h_CoG (load transfer correction)
    
    Simplification: lateral load transfer is not distributed
    between front/rear axles — use total normal load for the
    point mass friction circle. This is a known limitation of
    the point mass model vs a 4-wheel model.
    
    Args:
        v_ms     : speed (m/s)
        car      : dict from CAR_2025 or CAR_2026
        kappa    : curvature (1/m)
        gradient : longitudinal grade (m/m)
    
    Returns:
        N_total : total normal load (N)
    """
    import math
    g = 9.81
    theta = math.atan(gradient)  # road angle (rad)
    _, F_df = aero_forces(v_ms, car)
    N_gravity   = car['m_kg'] * g * math.cos(theta)
    N_total = N_gravity + F_df
    # Note: centrifugal normal load component (mv²kappa*h) is second order
    # for a point mass — omitted here, included in future 4-wheel model
    return N_total

# ── SINGLE SOURCE OF TRUTH FOR AERODYNAMICS ──────────────────────────────
# The CAR_* dicts feed the longitudinal model (drag, downforce in the
# accel/brake capability, and the whole point-mass path) while the AeroModel
# objects feed the four-wheel corner solver. If the two disagree, the car
# corners on one aero map and accelerates on another - which produced a
# four-wheel model FASTER than point mass, an impossible result, because it
# got high-downforce cornering with low-downforce drag.
#
# The AeroModel objects in vehicle/f1_2026_params.py are authoritative.
# These overwrites keep the dicts synchronised with them.

def _sync_aero():
    """
    Synchronise the CAR_* dicts with the authoritative AeroModel and
    TyreModel objects.

    The tyre is synchronised as well as the aero, and the reason is a bug
    this file has now produced TWICE: any parameter that exists in both the
    dicts (point-mass path, longitudinal capability) and the model objects
    (four-wheel corner solver) will silently diverge when one is updated,
    and the symptom is a four-wheel model FASTER than point mass - a
    physically impossible result, since load transfer only ever costs grip.

    The point-mass mu is a single scalar standing in for a load-sensitive
    curve, so it is taken as the TyreModel evaluated at a representative
    per-corner load (static weight plus downforce at 150 km/h, quarter-car).
    Any single choice is an approximation - that approximation being
    unavoidable is precisely why the four-wheel model exists.
    """
    import math
    from vehicle.f1_2026_params import (AERO_2025, AERO_2026,
                                        TYRE_2025, TYRE_2026)
    v_ref = 150.0 / 3.6
    for car, aero, tyre in ((CAR_2025, AERO_2025, TYRE_2025),
                            (CAR_2026, AERO_2026, TYRE_2026)):
        car['ClA_m2'] = aero.ClA_corner_m2
        car['CdA_m2'] = aero.CdA_corner_m2
        car['ClA_straight_m2'] = aero.ClA_straight_m2
        car['CdA_straight_m2'] = aero.CdA_straight_m2
        Fz_ref = (car['m_kg'] * 9.80665
                  + 0.5 * car['rho_kgm3'] * aero.ClA_corner_m2 * v_ref ** 2) / 4.0
        car['mu'] = tyre.mu(Fz_ref)
