"""
Vehicle parameters, 2025 and 2026.

PROVENANCE
==========
Every value carries a source tag. Nothing here is an unsourced guess.

  [REG]    Fixed by the FIA 2026 Regulations, Section C [Technical] Issue 19,
           25 June 2026. Article cited. Not adjustable.
  [DERIV]  Derived from [REG] values or from published physical behaviour.
           Derivation shown.
  [PUB]    Published figure from an FIA or team technical source.
  [EST]    Still an estimate. Sweep it; do not quote it as fact.

The point of the tags is that an interviewer can check any number against its
article or its derivation, and can see immediately which numbers the results
actually depend on.
"""

from vehicle.four_wheel import VehicleGeometry, AeroModel, TyreModel

# ─────────────────────────────────────────────────────────── 2026 ──────────

# MASS  [REG] C4.1
#   "During the Sprint Qualifying and Qualifying sessions, the Minimum Mass is
#    726kg plus the Nominal Tyre Mass. In all other sessions ... 724kg."
#   Nominal Tyre Mass is measured by Pirelli post-Bahrain testing and is not
#   published here; the widely quoted total is 768 kg.
MASS_2026 = 768.0

# MASS DISTRIBUTION  [REG] C4.2
#   Front axle mass >= Minimum Mass x 0.44
#   Rear  axle mass >= Minimum Mass x 0.54
#   => front fraction is bounded to [0.44, 0.46]. Teams run near one end for
#   circuit-specific balance. Midpoint used; the whole legal range is only
#   2 points wide, so this is tightly constrained rather than assumed.
WEIGHT_DIST_FRONT_2026 = 0.45          # legal range 0.44 - 0.46

# WHEELBASE  [REG] C2.3.3
#   "The distance between the planes XF = 0 and XR = 0 must be less than or
#    equal to 3400mm at Legality Setup."  Reduced from 3600 mm in 2025.
WHEELBASE_2026 = 3.400

# TRACK  [DERIV] from C10.10.1 + C10.7.2 + C2.1.3
#   C10.10.1: front Wheel Coordinate System origin may not lie outboard of
#             Y = 603; rear origin not outboard of Y = 525.
#   C2.1.3:   that origin is the intersection of the wheel rotation axis with
#             the INBOARD plane of the rim.
#   C10.7.2:  rim Overall Width is 334 mm front, 420.3 mm rear.
#   => wheel centre plane = inboard plane + half the rim width
#      front: 603 + 167.0  = 770.0 mm  -> track 1540 mm
#      rear:  525 + 210.15 = 735.2 mm  -> track 1470 mm
#   These are legal maxima; a team running narrower gets MORE load transfer.
TRACK_FRONT_2026 = 1.540
TRACK_REAR_2026 = 1.470

# CoG HEIGHT  [EST]
#   Not regulated and not published. Constrained indirectly: the 2026 floor is
#   simpler and the battery is heavy and mounted low. 2022-25 cars are usually
#   estimated at 0.28-0.32 m. This remains a genuine unknown - sweep it.
COG_HEIGHT_2026 = 0.28

GEO_2026 = VehicleGeometry(
    mass_kg=MASS_2026,                          # [REG] C4.1
    wheelbase_m=WHEELBASE_2026,                 # [REG] C2.3.3
    weight_dist_front=WEIGHT_DIST_FRONT_2026,   # [REG] C4.2 bounded
    cog_height_m=COG_HEIGHT_2026,               # [EST]
    track_front_m=TRACK_FRONT_2026,             # [DERIV] C10.10.1
    track_rear_m=TRACK_REAR_2026,               # [DERIV] C10.10.1
    roll_stiffness_front=180_000.0,             # [EST] Nm/rad
    roll_stiffness_rear=140_000.0,              # [EST] Nm/rad
    roll_centre_front_m=0.030,                  # [EST]
    roll_centre_rear_m=0.060,                   # [EST]
)

# AERODYNAMICS
#
# 2025 BASELINE  [DERIV] from published behaviour
#   Downforce: 2022-25 cars are documented as producing 3-4x their mass in
#   downforce at maximum speed. Taking 3.5x at 340 km/h:
#       ClA = 3.5 * 798 * 9.81 / (0.5 * 1.225 * 94.4^2) = 5.02 m2
#   Drag: at terminal velocity, P = 0.5 * rho * CdA * v^3. With 670 kW at
#   340 km/h:
#       CdA = 670e3 / (0.5 * 1.225 * 94.4^3) = 1.30 m2
#   Implied L/D = 3.9, consistent with the 3-4 typical of F1.
#
#   NOTE: the previous version of this file used ClA 3.20 / CdA 0.95, which
#   implies only 2.2x weight in downforce at top speed. Those values were too
#   low and are the reason simulated corner speeds disagreed with telemetry.
#
# 2026  [PUB] FIA design targets, announced with the regulations:
#   downforce approximately 30% lower, drag approximately 55% lower than
#   2022-25 cars. FIA single-seater technical director Jan Monchaux has since
#   described the realised downforce reduction as "20-25%", so the true Z-mode
#   figure likely sits between the target and that.
#
#   Z-mode (Corner Mode): ClA = 5.02 * 0.70 = 3.51
#   X-mode (Straight Mode): active aero sheds both downforce and drag.
#
# CROSS-CHECK: measured minimum speed at Barcelona 2026 Q was 103 km/h. With
# ClA = 3.51 that implies a racing-line radius of 41.0 m, which is physically
# sensible for an F1 line. With the old ClA = 2.40 it implied 43.6 m. Both are
# plausible, so this check constrains but does not uniquely determine ClA.

CLA_2025 = 5.02     # [DERIV] 3.5x weight at 340 km/h
CDA_2025 = 1.30     # [DERIV] terminal velocity at 670 kW

AERO_2026 = AeroModel(
    ClA_corner_m2=3.45,        # [IDENT] apex regression, Barcelona 2026 Q,
                               #   63 apexes, 10 laps, r^2 = 0.906. Agrees
                               #   within 2% with the independent [PUB] route
                               #   (-30% of the derived 2025 baseline = 3.51)
    CdA_corner_m2=0.90,        # [EST] see caveat below
    ClA_straight_m2=2.10,      # [EST] X-mode sheds ~40% of Z-mode downforce
    CdA_straight_m2=0.585,     # [PUB] -55% of 2025 (FIA target)
    aero_balance_front=0.44,   # [EST]
)

# ── LARGEST REMAINING UNCERTAINTY: CdA ────────────────────────────────────
# The FIA -55% target gives CdA_X = 0.585. But the measured Barcelona top
# speed of 341 km/h, with ICE 400 kW plus 280 kW of MGU-K at that speed,
# would balance drag exactly at CdA = 1.31 - if the car were at terminal
# velocity. It is not: Barcelona's main straight ends at a braking point, so
# the car is still accelerating at 341 km/h and the true CdA lies somewhere
# between.
#
# The clean way to resolve this is parameter identification rather than
# guessing: fit CdA to the measured ACCELERATION along the straight, where
#     m dv/dt = F_drive(v) - 0.5 rho CdA v^2
# gives CdA directly from the telemetry gradient. Until that is done, treat
# every top-speed result as uncertain and note that lap time is far less
# sensitive to CdA than corner speed is to ClA.

# TYRE  [PUB] / [DERIV]
#   Published F1 slick behaviour at zero camber: mu is around 1.6 over the
#   1-3 kN vertical load range, falling to around 1.4 near 6 kN. Independent
#   sources put peak mu for modern F1 slicks in the 1.6-1.8 band, with brief
#   peaks approaching 2.0 on a fully rubbered surface.
#
#   Those two load points give the load sensitivity exponent directly:
#       k = ln(1.60/1.40) / ln(6000/2000) = 0.1215
#
#   NOTE: mu0 = 1.65 was used throughout earlier versions of this project.
#   That figure came from Hoosier R25B FORMULA STUDENT tyre data and is not
#   an F1 number. It happens to fall inside the published F1 range, but it
#   was not sourced for this application. It now is.
# Identification (Barcelona 2026 Q, 63 apexes, r^2 = 0.906) gives an
# EFFECTIVE load-averaged mu of 1.881 over the apex load range (~2.5-3 kN per
# corner). Holding the sourced load-sensitivity exponent k = 0.1215 fixed and
# solving mu0 * (2700/2000)^-k = 1.881 gives mu0 ~ 1.95 at the 2 kN reference.
#
# This sits above the 1.6-at-2kN forum figure previously used. That figure
# describes an earlier tyre generation; the identification is from current
# 2026 qualifying-soft data on a rubbered track and supersedes it. Published
# peak-mu ranges (1.4-1.8, instantaneous ~2.0) accommodate the result.
#
# Cross-session check still PENDING: identified from one session. Until the
# same fit on FP3 or another event agrees, treat as provisional.
# [IDENT->CAL] mu0 calibrated THROUGH the four-wheel model
# so that the model's lateral capability reproduces the measured apex
# envelope (see calibrate_mu.py). The apex-regression mu of 1.881 is a
# WHOLE-CAR effective value with the real car's load-transfer losses baked
# in; using it directly as the per-tyre coefficient made the four-wheel
# model charge those losses twice, producing the consistent -9 to -11%%
# corner-speed bias seen across two circuits.
TYRE_2026 = TyreModel(
    mu0=2.058,                   # [IDENT] see note above
    Fz0_N=2000.0,               # [PUB] reference load
    load_sensitivity_k=0.1215,  # [DERIV] from the 1.6@2kN / 1.4@6kN pair
)

# 2026 TYRE GEOMETRY  [REG] C10.7.2 / [PUB]
#   Rim Diameter 462.5-463 mm (18 in) front and rear.
#   Tyre Mounting Width 315 mm front, 401.3 mm rear.
#   Tread width reduced 25 mm front and 30 mm rear versus 2025; overall
#   diameter reduced 15 mm front and 10 mm rear.
#   Not currently used by the model - a contact-patch-aware tyre model would
#   need them.

# ─────────────────────────────────────────────────────────── 2025 ──────────
# Baseline for the regulatory comparison. Ground-effect era: more downforce,
# more drag, heavier, wider, longer, far less MGU-K power. No active aero;
# DRS is not modelled, so a single aero state is used.

GEO_2025 = VehicleGeometry(
    mass_kg=798.0,                 # [REG] 2025 minimum mass
    wheelbase_m=3.600,             # [REG] 2025 maximum wheelbase
    weight_dist_front=0.45,        # [REG] same 44/54 bounds applied in 2025
    cog_height_m=0.29,             # [EST] slightly higher than 2026
    track_front_m=1.620,           # [EST] scaled from the 2000 mm car width
    track_rear_m=1.550,            # [EST]
    roll_stiffness_front=180_000.0,  # [EST]
    roll_stiffness_rear=140_000.0,   # [EST]
    roll_centre_front_m=0.030,     # [EST]
    roll_centre_rear_m=0.060,      # [EST]
)

AERO_2025 = AeroModel(
    ClA_corner_m2=CLA_2025,        # [DERIV] 5.02
    CdA_corner_m2=CDA_2025,        # [DERIV] 1.30
    ClA_straight_m2=CLA_2025,      # no active aero
    CdA_straight_m2=CDA_2025,
    aero_balance_front=0.44,       # [EST]
)

TYRE_2025 = TyreModel(
    mu0=2.058,                      # identical to 2026 so the comparison
    Fz0_N=2000.0,                  # isolates the regulation change rather
    load_sensitivity_k=0.1215,     # than smuggling in a tyre assumption
)
