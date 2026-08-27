"""
Tier 2 — four-wheel quasi-static vehicle model.

Objective
---------
Given speed, curvature, longitudinal acceleration and gradient, compute the
normal load at each of the four contact patches, the grip available at each,
and the resulting steady-state lateral acceleration limit of the car.

This module is GEOMETRY-AGNOSTIC. It takes a VehicleGeometry and a TyreModel
and knows nothing about which car it describes. The same code serves the F1
simulator and the Formula Student validation study; only the parameter file
differs.

Sign conventions (SAE, Z up)
----------------------------
    kappa  > 0  ->  left turn
    a_y    > 0  ->  leftward lateral acceleration; load transfers RIGHT
    a_x    > 0  ->  acceleration; load transfers REAR
    a_x    < 0  ->  braking;      load transfers FRONT

Corner indices: 0 = front-left, 1 = front-right, 2 = rear-left, 3 = rear-right

Method
------
Lateral load transfer splits into two paths (Milliken & Milliken Ch. 18):

    geometric  - acts instantaneously through the roll centre, no roll required
    elastic    - acts through springs and anti-roll bars, requires body roll

Front axle transfer:

    dFz_f = (m*a_y / t_f) * [ h_roll * K_phi_f/(K_phi_f + K_phi_r)
                              + (l_r/L) * z_RC_f ]

The elastic term is the LLTD knob: changing the roll stiffness split moves
transfer between axles without changing the total.

Grip is load-sensitive:

    mu(Fz) = mu0 * (Fz / Fz0) ** (-k)

This is why load transfer costs lap time. The outer tyre gains less than the
inner tyre loses, so an axle nets out with less grip after transfer than
before. If k were zero, LLTD would have no effect on balance whatsoever.

Steady-state cornering requires yaw moment balance about the CoG:

    F_yf * l_f = F_yr * l_r

so each axle must produce lateral force in proportion to its static weight
share. The car is therefore limited by whichever axle saturates first:

    a_y_max = min( F_yf_max / (m * l_r/L),  F_yr_max / (m * l_f/L) )

Implicit loop
-------------
a_y depends on grip, which depends on load transfer, which depends on a_y.
Solved by fixed-point iteration at each track point.

Limitations
-----------
- Quasi-static. No roll, pitch or yaw transients; no damper forces.
- Roll centre heights are constants here. A kinematics solver can supply
  z_RC(roll, ride height) maps instead - see set_roll_centre_map().
- No suspension compliance; links assumed rigid.
- No camber or toe effects on grip (needs tyre data not publicly available
  for F1 tyres).
- No tyre thermal model. mu0 is a cold-to-optimum constant.
- Aero balance is a constant, not a function of ride height or roll.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional
import math
import numpy as np

G = 9.80665

FL, FR, RL, RR = 0, 1, 2, 3


# ---------------------------------------------------------------- geometry

@dataclass
class VehicleGeometry:
    """Mass and geometric properties. All SI."""
    mass_kg: float
    wheelbase_m: float
    weight_dist_front: float      # static fraction of mass on front axle
    cog_height_m: float
    track_front_m: float
    track_rear_m: float

    # Roll stiffness (Nm/rad) about the roll axis, per axle.
    # These come from spring rate x motion_ratio^2 x track^2 / 2, plus ARB.
    roll_stiffness_front: float
    roll_stiffness_rear: float

    # Roll centre heights above ground (m), per axle.
    roll_centre_front_m: float
    roll_centre_rear_m: float

    @property
    def l_f(self) -> float:
        """CoG to front axle (m)."""
        return self.wheelbase_m * (1.0 - self.weight_dist_front)

    @property
    def l_r(self) -> float:
        """CoG to rear axle (m)."""
        return self.wheelbase_m * self.weight_dist_front

    @property
    def lltd(self) -> float:
        """
        Elastic roll stiffness fraction on the front axle.

        This is the primary balance knob. It is NOT the full lateral load
        transfer distribution, because the geometric term also contributes -
        use effective_lltd() for the realised split.
        """
        return self.roll_stiffness_front / (
            self.roll_stiffness_front + self.roll_stiffness_rear)

    def roll_axis_height_at_cog(self) -> float:
        """Height of the roll axis directly beneath the CoG (m)."""
        f = self.l_f / self.wheelbase_m
        return self.roll_centre_front_m + \
            (self.roll_centre_rear_m - self.roll_centre_front_m) * f

    def h_roll(self) -> float:
        """CoG height above the roll axis (m). Drives the elastic term."""
        return self.cog_height_m - self.roll_axis_height_at_cog()


@dataclass
class AeroModel:
    """
    Downforce and drag. Supports the 2026 two-state active aero.

    Corner Mode  = high incidence, high downforce, high drag
    Straight Mode = reduced incidence (FIA Std ECU commanded, C3.10.10/C3.11.6)
    """
    ClA_corner_m2: float
    CdA_corner_m2: float
    ClA_straight_m2: float
    CdA_straight_m2: float
    aero_balance_front: float     # fraction of downforce on the front axle
    rho_kgm3: float = 1.225

    def coeffs(self, straight_mode: bool = False):
        if straight_mode:
            return self.ClA_straight_m2, self.CdA_straight_m2
        return self.ClA_corner_m2, self.CdA_corner_m2

    def downforce_N(self, v_ms: float, straight_mode: bool = False) -> float:
        ClA, _ = self.coeffs(straight_mode)
        return 0.5 * self.rho_kgm3 * ClA * v_ms * v_ms

    def drag_N(self, v_ms: float, straight_mode: bool = False) -> float:
        _, CdA = self.coeffs(straight_mode)
        return 0.5 * self.rho_kgm3 * CdA * v_ms * v_ms


@dataclass
class TyreModel:
    """
    Load-sensitive friction.

        mu(Fz) = mu0 * (Fz/Fz0) ** (-k)

    k is the load sensitivity exponent. For racing slicks it lies roughly in
    0.10-0.20, but the true value for a 2026 Pirelli is not public. Treat it
    as a swept parameter and report lap time sensitivity to it rather than
    quoting a single absolute lap time.
    """
    mu0: float
    Fz0_N: float
    load_sensitivity_k: float = 0.15
    mu_floor: float = 0.1

    def mu(self, Fz_N: float) -> float:
        if Fz_N <= 1.0:
            return 0.0
        return max(self.mu_floor,
                   self.mu0 * (Fz_N / self.Fz0_N) ** (-self.load_sensitivity_k))

    def lateral_capacity_N(self, Fz_N: float) -> float:
        """Peak lateral force this contact patch can carry."""
        return self.mu(Fz_N) * max(Fz_N, 0.0)


# ------------------------------------------------------------ load transfer

@dataclass
class CornerLoads:
    """Per-corner state at one track point."""
    Fz: np.ndarray                # (4,) normal load, N
    mu: np.ndarray                # (4,) friction coefficient
    Fy_capacity: np.ndarray       # (4,) peak lateral force, N
    a_y_achieved: float           # m/s^2
    n_iterations: int
    converged: bool
    wheel_lifted: bool
    lltd_effective: float         # realised front share of lateral transfer


class FourWheelModel:
    """
    Quasi-static four-wheel load transfer and grip model.

    Typical use:
        model = FourWheelModel(geometry, aero, tyre)
        loads = model.solve_corner_loads(v_ms=60.0, kappa=0.01, a_x=0.0)
        a_y_max = model.max_lateral_acceleration(v_ms=60.0, kappa=0.01)
    """

    def __init__(self, geometry: VehicleGeometry, aero: AeroModel,
                 tyre: TyreModel):
        self.geo = geometry
        self.aero = aero
        self.tyre = tyre
        self._rc_map: Optional[Callable[[float, float], tuple]] = None

    # -- optional kinematics coupling ------------------------------------

    def set_roll_centre_map(self, fn: Callable[[float, float], tuple]) -> None:
        """
        Supply roll centre heights as a function of state instead of constants.

        fn(roll_angle_rad, ride_height_m) -> (z_RC_front_m, z_RC_rear_m)

        This is the hook for a suspension kinematics solver. Roll centres
        migrate with roll and heave; a static value is an approximation.
        Once supplied, z_RC enters the same fixed-point loop as a_y, because
        roll angle depends on load transfer which depends on z_RC.
        """
        self._rc_map = fn

    def _roll_centres(self, roll_rad: float, ride_height_m: float):
        if self._rc_map is None:
            return self.geo.roll_centre_front_m, self.geo.roll_centre_rear_m
        return self._rc_map(roll_rad, ride_height_m)

    # -- roll -------------------------------------------------------------

    def roll_angle_rad(self, a_y: float) -> float:
        """
        Steady-state body roll.

            phi = m * a_y * h_roll / (K_phi_total - m * g * h_roll)

        The -m*g*h_roll term is the roll-over moment from the CoG rising as
        the body rolls. It reduces effective roll stiffness.
        """
        g = self.geo
        h = g.h_roll()
        K = g.roll_stiffness_front + g.roll_stiffness_rear
        denom = K - g.mass_kg * G * h
        if denom <= 0.0:
            return 0.0          # roll-unstable; guarded by caller
        return g.mass_kg * a_y * h / denom

    # -- core -------------------------------------------------------------

    def solve_corner_loads(self, v_ms: float, kappa: float, a_x: float = 0.0,
                           gradient: float = 0.0, straight_mode: bool = False,
                           a_y_override: Optional[float] = None,
                           max_iter: int = 50, tol: float = 1e-4
                           ) -> CornerLoads:
        """
        Compute the four normal loads and the achievable lateral acceleration.

        Inputs:
            v_ms          speed (m/s)
            kappa         SIGNED curvature (1/m), +ve = left
            a_x           longitudinal acceleration (m/s^2), -ve = braking
            gradient      road slope (m/m)
            straight_mode active aero state
            a_y_override  force a specific a_y instead of solving for the limit

        Returns CornerLoads.
        """
        g = self.geo
        m = g.mass_kg
        L = g.wheelbase_m

        # Static + aero + longitudinal transfer are independent of a_y.
        theta = math.atan(gradient)
        W = m * G * math.cos(theta)
        F_df = self.aero.downforce_N(v_ms, straight_mode)

        Fz_f_axle = W * g.weight_dist_front + F_df * self.aero.aero_balance_front
        Fz_r_axle = W * (1.0 - g.weight_dist_front) + \
            F_df * (1.0 - self.aero.aero_balance_front)

        # Longitudinal transfer: braking loads the front.
        dFz_x = m * a_x * g.cog_height_m / L
        Fz_f_axle -= dFz_x
        Fz_r_axle += dFz_x

        Fz_f_axle = max(Fz_f_axle, 0.0)
        Fz_r_axle = max(Fz_r_axle, 0.0)

        # Fixed-point loop on a_y.
        a_y = 0.0 if a_y_override is None else a_y_override
        if a_y_override is None and abs(kappa) > 1e-9:
            a_y = v_ms * v_ms * kappa          # first guess: kinematic demand

        Fz = np.zeros(4)
        mu = np.zeros(4)
        cap = np.zeros(4)
        lltd_eff = g.lltd
        converged = False
        it = 0

        for it in range(1, max_iter + 1):
            roll = self.roll_angle_rad(a_y)
            z_rc_f, z_rc_r = self._roll_centres(roll, 0.0)

            f = g.l_f / L
            roll_axis_at_cog = z_rc_f + (z_rc_r - z_rc_f) * f
            h_roll = g.cog_height_m - roll_axis_at_cog

            K_f, K_r = g.roll_stiffness_front, g.roll_stiffness_rear
            K_tot = K_f + K_r

            # Elastic (roll stiffness) + geometric (roll centre) contributions.
            dFz_f = (m * a_y / g.track_front_m) * (
                h_roll * K_f / K_tot + (g.l_r / L) * z_rc_f)
            dFz_r = (m * a_y / g.track_rear_m) * (
                h_roll * K_r / K_tot + (g.l_f / L) * z_rc_r)

            total = abs(dFz_f) + abs(dFz_r)
            lltd_eff = abs(dFz_f) / total if total > 1e-9 else g.lltd

            # Positive a_y (left turn) transfers load to the RIGHT wheels.
            Fz[FL] = 0.5 * Fz_f_axle - dFz_f
            Fz[FR] = 0.5 * Fz_f_axle + dFz_f
            Fz[RL] = 0.5 * Fz_r_axle - dFz_r
            Fz[RR] = 0.5 * Fz_r_axle + dFz_r
            Fz = np.maximum(Fz, 0.0)          # a lifted wheel carries nothing

            for i in range(4):
                mu[i] = self.tyre.mu(Fz[i])
                cap[i] = self.tyre.lateral_capacity_N(Fz[i])

            if a_y_override is not None:
                converged = True
                break

            # Yaw moment balance: each axle carries lateral force in
            # proportion to its static weight share.
            Fy_f = cap[FL] + cap[FR]
            Fy_r = cap[RL] + cap[RR]
            share_f = g.l_r / L
            share_r = g.l_f / L

            a_y_f = Fy_f / (m * share_f) if share_f > 1e-9 else 1e9
            a_y_r = Fy_r / (m * share_r) if share_r > 1e-9 else 1e9
            a_y_cap = min(a_y_f, a_y_r)

            # The car cannot exceed the kinematic demand of the corner.
            if abs(kappa) > 1e-9:
                a_y_demand = v_ms * v_ms * abs(kappa)
                a_y_new = min(a_y_cap, a_y_demand)
            else:
                a_y_new = 0.0

            a_y_new = math.copysign(a_y_new, kappa if kappa != 0 else 1.0)

            if abs(a_y_new - a_y) < tol:
                a_y = a_y_new
                converged = True
                break
            # Damped update - the loop is stiff at high load transfer.
            a_y = a_y + 0.6 * (a_y_new - a_y)

        return CornerLoads(
            Fz=Fz.copy(), mu=mu.copy(), Fy_capacity=cap.copy(),
            a_y_achieved=a_y, n_iterations=it, converged=converged,
            wheel_lifted=bool((Fz <= 1.0).any()),
            lltd_effective=lltd_eff,
        )

    def max_lateral_acceleration(self, v_ms: float, kappa: float,
                                 a_x: float = 0.0, gradient: float = 0.0,
                                 straight_mode: bool = False) -> float:
        """
        Peak |a_y| the car can sustain at this speed, i.e. the grip CAPACITY.

        Note this is deliberately not solve_corner_loads().a_y_achieved, which
        returns min(capacity, kinematic demand). Probing with a curvature far
        larger than any real corner forces the capacity to bind.
        """
        sign = math.copysign(1.0, kappa) if kappa != 0 else 1.0
        big = sign * 1e4 / max(v_ms * v_ms, 1.0)
        loads = self.solve_corner_loads(v_ms, big, a_x, gradient, straight_mode)
        return abs(loads.a_y_achieved)

    def max_longitudinal_acceleration(self, v_ms: float, kappa: float,
                                      gradient: float = 0.0,
                                      straight_mode: bool = False,
                                      braking: bool = True,
                                      max_iter: int = 30,
                                      tol: float = 0.01) -> float:
        """
        Maximum |a_x| available while cornering at curvature kappa, from the
        friction ellipse:

            (Fy/Fy_max)^2 + (Fx/Fx_max)^2 = 1

        Doubly implicit: a_x transfers load between axles, changing per-corner
        grip, which changes both lateral and longitudinal capacity - while the
        lateral demand is fixed by the corner. Fixed-point iteration on a_x.

        braking=True  -> all four wheels contribute
        braking=False -> traction on the driven (rear) axle only

        Returns |a_x| in m/s^2, always positive.
        """
        m = self.geo.mass_kg
        a_x = 0.0
        idx = [FL, FR, RL, RR] if braking else [RL, RR]
        sgn = -1.0 if braking else +1.0

        for _ in range(max_iter):
            loads = self.solve_corner_loads(v_ms, kappa, sgn * a_x,
                                            gradient, straight_mode)
            a_y_demand = v_ms * v_ms * abs(kappa)
            cap_all = float(np.sum(loads.Fy_capacity))

            Fx_total = 0.0
            if cap_all > 1.0:
                for i in idx:
                    cap_i = loads.Fy_capacity[i]
                    if cap_i <= 1.0:
                        continue
                    # Lateral demand shared in proportion to capacity.
                    Fy_i = (m * a_y_demand) * (cap_i / cap_all)
                    util = min(Fy_i / cap_i, 1.0)
                    Fx_total += cap_i * math.sqrt(max(0.0, 1.0 - util * util))

            a_x_new = Fx_total / m
            if abs(a_x_new - a_x) < tol:
                a_x = a_x_new
                break
            a_x = a_x + 0.6 * (a_x_new - a_x)

        return max(a_x, 0.0)

    def max_corner_speed(self, kappa: float, gradient: float = 0.0,
                         straight_mode: bool = False,
                         v_lo: float = 5.0, v_hi: float = 130.0,
                         tol: float = 0.01) -> float:
        """
        Highest speed at which the corner of curvature kappa can be held.

        Bisection on  a_y_available(v) - v^2*|kappa| = 0.
        Downforce grows as v^2, so a_y_available also rises with speed; the
        root is where the two curves cross.
        """
        k = abs(kappa)
        if k < 1e-9:
            return v_hi

        def excess(v):
            """Grip capacity minus kinematic demand. Root is the limit speed."""
            cap = self.max_lateral_acceleration(v, math.copysign(k, kappa),
                                                0.0, gradient, straight_mode)
            return cap - v * v * k

        if excess(v_lo) < 0.0:
            return v_lo
        if excess(v_hi) > 0.0:
            return v_hi

        for _ in range(60):
            v_mid = 0.5 * (v_lo + v_hi)
            if excess(v_mid) > 0.0:
                v_lo = v_mid
            else:
                v_hi = v_mid
            if v_hi - v_lo < tol:
                break
        return 0.5 * (v_lo + v_hi)
