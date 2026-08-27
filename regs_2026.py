"""
FIA 2026 energy and active-aero regulations.

Source: FIA 2026 F1 Regulations, Section C [Technical] Issue 19 and
        Section B [Sporting] Issue 07, both 25 June 2026.

Article references are given inline. Where a value is a per-event parameter
rather than a fixed constant, that is stated - the FIA publishes those in a
document issued at least four weeks before each Competition (B7.2.1(b)), and
they are config inputs here, not hard-coded truths.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class Session(Enum):
    """
    B1 Appendix definitions.

    LTCS - Lap Time Classified Session: FP, Sprint Qualifying, Qualifying.
           Classified on a single lap.
    TTCS - Total Time Classified Session: Sprint, Race.
           Classified on total time over many laps.

    The distinction selects the deployment profile, because Overtake is
    enabled and ACTIVATED AT ALL TIMES in an LTCS (B7.2.3(b)) but is
    proximity-gated in a TTCS (B7.2.3(c)).
    """
    LTCS = "ltcs"
    TTCS = "ttcs"


# --------------------------------------------------------- deployment power

def deployment_limit_kw(v_ms: float, overtake_active: bool = False,
                        in_specified_sector: bool = False) -> float:
    """
    Maximum ERS-K DC power used to PROPEL the car (C5.2.8), in kW.

    Three profiles:

    (i)   Overtake not active
              v < 340        P = 1800 - 5v
              340 <= v < 345 P = 6900 - 20v
              v >= 345       P = 0
          The 350 kW ceiling from C5.2.7 binds below 290 km/h.

    (ii)  Overtake active
              v < 355        P = 7100 - 20v
              v >= 355       P = 0
          Ceiling binds below 337.5 km/h.

    (iii) Race/Sprint, specified sectors, subject to B7.2
              v < 310        P = 250
              310 <= v < 340 P = 1800 - 5v
              340 <= v < 345 P = 6900 - 20v
              v >= 345       P = 0
          Note the 250 kW cap only bites BELOW 310 km/h; above that it is
          identical to profile (i).

    Per B7.2.1(b)(i)-(ii) these curves may be adjusted per circuit. Treat the
    coefficients here as the regulatory baseline.
    """
    v = v_ms * 3.6
    CEIL = 350.0                                   # C5.2.7 absolute limit

    if overtake_active:                            # profile (ii)
        return 0.0 if v >= 355.0 else min(CEIL, 7100.0 - 20.0 * v)

    if in_specified_sector and v < 310.0:          # profile (iii)
        return 250.0

    if v < 340.0:                                  # profile (i)
        return min(CEIL, 1800.0 - 5.0 * v)
    if v < 345.0:
        return max(0.0, 6900.0 - 20.0 * v)
    return 0.0


def harvest_limit_kw(v_ms: float) -> float:
    """
    Maximum ERS-K DC power for Recharge, in kW.

    C5.2.8's speed taper applies to power "used to propel the car" only.
    Harvest is bounded by the ABSOLUTE limit in C5.2.7 (350 kW, both
    directions), with no speed dependence.

    A physical limit also applies via C5.2.11 (MGU-K torque <= 500 Nm at the
    crank), which binds at low speed, and by rear-axle grip. Those are
    applied by the caller, which knows the load state.
    """
    return 350.0


@dataclass
class EnergyConfig:
    """
    Per-event energy configuration.

    Defaults are the regulatory baseline. Real values come from the FIA
    pre-event document (B7.2.1(b)) and should be set explicitly per circuit.
    """
    session: Session = Session.LTCS

    # Terminal state of charge condition for the DP optimiser.
    #
    #   'deplete'   SoC starts full, finishes empty. Correct for a QUALIFYING
    #               flying lap: the car arrives from an out-lap with a full
    #               store, and any charge left at the flag is lap time thrown
    #               away. Budget = the 4 MJ carried in PLUS everything
    #               harvested during the lap.
    #
    #   'periodic'  SoC(finish) = SoC(start). Correct for a RACE stint, where
    #               the next lap must also be possible, and which forces
    #               deployment to equal harvest.
    #
    # Using 'periodic' for a flying lap understates deployment badly.
    terminal_mode: str = 'deplete'

    # C5.2.9 - max minus min state of charge on track.
    delta_soc_max_MJ: float = 4.0

    # C5.2.10 - Recharge per lap at the CU-K HV DC Bus.
    #   baseline 8.5 MJ
    #   reducible to 7 MJ where the FIA determines max harvestable <= 7 MJ
    #   further reducible to >= 5 MJ, Sprint Qualifying and Qualifying ONLY
    #   +0.5 MJ subject to B7.2
    recharge_max_MJ: float = 8.5

    # Which sectors carry the profile (iii) 250 kW limb. TTCS only.
    specified_sectors: tuple = ()

    # ICE at maximum permitted fuel flow on all non-braking segments.
    # Justified because fuel is not the binding constraint in 2026;
    # electrical energy is. Stated as an assumption, not a derivation.
    ice_power_kW: float = 400.0

    def overtake_active(self) -> bool:
        """
        B7.2.2(a) - Overtake enabled before any LTCS.
        B7.2.3(b) - in an LTCS it is ACTIVATED AT ALL TIMES.

        So a qualifying lap runs profile (ii), not profile (i). This is the
        single most consequential regulation detail for a single-lap sim.
        In a TTCS activation is proximity-gated and needs a second car, which
        a single-car simulation cannot represent.
        """
        return self.session == Session.LTCS


# ------------------------------------------------------------- active aero

@dataclass
class ActivationZones:
    """
    Driver Adjustable Bodywork zones (B7.1).

    Unlike DRS, activation is NOT proximity-gated (B7.1.2(a)) - the driver may
    activate whenever notified it is enabled. Zones are published at least
    four weeks before the Competition (B7.1.1(e)) and marked trackside
    (B7.1.1(f)).

    Two aero states (C3.10.10(n), C3.11.6(c)):
        Corner Mode   - both front wing and rear flap at high incidence
        Straight Mode - both at reduced incidence (full activation)
    Partial activation (front wing Straight, rear flap Corner) is the Low Grip
    configuration and is not modelled here.

    Transition takes 400 ms (C3.10.10(o), C3.11.6(d)), applied as a ramp.
    """
    zones: tuple = ()          # ((s_start_m, s_end_m), ...)
    transition_s: float = 0.4

    def mask(self, s: np.ndarray, v: np.ndarray | None = None) -> np.ndarray:
        """
        Boolean array: True where the car is in Straight Mode.

        If v is supplied, the zone entry is delayed by the transition time
        converted to distance at local speed, so the aero change is not
        instantaneous at the zone boundary.
        """
        m = np.zeros(len(s), dtype=bool)
        for s0, s1 in self.zones:
            if v is None:
                m |= (s >= s0) & (s <= s1)
            else:
                lag = self.transition_s * np.maximum(v, 1.0)
                m |= (s >= (s0 + lag)) & (s <= s1)
        return m


# Barcelona activation zones - ESTIMATED from straight positions in the track
# model. The real zones come from the FIA pre-event document.
BARCELONA_ZONES = ActivationZones(zones=(
    (0, 640),        # start/finish straight, line to T1 braking. Omitting
                     # this ran corner-mode drag on the circuit's main
                     # straight - visible as a flat simulated trace at
                     # ~307 km/h while the real car pulled to 341.
    (1110, 1380),    # T4-T5 straight
    (1460, 1600),    # T5-T6 straight
    (2490, 2580),    # T9-T10
    (3480, 3600),    # T13-T14
    (4080, 4640),    # final straight to start/finish
))
