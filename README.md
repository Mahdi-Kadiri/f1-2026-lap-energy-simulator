# F1 2026 Regulation Performance Simulator

A quasi-steady-state minimum lap time simulator with four-wheel load transfer,
dynamic-programming optimisation of hybrid energy deployment under the 2026
FIA regulations, and tyre/aero parameters identified from real 2026 telemetry.

Circuit: Circuit de Barcelona-Catalunya. Language: Python.

---

## Headline result — validated across two circuits

Tyre grip and downforce were identified from ten Barcelona Qualifying laps,
then applied UNCHANGED to two independent tests: a Barcelona free practice lap
outside the fit set, and a different circuit entirely.

| | In-sample (Barcelona Q) | Out-of-sample (Barcelona FP3) | **Different circuit (Silverstone Q)** |
|---|---|---|---|
| Reference | RUS 74.679 s | RUS 75.679 s | ANT 88.111 s |
| Simulated | 74.110 s | 74.110 s | 83.547 s |
| Lap delta | −0.76% | −2.07% | −5.18% |
| **Corner speed bias** | **−9.5%** | **−7.0%** | **−9.4%** |
| Correlation | 0.937 | 0.926 | **0.941** |
| Velocity RMS | 24.2 km/h | 26.6 km/h | 30.5 km/h |
| Terminal speed error | +6.5 km/h | +10.5 km/h | **+30.7 km/h** |

### What is validated

**Cornering performance transfers across circuits.** Corner speed bias is
−9.5% at Barcelona and −9.4% at Silverstone: parameters fitted at one circuit
reproduce cornering at a different layout, with a different corner mix and a
different driver, to within 0.1 percentage points. Correlation at Silverstone
(0.941) is higher than at the circuit the parameters came from. Bias also
improved out-of-sample at Barcelona (−9.5% to −7.0%), which is the opposite
of what overfitting produces.

### What is NOT validated

**Straight-line performance.** Terminal speed error grows from +6.5 km/h at
Barcelona to +30.7 km/h at Silverstone. The cause is visible in the velocity
trace: the simulator flat-tops at its terminal speed on Silverstone's long
straights, while the real car peaks 30 km/h lower. Barcelona masked this
because its main straight ends at a braking point before the terminal speed
binds; Silverstone's straights are long enough to expose it.

CdA is therefore too low, confirmed independently at two circuits. It remains
the model's least constrained parameter. Identifying it from straight-line
acceleration was attempted and REJECTED by its own separation check: the fit
returned an ICE power of −4 kW with r² = 0.19, because it assumed the car
deploys the MGU-K regulatory ceiling continuously. It does not - this
project's own DP optimiser shows a legal qualifying lap deploys only ~2.2 MJ
of a 4 MJ store. The true deployment schedule is not observable from public
telemetry, so drag and power cannot be separated from it. That is a data
limitation, not a modelling one, and the honest response is to leave CdA
tagged [EST] and say so.

The −5.18% lap delta at Silverstone is inflated by this straight-line error
and should not be read as cornering accuracy.

---

## How the model got here — the diagnostic chain

This section exists because the debugging is the engineering. Each failure
below was caught by a physical invariant, not by inspection luck.

**1. Estimated geometry failed physically.** The first validation run was
9.7 s SLOWER than the real lap while being faster at every point on track —
only possible if the track model was wrong. The hand-estimated geometry had
no corner tighter than R = 80 m; reality is nearer 43 m.

**2. GPS curvature is noise-limited.** Curvature is a second derivative;
with pooled FastF1 position samples (~1.6 m spacing, ~1.5 m scatter) a spline
fit returned a 45 m circle as 2.6 m. Fixed with two-stage smoothing: the path
is pre-smoothed before its arc length is measured (the raw random walk
inflates a 283 m circle to kilometres), then Savitzky–Golay derivatives are
taken with a window specified in metres.

**3. The curvature scale is certified by an exact invariant.** For any simple
closed circuit the heading integral is exactly ±2π. The extracted centreline
reads −1.006 — Barcelona clockwise, correct to under 1%.

**4. The two distance axes were misaligned.** Telemetry `Distance` and
centreline arc length have independent zeros; the ~23 m offset put measured
straight-line speeds onto corners, making 25% of the lap appear physically
impossible. Found and corrected by circular cross-correlation of −v against
|κ| — the same failure class, and cure, as a timing skew between two sensors.

**5. Tyre and aero were identified from data, not assumed.** At a
grip-limited apex, a_y = μg + (μρ·SCz/2m)·v² — linear in v², so a fit over
apexes yields both parameters. 63 apexes pooled from the ten fastest laps,
pedal-gated (brake released everywhere; throttle gated only below 160 km/h,
because fast corners are driven at sustained throttle while fully laterally
limited), with 2σ outlier rejection:

    μ (effective) = 1.881,  SCz = 3.45 m²,  r² = 0.906

SCz agrees within 2% with the independent route (FIA −30% target applied to
a physics-derived 2025 baseline of 5.02 m²). The tyre figure sits at the top
of the published 1.4–2.0 band, consistent with qualifying softs on a
rubbered track, and supersedes an older-generation forum figure of 1.6.

**6. Two impossible-result bugs caught by the same invariant.** Twice during
development the four-wheel model came out FASTER than point mass — impossible,
since load transfer only ever costs grip. Both times the cause was a
parameter existing in two places (aero, then tyre) and being updated in one.
The sync helper now covers both, with the failure mode documented in it.

---

## Model results (estimated Barcelona geometry, identified parameters)

| Effect | Δ lap time |
|---|---|
| 2025 → 2026 regulatory delta | 2026 slower by ~3.7 s |
| Lateral load transfer (point mass → four wheel) | +2.6 s at k = 0.1215 |
| Overtake profile vs baseline (LTCS correction) | −0.32 s |
| Energy constraint (unconstrained → DP-optimised legal lap) | ~+1.5 s |

The energy result is the project's original motivation: an unconstrained
deployment schedule consumes ~10 MJ against a 4 MJ store — not suboptimal but
infeasible. The DP produces a schedule with state of charge closing to
< 0.1 MJ over the lap, i.e. genuinely repeatable.

---

## Regulations

Implemented from primary source: FIA 2026 F1 Regulations, Section C [Technical]
Issue 19 and Section B [Sporting] Issue 07, both dated 25 June 2026.

| Article | Constraint |
|---|---|
| C5.2.7 | Absolute ERS-K DC power ≤ 350 kW, both directions |
| C5.2.8 | Propulsion power ≤ speed-dependent profile (three limbs) |
| C5.2.9 | Max minus min state of charge ≤ 4 MJ on track |
| C5.2.10 | Recharge ≤ 8.5 MJ per lap; reducible to 7 MJ; ≥ 5 MJ floor for Sprint Qualifying and Qualifying; +0.5 MJ per B7.2 |
| C5.2.11 | MGU-K torque ≤ 500 Nm at the crank |
| B7.1.2(a) | Active aero not proximity-gated |
| B7.2.2(a), B7.2.3(b) | Overtake enabled and **activated at all times** in an LTCS |
| B7.2.3(c) | Overtake proximity-gated in a TTCS |

Three details that are easy to get wrong and that most secondary sources report
incorrectly:

**4 MJ is a state-of-charge window, not a per-lap budget.** The battery stores
4 MJ but up to 8.5 MJ may be recharged per lap, so it cycles roughly twice per
lap. State of charge is a continuously integrated state.

**The speed taper applies to propulsion only.** C5.2.8 begins "the electrical DC
power of the ERS-K used to propel the car". Harvest is bounded by the flat
350 kW absolute limit in C5.2.7 with no speed dependence. Consequence: above
345 km/h the car cannot deploy at all but can still harvest at full power, which
is why superclipping clusters at the end of straights.

**A qualifying lap runs the Overtake power profile.** Qualifying is an LTCS, and
in an LTCS Overtake is activated at all times with no proximity condition. Using
the baseline profile for a single-lap simulation is a regulatory error worth
0.181 s here.

Per-event values — the Detection Gap, Activation Zone positions, circuit-specific
adjustments to the power curves — are published by the FIA at least four weeks
before each Competition (B7.2.1(b)). They are configuration inputs in this model,
not hard-coded constants.

---

## Estimated parameters

Every value below is an estimate. F1 hardpoints, roll centre heights, roll
rates, CoG height, aero maps and Pirelli tyre data are not public.

| Parameter | Value used | Basis |
|---|---|---|
| Mass | 768 kg | C4.1 minimum |
| Wheelbase | 3.40 m | C2.3.3 maximum |
| Weight distribution | 45.5 % front | estimated |
| CoG height | 0.28 m | estimated |
| Roll stiffness | 180 / 140 kNm/rad | estimated |
| Roll centre heights | 30 / 60 mm | estimated |
| ClA, CdA (Corner Mode) | 2.40 / 0.90 m² | estimated |
| ClA, CdA (Straight Mode) | 1.35 / 0.55 m² | estimated |
| Aero balance | 44 % front | estimated |
| μ₀ | 1.95 at 2 kN | [IDENT] apex regression, r²=0.906 |
| Load sensitivity k | swept 0.05–0.20 | not public |
| Track curvature | published circuit geometry | estimated; replaceable with GPS-derived via `data/fastf1_track.py` |

Absolute lap times are therefore indicative. The comparative results — the
regulatory delta, the cost of load transfer, the value of the energy constraint —
are more robust than the absolute numbers, because parameter errors largely
cancel between configurations run on the same parameter set.

---

## Known limitations

**Dominant sensitivity: track geometry extraction.**

Two data-processing choices move the simulated lap time more than any vehicle
parameter does, and both were quantified by accident during validation:

| Choice | Effect on lap time |
|---|---|
| Curvature filter window, 30 m vs 55 m | ~3.0 s |
| Centreline from FP3 laps vs Qualifying laps | ~2.6 s |

For comparison, lateral load transfer is worth +2.6 s and the entire energy
constraint +1.5 s. The highest-value further work on this model is therefore
NOT refining the vehicle physics - it is reducing geometry uncertainty, by
pooling more laps or by using a surveyed track map instead of GPS-derived
curvature.

The window is now defined once (`DEFAULT_SMOOTH_M = 55`), recorded in each
centreline's metadata, and `validate.py` warns loudly if a centreline was
built at a different value, because runs at different windows are not
comparable.

**Not modelled, and material:**

- **C5.12 power demand ramp constraints.** Once the car enters a power-limited
  state the demand trajectory is ratcheted and rate-bounded: a first step down of
  at most 150 kW held for one second, no increase except via Boost, and reduction
  at 50 or 100 kW/s. The trigger condition is specified in FIA-F1-DOC-058, which
  is not public. This solver treats MGU-K power as independently selectable per
  segment, so predicted lap times are optimistic by the time lost to the mandated
  ramp.
- **Fixed racing line.** The trajectory is an input, not an output. A trajectory
  optimiser would find a faster line.
- **No tyre thermal model.** μ₀ is a constant; there is no warm-up, no
  degradation, no track temperature dependence.
- **Quasi-steady-state.** No roll, pitch or yaw transients; no damper forces.
- **Static roll centres.** Roll centres migrate with roll and heave. The model
  accepts a z_RC(roll, ride height) map via `set_roll_centre_map()` but is
  currently run with constants.

**Simplifying assumptions:**

- The ICE is assumed at maximum permitted fuel flow on all non-braking segments,
  so the only control variable is MGU-K power. Justified because fuel is not the
  binding constraint in 2026 — electrical energy is — but it is an assumption,
  not a derivation.
- No suspension compliance; links are rigid.
- No camber or toe effects on grip.
- Activation Zone positions are estimated from the straight positions in the
  track model, not taken from the FIA pre-event document.

---

## Structure

```
data/barcelona_track.py    track geometry, SIGNED curvature (estimated)
data/fastf1_track.py       GPS centreline reconstruction, measured curvature
extract_telemetry.py       telemetry download and export (run with network access)
data/vehicle_params.py     2025 and 2026 parameter sets
vehicle/four_wheel.py      Tier 2 load transfer and grip model
vehicle/f1_2026_params.py  F1 geometry, aero and tyre parameters
energy/regs_2026.py        FIA deployment profiles, Activation Zones
energy/deployment_dp.py    Layer 2 DP optimiser
solver_v2.py               lap-periodic three-pass QSS solver
validate.py                validation harness against measured telemetry
```

Curvature sign convention is SAE: positive is a left turn. The sign is
load-bearing — it determines which side of the car the lateral load transfers to,
and stripping it reduces the four-wheel model to an expensive point mass.

`vehicle/four_wheel.py` is deliberately geometry-agnostic. It takes a
`VehicleGeometry`, an `AeroModel` and a `TyreModel` and knows nothing about which
car it describes.

---

## References

- Milliken, W.F. and Milliken, D.L. (1995). *Race Car Vehicle Dynamics*. SAE International.
- Genta, G. and Morello, L. (2009). *The Automotive Chassis*, Vol. 2.
- FIA 2026 Formula 1 Regulations, Sections B and C, Issue 07 / Issue 19, 25 June 2026.
