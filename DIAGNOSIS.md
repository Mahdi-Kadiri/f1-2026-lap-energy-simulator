# First validation run — what it found

Session: 2026 Barcelona Grand Prix, Qualifying. Reference lap: RUS, 74.679 s.

## Result

| | Simulated | Measured | Delta |
|---|---|---|---|
| Lap time | 77.101 s | 74.679 s | **+2.422 s** |
| Top speed | 352.2 km/h | 341.0 km/h | +11.2 |
| Minimum speed | 137.8 km/h | 103.0 km/h | +34.8 |

**The simulator was slower than reality.** For a theoretical minimum lap time
model — no driver error, no tyre thermal state, no fuel burn, a fixed racing
line — that is a failure condition, not a small error. It should be faster.

## Diagnosis

The simulator was faster at *every point on track* (35 km/h faster in the
slowest corner, 11 km/h at the top end) yet slower over the lap. That
combination is only possible if the track geometry is wrong.

It was. The hand-estimated track model had:

| | Estimated model | GPS measured |
|---|---|---|
| Lap length | 4657 m | 4592.5 m |
| Tightest corner | R = 80 m | R = 28.9 m |
| Left / right samples | 86 / 141 | 694 / 1132 |

No corner tighter than 80 m existed in the estimate. Barcelona has corners at
roughly a third of that radius. The car therefore carried speed through
corners that do not exist at that severity, and spent lap distance in
medium-speed sections that are really slow ones.

## Why this matters more than the number

This is the entire justification for the GPS pipeline. A model tuned to land
near a published pole time on estimated geometry would have looked *correct*
and been wrong for compensating reasons. Comparing distributed quantities —
corner minima, terminal speed, profile shape — is what exposed it; comparing
lap time alone would not have.

## Two fixes applied

**1. Centreline resolution.** FastF1 position data samples at roughly 4 Hz,
giving about 280 points over a 75 s lap — one every 16 m. That cannot resolve a
30 m radius corner, and resampling a single sparse lap up to 2000 points
invents detail that was never measured.

Raw samples from every lap are now POOLED rather than each lap being resampled
and averaged. Laps are sampled at different points along the track, so N laps
give roughly N times the spatial density. Verified on synthetic data: effective
spacing improved from 5.92 m to 0.59 m with 10 laps.

**2. Path discovery.** `extract_telemetry.py` wrote to
`telemetry/centreline_Barcelona_Grand_Prix_2026` while `validate.py` looked for
`data/centreline_barcelona_2026`. The centreline is now auto-discovered.

## Sector splits are not comparable

Measured splits were S1 21.849, S2 30.115, S3 22.715. The simulator reported
20.014 / 23.125 / 33.555 against its own internal sector boundaries, which are
arbitrary — the FIA does not publish sector boundaries as lap distances. The
totals are comparable; the individual splits are not. The harness now says so
rather than inviting a false comparison.

## Status

Not yet re-run against measured geometry. Until it is, no lap time from this
model should be described as validated.
