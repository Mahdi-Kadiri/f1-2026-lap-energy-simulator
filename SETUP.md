# Setup and run guide

Windows, VS Code. Everything in section 1 is offline — no internet needed.

## 0. Extract the zip first

In File Explorer you are currently browsing *inside* the zip. Nothing there
can run. Click **Extract all**, and extract to a path with no spaces, e.g.

    C:\Users\<you>\Downloads\f1_2026_simulator

You should end up with a folder containing `run_analysis.py`, not a zip icon.

## 1. Open in VS Code and trust the folder

1. VS Code → **File → Open Folder** → pick the extracted folder
2. A prompt appears: **"Do you trust the authors of the files in this folder?"**
   → click **Yes, I trust the authors**

Restricted Mode blocks the Python extension from running anything. The banner
at the top of your window must disappear before step 3 will work.

## 2. Install the dependencies

Open the terminal in VS Code: **Terminal → New Terminal** (or Ctrl+Shift+`).
Then:

    pip install -r requirements.txt

If `pip` is not recognised, try `python -m pip install -r requirements.txt`.
If Python itself is missing, install it from python.org and tick
**"Add Python to PATH"** during setup.

## 3. Run the analysis

    python run_analysis.py

Takes about 40 seconds. Prints five sections and writes four plots to
`outputs/`. Start here — if this works, the install is correct.

Faster variant while you are experimenting:

    python run_analysis.py --quick        # skips the sweep and the DP, ~5 s
    python run_analysis.py --ds 20        # coarser track spacing

## 4. Telemetry and validation (needs internet)

Only after step 3 works.

    python extract_telemetry.py --list --year 2026
    python extract_telemetry.py --year 2026 --event Barcelona --session Q
    python validate.py --year 2026 --event Barcelona --session Q --plot

The first call downloads and caches the session, which is slow the first time.

**Note on the event name:** the 2026 calendar has both a *Barcelona* Grand Prix
(14 June, Circuit de Barcelona-Catalunya) and a *Spanish* Grand Prix (13
September, Madrid). Use `--event Barcelona`. Passing `Spain` gets you Madrid.

## What each file does

| File | Purpose |
|---|---|
| `run_analysis.py` | **start here** — runs everything offline |
| `extract_telemetry.py` | downloads real telemetry (needs internet) |
| `validate.py` | compares simulation against measured telemetry |
| `solver_v2.py` | lap-periodic three-pass QSS solver |
| `vehicle/four_wheel.py` | four-wheel load transfer and grip model |
| `vehicle/f1_2026_params.py` | 2025 and 2026 car parameters |
| `energy/deployment_dp.py` | DP energy optimiser |
| `energy/regs_2026.py` | FIA deployment profiles and Activation Zones |
| `data/barcelona_track.py` | track geometry (estimated curvature) |
| `data/fastf1_track.py` | GPS centreline reconstruction |

## Common problems

**`ModuleNotFoundError: No module named 'data'`**
You are running from the wrong directory. The terminal prompt must show the
project folder. Use `cd` to get there, or reopen the folder in VS Code.

**`pip` not recognised**
Use `python -m pip` instead. On some Windows setups it is `py -m pip`.

**Plots do not appear**
They are not shown on screen — they are written to `outputs/` as PNG files.

**FastF1 fails with a connection error**
Either no internet, or the session has not happened yet. Check with
`python extract_telemetry.py --list --year 2026`.

## Where to change things

Car parameters: `vehicle/f1_2026_params.py`. Every value there is an estimate —
F1 hardpoints, roll rates, CoG height and tyre data are not public. Changing
`load_sensitivity_k` in `TYRE_2026` has the largest effect on the results;
that is why the sweep exists.
