"""
Track geometry from FastF1 GPS position data.

Objective
---------
Replace hand-estimated corner radii with curvature measured from real car
position telemetry, and produce a lap-periodic centreline in track
coordinates.

Why this is not just "differentiate the GPS trace"
--------------------------------------------------
Raw position data is sampled irregularly in distance, is noisy at the metre
level, and curvature is a second derivative - so naive finite differencing
amplifies that noise by roughly (1/ds)^2 and returns garbage. Three things
are needed:

  1. Resample to uniform arc length, so the derivative step is constant.
  2. Fit a PERIODIC spline, so the lap closes and curvature is continuous
     across the start/finish line rather than showing a discontinuity there.
  3. Average across several clean laps, so uncorrelated position noise and
     line variation between laps average down.

Curvature is then computed analytically from the spline derivatives rather
than by finite differences:

    kappa = (x' y'' - y' x'') / (x'^2 + y'^2)^(3/2)

Sign convention
---------------
SAE, Z up: positive kappa is a LEFT turn. This matches the vehicle model.
FastF1 X/Y are in a track-local frame in units of 1/10 metre.

Usage
-----
    python -m data.fastf1_track --year 2026 --event Barcelona --session Q

Requires network access to livetiming.formula1.com. Run locally, not in a
sandboxed environment.
"""

from __future__ import annotations
import argparse
import json
import os
from dataclasses import dataclass, field

import math

import numpy as np

# CANONICAL curvature filter window (metres), used by every entry point.
# Defined once because lap time is materially sensitive to it: rebuilding the
# same session's centreline at 30 m instead of 55 m moved the simulated lap
# by ~3 s with identical car parameters. Any comparison between runs is
# meaningless unless they share this value.
DEFAULT_SMOOTH_M = 55.0

try:
    from scipy.interpolate import splprep, splev
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


# ------------------------------------------------------------------ result

@dataclass
class Centreline:
    """Lap-periodic reference line with measured curvature."""
    s: np.ndarray               # arc length from start/finish (m)
    x: np.ndarray               # position (m)
    y: np.ndarray
    kappa: np.ndarray           # signed curvature (1/m), +ve = left
    length_m: float
    circuit: str
    n_laps_used: int
    metadata: dict = field(default_factory=dict)

    def resample(self, ds: float):
        """Return (s, kappa) on a uniform grid of spacing ds."""
        s_new = np.arange(0.0, self.length_m, ds)
        return s_new, np.interp(s_new, self.s, self.kappa)

    def save(self, path: str) -> None:
        np.savez_compressed(path + '.npz', s=self.s, x=self.x, y=self.y,
                            kappa=self.kappa)
        with open(path + '.json', 'w') as f:
            json.dump({'circuit': self.circuit, 'length_m': self.length_m,
                       'n_laps_used': self.n_laps_used,
                       'n_samples': int(len(self.s)),
                       'metadata': self.metadata}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Centreline":
        arr = np.load(path + '.npz')
        with open(path + '.json') as f:
            meta = json.load(f)
        return cls(s=arr['s'], x=arr['x'], y=arr['y'], kappa=arr['kappa'],
                   length_m=float(meta['length_m']),
                   circuit=meta['circuit'],
                   n_laps_used=int(meta['n_laps_used']),
                   metadata=meta.get('metadata', {}))


# ------------------------------------------------------------- core maths

def resample_uniform(x, y, n_samples):
    """Resample an (x, y) path to uniform spacing in arc length."""
    dx = np.diff(x)
    dy = np.diff(y)
    seg = np.sqrt(dx * dx + dy * dy)
    s_in = np.concatenate(([0.0], np.cumsum(seg)))
    length = float(s_in[-1])
    s_new = np.linspace(0.0, length, n_samples)
    return s_new, np.interp(s_new, s_in, x), np.interp(s_new, s_in, y), length


def _estimate_scatter(x, y, window=9):
    """
    Estimate lateral position scatter (m) from local roughness.

    A racing line is smooth at the scale of a few metres, so deviation of each
    point from a short local straight fit is dominated by measurement noise
    and by variation between laps, not by genuine track curvature. Taking the
    median absolute deviation makes this robust to real corners.
    """
    n = len(x)
    if n < window + 2:
        return 1.0
    dev = []
    h = window // 2
    for i in range(h, n - h):
        xs = x[i - h:i + h + 1]
        ys = y[i - h:i + h + 1]
        # perpendicular distance of the centre point from the chord
        dx, dy = xs[-1] - xs[0], ys[-1] - ys[0]
        L = math.hypot(dx, dy)
        if L < 1e-6:
            continue
        dev.append(abs((xs[h] - xs[0]) * dy - (ys[h] - ys[0]) * dx) / L)
    if not dev:
        return 1.0
    # MAD -> sigma for a normal distribution
    return float(np.median(dev)) * 1.4826


def curvature_from_spline(x, y, n_samples=2000, smooth=None,
                          smooth_length_m=None):
    """
    Compute signed curvature from a closed path by Savitzky-Golay filtering.

    Why not fit a spline through the points
    ---------------------------------------
    Curvature is a second derivative, and it cannot be resolved at a length
    scale shorter than the position noise. FastF1 position samples pooled
    across laps arrive at roughly 1-2 m spacing with 1-2 m of lap-to-lap
    scatter, so consecutive points behave like a random walk. A spline that
    interpolates them - at any smoothing factor tested - returns curvature
    dominated by noise, reporting corners an order of magnitude tighter than
    the car ever drove. Verified: a true 45 m circle recovered as 2.6 m.

    What is done instead
    --------------------
    The path is resampled to uniform arc length, then x(s) and y(s) are each
    filtered with a Savitzky-Golay filter whose window is set in METRES. The
    filter fits a local quadratic and returns its analytic derivatives, so
    curvature comes from

        kappa = (x' y'' - y' x'') / (x'^2 + y'^2)^(3/2)

    evaluated on the smooth local fit rather than on raw differences.

    The window is a physical band limit: features shorter than roughly the
    window length are removed. It must therefore be shorter than the corners
    to be resolved and longer than the noise correlation length. For an F1
    circuit, 25-40 m works: comfortably below the arc length of a real corner,
    comfortably above GPS scatter.

    Args:
        smooth_length_m : filter window in metres (default 30)
        smooth          : accepted for backward compatibility; if given and
                          smooth_length_m is None it is interpreted as a
                          window length in metres.

    Returns s, x, y, kappa, length
    """
    from scipy.signal import savgol_filter

    keep = np.concatenate(([True], (np.diff(x) ** 2 + np.diff(y) ** 2) > 1e-9))
    x, y = np.asarray(x, float)[keep], np.asarray(y, float)[keep]
    n_raw = len(x)

    if smooth_length_m is None:
        smooth_length_m = float(smooth) if smooth else DEFAULT_SMOOTH_M

    def _wrap_savgol(a, win, deriv=0, delta=1.0):
        pad = win
        ap = np.concatenate([a[-pad:], a, a[:pad]])
        return savgol_filter(ap, win, 3, deriv=deriv, delta=delta)[pad:pad + len(a)]

    # STAGE 1 - pre-smooth in index space.
    # Arc length computed from raw noisy points is inflated by the random walk
    # the noise induces: with scatter comparable to sample spacing, a 283 m
    # circle can measure as several kilometres. Every metre-based quantity
    # downstream then becomes meaningless, so the path must be smoothed BEFORE
    # its length is measured.
    w0 = max(5, int(n_raw * 0.02) | 1)
    w0 = min(w0, (n_raw // 2) * 2 - 1)
    xs0 = _wrap_savgol(x, w0)
    ys0 = _wrap_savgol(y, w0)

    # STAGE 2 - arc length from the pre-smoothed path, then uniform resample.
    s_u, xu, yu, length = resample_uniform(xs0, ys0, n_samples)
    ds = length / n_samples

    # STAGE 3 - Savitzky-Golay derivatives with a window set in METRES.
    win = int(round(smooth_length_m / ds)) | 1
    win = max(5, min(win, (n_samples // 2) * 2 - 1))

    xs = _wrap_savgol(xu, win)
    ys = _wrap_savgol(yu, win)
    d1x = _wrap_savgol(xu, win, deriv=1, delta=ds)
    d1y = _wrap_savgol(yu, win, deriv=1, delta=ds)
    d2x = _wrap_savgol(xu, win, deriv=2, delta=ds)
    d2y = _wrap_savgol(yu, win, deriv=2, delta=ds)

    denom = np.maximum((d1x * d1x + d1y * d1y) ** 1.5, 1e-12)
    kappa = (d1x * d2y - d1y * d2x) / denom

    return s_u, xs, ys, kappa, length


def average_laps(lap_xy, n_samples=2000):
    """
    Combine several laps into one high-density reference line.

    FastF1 position data is sampled at roughly 4 Hz, which over a 75 s lap
    gives only ~300 points - about one every 15 m. That is far too sparse to
    resolve a 30 m radius corner, and resampling a single sparse lap up to
    2000 points invents detail that was never measured.

    Instead the raw samples from every lap are POOLED. Because each lap is
    sampled at slightly different points along the track, N laps give
    roughly N times the effective spatial density. The samples are sorted by
    arc length along a coarse reference line, then binned and averaged, which
    both raises resolution and averages down position noise.

    Laps whose length deviates more than 2% from the median are dropped -
    those usually indicate an off-track excursion or a bad lap splice.
    """
    lengths = []
    for x, y in lap_xy:
        _, _, _, L = resample_uniform(x, y, 200)
        lengths.append(L)
    med = float(np.median(lengths))
    keep = [i for i, L in enumerate(lengths) if abs(L - med) / med < 0.02]
    if not keep:
        keep = list(range(len(lap_xy)))

    # Coarse reference from the first kept lap, used only to assign arc length.
    s_ref, x_ref, y_ref, _ = resample_uniform(*lap_xy[keep[0]], 400)

    px, py, ps = [], [], []
    for i in keep:
        x, y = lap_xy[i]
        for xi, yi in zip(x, y):
            j = int(np.argmin((x_ref - xi) ** 2 + (y_ref - yi) ** 2))
            px.append(xi)
            py.append(yi)
            ps.append(s_ref[j])

    px = np.asarray(px)
    py = np.asarray(py)
    ps = np.asarray(ps)
    order = np.argsort(ps)
    px, py, ps = px[order], py[order], ps[order]

    # Bin to a uniform arc-length grid; empty bins are filled by interpolation.
    n_bins = min(n_samples, max(200, len(px) // 3))
    edges = np.linspace(0.0, med, n_bins + 1)
    idx = np.clip(np.digitize(ps, edges) - 1, 0, n_bins - 1)

    xb = np.full(n_bins, np.nan)
    yb = np.full(n_bins, np.nan)
    for b in range(n_bins):
        m = idx == b
        if m.any():
            xb[b] = px[m].mean()
            yb[b] = py[m].mean()

    good = ~np.isnan(xb)
    if good.sum() < 20:
        raise RuntimeError("too few position samples to build a centreline")
    centres = 0.5 * (edges[:-1] + edges[1:])
    xb = np.interp(centres, centres[good], xb[good])
    yb = np.interp(centres, centres[good], yb[good])

    return xb, yb, len(keep), med


# --------------------------------------------------------------- pipeline

def build_centreline(year: int, event: str, session: str = 'Q',
                     n_laps: int = 10, n_samples: int = 2000,
                     smooth: float | None = None,
                     cache_dir: str | None = None) -> Centreline:
    """
    Build a measured centreline from the fastest clean laps of a session.

    Requires network access to the F1 timing API.
    """
    import fastf1

    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser('~'), 'f1sim_data', 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)

    ses = fastf1.get_session(year, event, session)
    ses.load(telemetry=True, laps=True, weather=False)

    laps = ses.laps.pick_quicklaps()          # excludes in/out and slow laps
    if len(laps) == 0:
        laps = ses.laps
    laps = laps.sort_values('LapTime').head(n_laps)

    lap_xy = []
    for _, lap in laps.iterrows():
        try:
            pos = lap.get_pos_data()
            if len(pos) < 100:
                continue
            # FastF1 X/Y are in 1/10 m
            lap_xy.append((pos['X'].to_numpy() / 10.0,
                           pos['Y'].to_numpy() / 10.0))
        except Exception:
            continue

    if not lap_xy:
        raise RuntimeError("no usable position data in this session")

    raw_total = sum(len(x) for x, _ in lap_xy)
    xs, ys, n_used, med_len = average_laps(lap_xy, n_samples)
    s, xf, yf, kappa, length = curvature_from_spline(xs, ys, n_samples, smooth)

    return Centreline(
        s=s, x=xf, y=yf, kappa=kappa, length_m=length,
        circuit=str(ses.event['EventName']), n_laps_used=n_used,
        metadata={'year': year, 'session': session,
                  'median_raw_lap_length_m': round(med_len, 1),
                  'raw_position_samples_pooled': int(raw_total),
                  'effective_sample_spacing_m': round(med_len / max(raw_total, 1), 2),
                  'curvature_filter_window_m': float(smooth) if smooth else DEFAULT_SMOOTH_M,
                  'sign_convention': 'SAE, +kappa = left turn'},
    )


def elevation_gradient(year, event, session='Q', n_samples=2000,
                       cache_dir='.f1cache'):
    """
    FastF1 does not publish elevation. Returns zeros with a warning so the
    caller can substitute a circuit elevation profile if available.
    """
    import warnings
    warnings.warn("FastF1 provides no elevation channel; gradient set to zero. "
                  "Supply a measured elevation profile for gradient effects.")
    return np.zeros(n_samples)


# ------------------------------------------------------------------- main

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--year', type=int, default=2026)
    ap.add_argument('--event', default='Barcelona')
    ap.add_argument('--session', default='Q')
    ap.add_argument('--laps', type=int, default=10)
    ap.add_argument('--samples', type=int, default=2000)
    ap.add_argument('--smooth', type=float, default=None,
                    help='curvature filter window in METRES (default 30). '
                         'Must be shorter than the corners you want to '
                         'resolve and longer than the GPS scatter.')
    ap.add_argument('--out', default='data/centreline_barcelona_2026')
    args = ap.parse_args()

    cl = build_centreline(args.year, args.event, args.session,
                          args.laps, args.samples, args.smooth)
    cl.save(args.out)

    R = np.where(np.abs(cl.kappa) > 1e-4, 1.0 / np.maximum(np.abs(cl.kappa), 1e-9), np.inf)
    print(f"circuit        : {cl.circuit}")
    print(f"laps averaged  : {cl.n_laps_used}")
    print(f"lap length     : {cl.length_m:.1f} m")
    print(f"kappa range    : {cl.kappa.min():+.5f} to {cl.kappa.max():+.5f} 1/m")
    print(f"tightest corner: R = {np.min(R[np.isfinite(R)]):.1f} m")
    print(f"left / right   : {(cl.kappa > 1e-4).sum()} / {(cl.kappa < -1e-4).sum()} samples")
    print(f"saved          : {args.out}.npz / .json")
