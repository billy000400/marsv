"""Strict validity criteria + summary metrics for a raw d(t) curve.

The plan requires that undefined or non-monotone curves are SHOWN, not forced into the correlation,
and that invalid-curve rates are reported by JSD bin. The first implementation only looked for the
first upward 0.1 / 0.9 crossing, so a curve that wandered back down, or crossed a level several
times, was silently accepted. This module makes the three criteria explicit:

  V1 span      -- d(0) <= 0.1 and d(1) >= 0.9, so both levels are actually attained;
  V2 monotone  -- max backslide  max_t ( max_{s<t} d(s) - d(t) )  <= MONO_TOL;
  V3 single    -- the curve crosses 0.1 exactly once and 0.9 exactly once (either direction).

A curve failing any criterion gets w = NaN and is excluded from the correlations but kept in the
figures and in the reported invalid rate.

Two summaries per valid curve:
  w  = t(d=0.9) - t(d=0.1)                     transition width, smaller = sharper;
  E  = edge drift, how far d moves away from its endpoint value inside the outer 20% of the path.
       E ~ 0 means genuinely flat endpoint regions (a plateau); a straight line d(t) = t gives
       E = E_LINEAR ~ 0.18, i.e. no plateau at all.
"""
import numpy as np

MONO_TOL = 0.02   # allowed backslide before a curve counts as non-monotone
EDGE = 0.2        # outer fraction of the path used for the edge-drift metric


def _cross_count(d, level):
    """Number of times the curve crosses `level`, in either direction."""
    lo = (d[:-1] < level) & (d[1:] >= level)
    hi = (d[:-1] > level) & (d[1:] <= level)
    return int(lo.sum() + hi.sum())


def _first_up(d, grid, level):
    """Position of the first upward crossing of `level`, linearly interpolated."""
    for i in range(len(d) - 1):
        if d[i] <= level <= d[i + 1] and d[i + 1] > d[i]:
            f = (level - d[i]) / (d[i + 1] - d[i])
            return float(grid[i] + f * (grid[i + 1] - grid[i]))
    return None


def edge_drift(d, grid):
    """Mean rise above d(0) in the first 20% of the path + mean fall below d(1) in the last 20%."""
    g = np.asarray(grid, float)
    lo = g <= EDGE
    hi = g >= 1 - EDGE
    return float(np.mean(d[lo] - d[0]) + np.mean(d[-1] - d[hi]))


def metrics(d, grid):
    """Validity flags and summaries for one raw curve."""
    d = np.asarray(d, dtype=float)
    backslide = float(np.max(np.maximum.accumulate(d) - d))
    n10, n90 = _cross_count(d, 0.1), _cross_count(d, 0.9)
    span = bool(d[0] <= 0.1 and d[-1] >= 0.9)
    mono = bool(backslide <= MONO_TOL)
    single = bool(n10 == 1 and n90 == 1)
    valid = span and mono and single
    t10 = _first_up(d, grid, 0.1)
    t90 = _first_up(d, grid, 0.9)
    w = float(t90 - t10) if (valid and t10 is not None and t90 is not None and t90 >= t10) else float("nan")
    return dict(valid=valid, span=span, mono=mono, single=single, backslide=backslide,
                n_cross_10=n10, n_cross_90=n90, w=w, edge_drift=edge_drift(d, grid))


E_LINEAR = None  # filled on import: edge drift of the no-plateau reference curve d(t) = t
_g = np.linspace(0.0, 1.0, 50)
E_LINEAR = edge_drift(_g, _g)
