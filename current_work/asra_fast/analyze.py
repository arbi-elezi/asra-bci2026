"""Analysis utilities: Pareto frontier, frontier AUC (2D dominated hypervolume)"""
from __future__ import annotations
import numpy as np


def pareto_frontier(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Upper-left Pareto frontier: minimize CR (x), maximize return (y)."""
    pts = sorted(points, key=lambda p: (p[0], -p[1]))
    front, best = [], -np.inf
    for cr, ret in pts:
        if ret > best:
            front.append((cr, ret)); best = ret
    return front


def frontier_auc(points: list[tuple[float, float]], cr_grid: np.ndarray,
                 floor: float | None = None) -> float:
    """Area under the best-perf-achievable-at-CR<=x curve, on a common CR grid.

    For each CR level x, take the max perf among points with cr <= x. A CR level a method
    CANNOT reach (no point with cr <= x) contributes `floor` -- a GLOBAL floor shared across
    methods, so a method is properly penalized where it offers no operating point. (Using the
    method's own min would unfairly credit it in regions it never reaches.) Higher = better.
    """
    pts = sorted(points, key=lambda p: p[0])
    if floor is None:
        floor = min((p[1] for p in pts), default=0.0)
    ys, best, j = [], -np.inf, 0
    for x in cr_grid:
        while j < len(pts) and pts[j][0] <= x:
            best = max(best, pts[j][1]); j += 1
        ys.append(best if best > -np.inf else floor)   # no reachable point => global floor
    return float(np.trapz(np.array(ys, dtype=np.float64), cr_grid))


def paired_bootstrap_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 10000,
                          seed: int = 42) -> dict:
    """CI on mean(a - b), paired by index. Excludes zero => significant."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    d = a - b
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # bootstrap percentile p-values (floored at 1/(n_boot+1) so they are never exactly 0)
    eps = 1.0 / (n_boot + 1)
    frac_le = max(float(np.mean(boot <= 0.0)), eps)   # mass at/below 0
    frac_ge = max(float(np.mean(boot >= 0.0)), eps)   # mass at/above 0
    p_two = min(1.0, 2.0 * min(frac_le, frac_ge))     # two-sided: mean(a-b) != 0
    p_greater = frac_le                               # one-sided H1: mean(a-b) > 0 (small when clearly >0)
    return {"diff": float(d.mean()), "lo": float(lo), "hi": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0), "n": int(len(d)),
            "pvalue": float(p_two), "p_greater": float(p_greater)}


def spearman_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 10000, seed: int = 42) -> dict:
    """Bootstrap CI on Spearman rho (monotonicity of y vs x)."""
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    rho = spearmanr(x, y).correlation
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        if len(np.unique(x[i])) < 2:
            continue
        boots.append(spearmanr(x[i], y[i]).correlation)
    boots = np.array([b for b in boots if not np.isnan(b)])
    lo, hi = np.percentile(boots, [2.5, 97.5]) if len(boots) else (np.nan, np.nan)
    return {"rho": float(rho), "lo": float(lo), "hi": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0)}


def holm_bonferroni(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm-Bonferroni step-down across a family of hypotheses."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, reject_all = {}, True
    for rank, (name, p) in enumerate(items):
        thresh = alpha / (m - rank)
        rej = (p <= thresh) and reject_all
        if not rej:
            reject_all = False
        out[name] = {"p": float(p), "thresh": float(thresh), "reject": bool(rej)}
    return out
