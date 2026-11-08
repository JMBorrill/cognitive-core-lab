"""Uncertainty estimates: bootstrap CIs over tasks, hierarchical over training seeds."""

from __future__ import annotations

import math

import numpy as np


def _clean(x):
    x = np.asarray(x, dtype=float)
    return x[~np.isnan(x)]


def bootstrap_ci(values, n_boot: int = 4000, alpha: float = 0.05, seed: int = 0) -> dict:
    x = _clean(values)
    if len(x) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    return {"mean": float(x.mean()), "lo": float(np.quantile(means, alpha / 2)),
            "hi": float(np.quantile(means, 1 - alpha / 2)), "n": int(len(x))}


def hierarchical_ci(per_seed: list[list[float]], n_boot: int = 4000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Resample training seeds, then tasks within each seed. per_seed: list of per-task value lists
    (tasks aligned across seeds)."""
    arrs = [np.asarray(v, dtype=float) for v in per_seed]
    if len(arrs) == 1:
        return {**bootstrap_ci(arrs[0], n_boot, alpha, seed), "seeds": 1, "seed_means": [float(np.nanmean(arrs[0]))]}
    rng = np.random.default_rng(seed)
    n_tasks = len(arrs[0])
    stacked = np.stack(arrs)  # seeds x tasks
    out = np.empty(n_boot)
    for b in range(n_boot):
        s = rng.integers(0, len(arrs), len(arrs))
        t = rng.integers(0, n_tasks, n_tasks)
        out[b] = np.nanmean(stacked[s][:, t])
    seed_means = [float(np.nanmean(a)) for a in arrs]
    return {"mean": float(np.nanmean(stacked)), "lo": float(np.quantile(out, alpha / 2)),
            "hi": float(np.quantile(out, 1 - alpha / 2)), "n": n_tasks, "seeds": len(arrs),
            "seed_means": seed_means, "seed_sd": float(np.std(seed_means, ddof=1))}


def paired_diff(a, b, n_boot: int = 4000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Bootstrap CI for mean(b - a) over aligned tasks, plus a two-sided sign-flip permutation p-value."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    d = b[m] - a[m]
    if len(d) == 0:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "p": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    signs = rng.choice([-1.0, 1.0], (n_boot, len(d)))
    perm = np.abs((signs * d).mean(1)) >= abs(d.mean()) - 1e-12
    return {"diff": float(d.mean()), "lo": float(np.quantile(boots, alpha / 2)),
            "hi": float(np.quantile(boots, 1 - alpha / 2)), "p": float((perm.sum() + 1) / (n_boot + 1)),
            "n": int(len(d))}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)
