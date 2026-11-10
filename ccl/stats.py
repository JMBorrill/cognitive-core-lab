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


# ------------------------------------------------------------------ Study 02: families as the unit

def cluster_bootstrap(y, n_boot: int = 10_000, seed: int = 0) -> np.ndarray:
    """Hierarchical bootstrap for arms scored on the same held-out families.

    y has shape (arms, replicates, families, tasks) with 0/1 entries. Each resample draws
    replicates (training seeds), then families, then tasks within each drawn family, all with
    replacement; the same draw is applied to every arm, so differences between arms are paired
    on replicate index, family and task. Returns the resampled arm means, shape (n_boot, arms)."""
    y = np.asarray(y, dtype=float)
    a, r, f, m = y.shape
    rng = np.random.default_rng(seed)
    fam = np.arange(f)[:, None]
    out = np.empty((n_boot, a))
    for b in range(n_boot):
        ri, fi, ti = rng.integers(0, r, r), rng.integers(0, f, f), rng.integers(0, m, (f, m))
        out[b] = y[:, ri][:, :, fi][:, :, fam, ti].mean(axis=(1, 2, 3))
    return out


def interval(samples, point: float, alpha: float = 0.05) -> dict:
    s = np.asarray(samples, dtype=float)
    return {"mean": float(point), "lo": float(np.quantile(s, alpha / 2)), "hi": float(np.quantile(s, 1 - alpha / 2))}


def variance_components(y) -> dict:
    """Method-of-moments components of success for one arm, on the probability scale.

    y: (replicates, families, tasks). `within` is task-to-task variance inside a
    (replicate, family) cell; `seed_family` is how much a family's success moves between
    replicates; `seed` is the replicate effect on the split mean left after that; `family_sd`
    is the spread of family means. The same estimators as scripts/study02_audit.py."""
    y = np.asarray(y, dtype=float)
    r, f, m = y.shape
    cell = y.mean(axis=2)
    within = float(np.mean(cell * (1 - cell)) * m / max(m - 1, 1))
    seed_family = float(np.mean(cell.var(axis=0, ddof=1)) - within / m)
    rep_means = cell.mean(axis=1)
    seed = float(rep_means.var(ddof=1) - max(seed_family, 0) / f - within / (f * m))
    return {"within": within, "seed_family": seed_family, "seed": seed,
            "seed_sd": float(np.sqrt(max(seed, 0.0))), "replicate_means_sd": float(rep_means.std(ddof=1)),
            "family_sd": float(cell.mean(axis=0).std(ddof=1))}


def arm_family_variance(ya, yb) -> dict:
    """How much the difference between two arms varies from family to family, after removing
    replicate and task noise. ya, yb: (replicates, families, tasks)."""
    ya, yb = np.asarray(ya, dtype=float), np.asarray(yb, dtype=float)
    diff = yb.mean(axis=(0, 2)) - ya.mean(axis=(0, 2))
    noise = 0.0
    for y in (ya, yb):
        c = variance_components(y)
        noise += max(c["seed_family"], 0) / y.shape[0] + c["within"] / (y.shape[0] * y.shape[2])
    return {"mean_diff": float(diff.mean()), "raw_sd": float(diff.std(ddof=1)),
            "arm_family": float(diff.var(ddof=1) - noise)}
