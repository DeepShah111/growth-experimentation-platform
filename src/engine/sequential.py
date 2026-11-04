"""The peeking problem, simulated live: how many false wins do you get when you look at an A/A test repeatedly?"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.experiments.stats_utils import obrien_fleming_bounds


def peeking_study(n_looks: int, alpha: float = 0.05, n_sims: int = 40_000, seed: int = 0) -> pd.DataFrame:
    """False-positive rate of several stopping rules on tests where the TRUE effect is zero.

    Each simulated test accumulates equal-sized batches of standard-normal noise; at look k the z statistic is the
    partial sum divided by sqrt(k). A rule declares a win if |z| crosses its boundary at any look it is allowed to use.
    """
    if n_looks < 1:
        raise ValueError("n_looks must be at least 1")
    rng = np.random.default_rng(seed)
    k = np.arange(1, n_looks + 1)
    z = np.abs(np.cumsum(rng.standard_normal((n_sims, n_looks)), axis=1)) / np.sqrt(k)
    z_naive = stats.norm.ppf(1 - alpha / 2)
    z_bonf = stats.norm.ppf(1 - alpha / (2 * n_looks))
    rows = [("One look at the end (correct)", float((z[:, -1] > z_naive).mean()), z_naive),
            (f"Peek {n_looks}x, stop at first p<{alpha:g} (naive)", float((z > z_naive).any(axis=1).mean()), z_naive),
            (f"Peek {n_looks}x, Bonferroni boundary", float((z > z_bonf).any(axis=1).mean()), z_bonf)]
    if n_looks >= 2:
        obf = obrien_fleming_bounds(n_looks, alpha, n_sims=min(n_sims, 60_000), seed=seed + 1)
        rows.append((f"Peek {n_looks}x, O'Brien-Fleming boundary", float((z > obf).any(axis=1).mean()), float(obf[-1])))
    out = pd.DataFrame(rows, columns=["strategy", "false_positive_rate", "final_boundary_z"])
    out["target"] = alpha
    return out


def fpr_by_looks(max_looks: int = 30, alpha: float = 0.05, n_sims: int = 20_000, seed: int = 0) -> pd.DataFrame:
    """Naive false-positive rate as a function of the number of equally spaced looks."""
    rng = np.random.default_rng(seed)
    inc = rng.standard_normal((n_sims, max_looks))
    ks = sorted({1, 2, 3, 5, 7, 10, 14, 20, max_looks} & set(range(1, max_looks + 1)))
    z_naive = stats.norm.ppf(1 - alpha / 2)
    rows = []
    cum = np.abs(np.cumsum(inc, axis=1)) / np.sqrt(np.arange(1, max_looks + 1))
    for k in ks:
        idx = np.unique(np.round(np.linspace(max_looks / k, max_looks, k)).astype(int) - 1)
        rows.append({"n_looks": k, "false_positive_rate": float((cum[:, idx] > z_naive).any(axis=1).mean())})
    return pd.DataFrame(rows)
