"""Pre-experiment planning: how many users, how many days, and what you could detect with the traffic you have.

Everything here is a closed-form normal approximation (the same maths the project validates by simulation in
src.experiments.studies), so a slider can move it instantly.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from src.experiments.stats_utils import analytic_power, mde_rel, required_n_per_arm

Z = stats.norm.ppf


def n_per_arm_binary(p0: float, rel_lift: float, alpha: float = 0.05, power: float = 0.8, rho: float = 0.0) -> float:
    """Users per arm to detect p0 -> p0*(1+rel_lift) with a two-sided test. rho = covariate correlation (CUPED)."""
    p1 = p0 * (1 + rel_lift)
    if not (0 < p0 < 1) or not (0 < p1 < 1):
        raise ValueError("baseline rate and the lifted rate must both be strictly between 0 and 1")
    var = p0 * (1 - p0) + p1 * (1 - p1)
    return float((Z(1 - alpha / 2) + Z(power)) ** 2 * var * (1 - rho**2) / (p1 - p0) ** 2)


def plan(kind: str, baseline: float, rel_lift: float, sd: float | None = None, alpha: float = 0.05, power: float = 0.8,
         rho: float = 0.0, daily_users: float | None = None, n_arms: int = 2, share_in_test: float = 1.0) -> dict:
    """kind = 'rate' (baseline is a conversion rate) or 'mean' (baseline is a mean per user, needs sd).

    With more than two arms, alpha is Bonferroni-split across the treatment-vs-control comparisons (conservative).
    """
    comparisons = max(1, n_arms - 1)
    a = alpha / comparisons
    if kind == "rate":
        n = n_per_arm_binary(baseline, rel_lift, a, power, rho)
        sd_eff = math.sqrt(baseline * (1 - baseline))
    elif kind == "mean":
        if not sd or sd <= 0:
            raise ValueError("a mean metric needs its standard deviation")
        n = required_n_per_arm(rel_lift, baseline, sd, a, power, rho)
        sd_eff = sd
    else:
        raise ValueError("kind must be 'rate' or 'mean'")
    n = math.ceil(n)
    total = n * n_arms
    out = {"n_per_arm": n, "total_users": total, "alpha_used": a, "sd": sd_eff, "days": None, "weeks": None}
    if daily_users:
        days = math.ceil(total / (daily_users * share_in_test))
        out["days"] = days
        out["weeks"] = math.ceil(days / 7)                       # run whole weeks: weekday/weekend behaviour differs
    return out


def mde_for_n(kind: str, baseline: float, n_per_arm: float, sd: float | None = None, alpha: float = 0.05,
              power: float = 0.8, rho: float = 0.0) -> float:
    """Smallest relative lift detectable with n users per arm."""
    if kind == "rate":
        sd = math.sqrt(baseline * (1 - baseline))
    return float(mde_rel(n_per_arm, baseline, sd, alpha, power, rho))


def power_curve(kind: str, baseline: float, n_per_arm: float, lifts: np.ndarray, sd: float | None = None,
                alpha: float = 0.05, rho: float = 0.0) -> pd.DataFrame:
    """Power as a function of the true relative lift, at fixed n per arm."""
    if kind == "rate":
        sd = math.sqrt(baseline * (1 - baseline))
    return pd.DataFrame({"true_lift": lifts,
                         "power": [analytic_power(float(x), n_per_arm, baseline, sd, alpha, rho) for x in lifts]})


def days_curve(kind: str, baseline: float, lifts: np.ndarray, daily_users: float, sd: float | None = None, alpha: float = 0.05,
               power: float = 0.8, rho: float = 0.0, n_arms: int = 2) -> pd.DataFrame:
    """Days needed as a function of the lift you want to detect: the trade-off a product manager actually argues about."""
    rows = []
    for x in lifts:
        try:
            p = plan(kind, baseline, float(x), sd, alpha, power, rho, daily_users, n_arms)
            rows.append({"detectable_lift": float(x), "days": p["days"], "n_per_arm": p["n_per_arm"]})
        except ValueError:
            continue
    return pd.DataFrame(rows)
