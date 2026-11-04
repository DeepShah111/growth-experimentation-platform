"""Who should get the treatment? Cross-fitted uplift scores on any randomised arm-vs-control table, plus a profit view
whose value and cost can be changed instantly (the expensive part, scoring, is done once)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.engine.uplift_core import SEED, fit_predict, gain_curve, holdout_profit, ipw_effect, qini_area


def feature_matrix(df: pd.DataFrame, features, categorical=()) -> tuple[np.ndarray, list[bool]]:
    """Numeric matrix + categorical mask. Categoricals become integer codes (what the boosted trees expect)."""
    cols = []
    for c in features:
        s = df[c]
        cols.append(s.astype("category").cat.codes.astype(float) if c in categorical else s.astype(float).fillna(s.astype(float).median()))
    return np.column_stack(cols), [c in categorical for c in features]


def score_uplift(df: pd.DataFrame, variant: str, treat_arm: str, control: str, outcome: str, features, categorical=(),
                 kinds: str = "STX", n_splits: int = 4, seed: int = SEED, max_rows: int | None = 60_000) -> dict:
    """Cross-fitted uplift scores: every user is scored by a model that never saw them. Returns scores + IPW effects."""
    sub = df[df[variant].astype(str).isin([control, treat_arm])]
    if max_rows and len(sub) > max_rows:
        sub = sub.sample(n=max_rows, random_state=seed)
    sub = sub.reset_index(drop=True)
    t = (sub[variant].astype(str) == treat_arm).to_numpy().astype(int)
    y = sub[outcome].to_numpy(float)
    if (y < 0).any():
        raise ValueError("the targeting models need a non-negative outcome (counts, 0/1 flags, spend)")
    if t.min() == t.max():
        raise ValueError("need users in both the treated and the control arm")
    X, cat_mask = feature_matrix(sub, features, categorical)
    scores = {k: np.zeros(len(sub)) for k in kinds}
    for tr, te in KFold(n_splits, shuffle=True, random_state=seed).split(X):
        for k in kinds:
            scores[k][te] = fit_predict(k, X[tr], y[tr], t[tr], X[te], cat_mask, seed)
    out = pd.DataFrame({f"{k}-learner": v for k, v in scores.items()})
    out["random"] = np.random.default_rng(seed).random(len(sub))
    tau = ipw_effect(y, t)
    qini = {c: qini_area(out[c].to_numpy(), tau) for c in out}
    return {"scores": out, "tau": tau, "y": y, "t": t, "n": len(sub), "ate": float(tau.mean()), "qini": qini,
            "features": list(features), "frame": sub}


def profit_curve(score: np.ndarray, tau: np.ndarray, value: float, cost: float) -> pd.DataFrame:
    """Estimated profit of treating the top-k users by score, for every k: value x incremental outcome - cost x treated."""
    n = len(score)
    G = gain_curve(score, tau)
    k = np.arange(1, n + 1)
    return pd.DataFrame({"fraction_treated": k / n, "profit": value * G - cost * k, "incremental": G})


def thin(curve: pd.DataFrame, step: int = 100) -> pd.DataFrame:
    """Thin a long curve for plotting but keep the exact peak and the treat-everyone end point."""
    n = len(curve)
    keep = np.unique(np.r_[np.arange(0, n, step), int(curve.profit.to_numpy().argmax()), n - 1])
    return curve.iloc[keep].reset_index(drop=True)


def policy_summary(res: dict, model: str, value: float, cost: float, seed: int = 0) -> dict:
    """Best reach in-sample, the honest hold-out profit, and the treat-everyone baseline (all per 1,000 users)."""
    s = res["scores"][model].to_numpy()
    c = profit_curve(s, res["tau"], value, cost)
    best = int(c.profit.to_numpy().argmax())
    h = holdout_profit(s, res["y"], res["t"], np.random.default_rng(seed), value=value, cost=cost)
    n = res["n"]
    return {"best_share": float(c.fraction_treated.iloc[best]), "profit_best_in_sample": float(c.profit.iloc[best] / n * 1000),
            "profit_treat_everyone": float(c.profit.iloc[-1] / n * 1000), "holdout": h}


def qini_table(res: dict, reps: int = 100, seed: int = 7) -> pd.DataFrame:
    """Qini area per model with bootstrap intervals, and the paired difference vs random targeting.

    A model only 'works' if the difference interval excludes zero; with heavy-tailed outcomes it often does not.
    """
    scores, tau = res["scores"], res["tau"]
    rng = np.random.default_rng(seed)
    n = len(tau)
    boots = {c: np.empty(reps) for c in scores}
    arr = {c: scores[c].to_numpy() for c in scores}
    for i in range(reps):
        idx = rng.integers(0, n, n)
        for c in scores:
            boots[c][i] = qini_area(arr[c][idx], tau[idx])
    rows = []
    for c in scores:
        lo, hi = np.percentile(boots[c], [2.5, 97.5])
        dlo, dhi = np.percentile(boots[c] - boots["random"], [2.5, 97.5])
        rows.append({"model": c, "qini_per_1000_users": res["qini"][c], "ci_low": lo, "ci_high": hi,
                     "vs_random_low": dlo, "vs_random_high": dhi,
                     "beats_random": bool(c != "random" and dlo > 0)})
    return pd.DataFrame(rows).sort_values("qini_per_1000_users", ascending=False).reset_index(drop=True)
