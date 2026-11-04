"""Uplift learners and honest evaluation, with no database, plotting or file dependencies.

S/T/X learners on gradient boosting, inverse-propensity gain curves (Qini style) for scoring on randomised data, and a
hold-out profit cut-off that is picked on one half of the users and scored on the other.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from src.config import COST_PER_TREATED, VALUE_PER_ORDER

SEED = 2026


def _hgb(cat_mask, loss="poisson", seed=SEED):
    return HistGradientBoostingRegressor(
        loss=loss, learning_rate=0.05, max_iter=150, max_depth=4, min_samples_leaf=200,
        l2_regularization=1.0, categorical_features=cat_mask, random_state=seed)


def fit_predict(kind: str, Xtr, ytr, ttr, Xte, cat_mask, seed: int = SEED) -> np.ndarray:
    """Return predicted uplift (effect of treatment on expected outcome) for the rows of Xte."""
    if kind == "S":
        m = _hgb([*cat_mask, False], seed=seed).fit(np.c_[Xtr, ttr], ytr)
        return m.predict(np.c_[Xte, np.ones(len(Xte))]) - m.predict(np.c_[Xte, np.zeros(len(Xte))])
    m1 = _hgb(cat_mask, seed=seed).fit(Xtr[ttr == 1], ytr[ttr == 1])
    m0 = _hgb(cat_mask, seed=seed).fit(Xtr[ttr == 0], ytr[ttr == 0])
    if kind == "T":
        return m1.predict(Xte) - m0.predict(Xte)
    if kind == "X":
        # impute each user's missing counterfactual, then regress the imputed effects
        d1 = ytr[ttr == 1] - m0.predict(Xtr[ttr == 1])
        d0 = m1.predict(Xtr[ttr == 0]) - ytr[ttr == 0]
        t1 = _hgb(cat_mask, "squared_error", seed).fit(Xtr[ttr == 1], d1)
        t0 = _hgb(cat_mask, "squared_error", seed).fit(Xtr[ttr == 0], d0)
        g = ttr.mean()                                   # propensity = share treated
        return g * t0.predict(Xte) + (1 - g) * t1.predict(Xte)
    raise ValueError(kind)


def ipw_effect(y, t) -> np.ndarray:
    """Per-user unbiased estimate of the treatment effect (randomised experiment, known propensity)."""
    p = t.mean()
    return y * (t / p - (1 - t) / (1 - p))


def gain_curve(score, tau_hat) -> np.ndarray:
    """Cumulative estimated incremental outcome when treating the top-k users by score."""
    order = np.argsort(-score, kind="stable")
    return np.cumsum(tau_hat[order])


def qini_area(score, tau_hat) -> float:
    """Incremental outcome per 1000 users gained over random targeting, averaged across budgets."""
    G = gain_curve(score, tau_hat)
    n = len(G)
    lift_over_random = G - (np.arange(1, n + 1) / n) * G[-1]
    return float(lift_over_random.mean() * 1000 / n)


def holdout_profit(score: np.ndarray, y: np.ndarray, t: np.ndarray, rng: np.random.Generator,
                   value: float = VALUE_PER_ORDER, cost: float = COST_PER_TREATED) -> dict:
    """Pick the best cut-off on half A, score it on half B. All profits are per 1,000 users."""
    n = len(score)
    perm = rng.permutation(n)
    a, b = perm[: n // 2], perm[n // 2:]
    tau_a, tau_b = ipw_effect(y[a], t[a]), ipw_effect(y[b], t[b])

    order_a = np.argsort(-score[a], kind="stable")
    k = np.arange(1, len(a) + 1)
    profit_a = value * np.cumsum(tau_a[order_a]) - cost * k
    k_star = int(np.argmax(profit_a)) + 1
    threshold = score[a][order_a][k_star - 1]                       # treat users scoring at or above this

    treated_b = score[b] >= threshold
    profit_b = value * tau_b[treated_b].sum() - cost * treated_b.sum()
    per1000 = lambda x, m: float(x / m * 1000)                      # noqa: E731
    return {"share_treated_chosen_on_A": k_star / len(a),
            "share_treated_on_B": float(treated_b.mean()),
            "profit_in_sample": per1000(profit_a[k_star - 1], len(a)),
            "profit_holdout": per1000(profit_b, len(b)),
            "treat_everyone_holdout": per1000(value * tau_b.sum() - cost * len(b), len(b))}
