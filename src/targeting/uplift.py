"""Uplift modelling on exp_008 (personalised_offer).

Question: can we predict, from pre-experiment behaviour only, WHICH users the offer helps?

Method
  * Outcome   : orders during the experiment (orders_in).
  * Treatment : randomised 50/50 (so we can evaluate honestly).
  * Features  : only things known BEFORE the experiment (pre-period activity, tenure, device...).
  * Learners  : S-learner, T-learner, X-learner on gradient boosting.
  * Evaluation: 4-fold cross-fitting (every user is scored by a model that never saw them),
                then Qini-style gain curves from inverse-propensity-weighted outcomes.
  * Baselines : random targeting, and the simple rule 'new users first'.

Run:  python -m src.targeting.uplift          (needs the DuckDB warehouse; outputs go to artifacts/)
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from src.config import COST_PER_TREATED, VALUE_PER_ORDER, DB_PATH, EXPERIMENTS, ROOT, SQL_DIR

ART = ROOT / "artifacts"
SEED = 2026
NUM = ["orders_pre", "revenue_pre", "sessions_pre", "days_since_signup", "is_new"]
CAT = ["device", "channel", "city"]

# Business assumptions for the profit view (ASSUMPTIONS, not measured facts).


# ------------------------------------------------------------------------------- data
def load(exp_id: str) -> tuple[pd.DataFrame, np.ndarray]:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    d = con.execute((SQL_DIR / "marts" / "experiment_user_metrics.sql").read_text(), {"exp_id": exp_id}).df()
    truth = con.execute("SELECT * FROM answer_key.user_truth").df()
    d = d.merge(truth, on="user_id", how="left")        # kept for diagnostics, not features
    return d, (d.variant == "treatment").to_numpy().astype(int)


def design_matrix(d: pd.DataFrame) -> tuple[np.ndarray, list[bool]]:
    X = d[NUM].astype(float).copy()
    for c in CAT:
        X[c] = d[c].astype("category").cat.codes.astype(float)
    return X.to_numpy(), [False] * len(NUM) + [True] * len(CAT)


def _hgb(cat_mask, loss="poisson", seed=SEED):
    return HistGradientBoostingRegressor(
        loss=loss, learning_rate=0.05, max_iter=150, max_depth=4, min_samples_leaf=200,
        l2_regularization=1.0, categorical_features=cat_mask, random_state=seed)


# ---------------------------------------------------------------------------- learners
def fit_predict(kind: str, Xtr, ytr, ttr, Xte, cat_mask, seed: int = SEED) -> np.ndarray:
    """Return predicted uplift (effect of treatment on expected orders) for the rows of Xte."""
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


def cross_fit(d: pd.DataFrame, t: np.ndarray, n_splits: int = 4, seed: int = SEED) -> pd.DataFrame:
    X, cat_mask = design_matrix(d)
    y = d.orders_in.to_numpy(float)
    scores = {k: np.zeros(len(d)) for k in "STX"}
    for tr, te in KFold(n_splits, shuffle=True, random_state=seed).split(X):
        for k in "STX":
            scores[k][te] = fit_predict(k, X[tr], y[tr], t[tr], X[te], cat_mask, seed)
    out = pd.DataFrame({f"{k}-learner": v for k, v in scores.items()})
    # simple baselines
    rng = np.random.default_rng(seed)
    out["random"] = rng.random(len(d))
    out["rule: new users first"] = d.is_new.to_numpy(float) + 0.001 * rng.random(len(d))
    return out


# -------------------------------------------------------------------------- evaluation
def ipw_effect(y, t) -> np.ndarray:
    """Per-user unbiased estimate of the treatment effect (randomised experiment, known propensity)."""
    p = t.mean()
    return y * (t / p - (1 - t) / (1 - p))


def gain_curve(score, tau_hat) -> np.ndarray:
    """Cumulative estimated incremental orders when treating the top-k users by score."""
    order = np.argsort(-score, kind="stable")
    return np.cumsum(tau_hat[order])


def qini_area(score, tau_hat) -> float:
    """Incremental orders per 1000 users gained over random targeting, averaged across budgets."""
    G = gain_curve(score, tau_hat)
    n = len(G)
    lift_over_random = G - (np.arange(1, n + 1) / n) * G[-1]
    return float(lift_over_random.mean() * 1000 / n)


def bootstrap_ci(scores: pd.DataFrame, tau_hat, reps=200):
    rng = np.random.default_rng(SEED)
    n = len(tau_hat)
    res = {c: [] for c in scores}
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        for c in scores:
            res[c].append(qini_area(scores[c].to_numpy()[idx], tau_hat[idx]))
    return {c: np.array(v) for c, v in res.items()}


# ------------------------------------------------------------------------------- main
def true_relative_lift(exp_id: str, d: pd.DataFrame) -> np.ndarray | None:
    """Hidden truth for diagnostics only: the relative lift the simulator built in for each user."""
    e = next(x for x in EXPERIMENTS if x.experiment_id == exp_id)
    if e.kind == "by_profile":
        ch = d.channel.map({"organic": 0, "paid": 1, "referral": 2}).to_numpy()
        dv = d.device.map({"android": 0, "ios": 1, "web": 2}).to_numpy()
        return np.asarray(e.channel_lifts)[ch] + np.asarray(e.device_lifts)[dv]
    if e.kind == "by_engagement":
        return np.asarray(e.seg_lifts)[d.engagement_segment.map({"casual": 0, "medium": 1, "loyal": 2}).to_numpy()]
    return None


def main(exp_id: str = "exp_008") -> None:
    ART.mkdir(exist_ok=True)
    tag = exp_id
    d, t = load(exp_id)
    y = d.orders_in.to_numpy(float)
    tau_hat = ipw_effect(y, t)
    n = len(d)
    print(f"{exp_id}: {n:,} users, {t.mean():.3f} treated. Overall effect on orders/user = "
          f"{tau_hat.mean():+.4f} (IPW).")

    scores = cross_fit(d, t)
    truth_lift = true_relative_lift(exp_id, d)
    if truth_lift is not None:                       # reference ranking (NOT a ceiling: extra orders also depend on each user's baseline volume)
        rng0 = np.random.default_rng(SEED)
        scores["ref: rank by true relative lift (not a ceiling)"] = truth_lift + 1e-6 * rng0.random(n)
    boots = bootstrap_ci(scores, tau_hat)
    base = boots["rule: new users first"]
    rows = []
    for c in scores:
        a = qini_area(scores[c].to_numpy(), tau_hat)
        lo, hi = np.percentile(boots[c], [2.5, 97.5])
        dlo, dhi = np.percentile(boots[c] - base, [2.5, 97.5])
        rows.append({"model": c, "qini_orders_per_1000_users": round(a, 2),
                     "ci_low": round(lo, 2), "ci_high": round(hi, 2),
                     "diff_vs_rule_low": round(dlo, 2), "diff_vs_rule_high": round(dhi, 2)})
    res = pd.DataFrame(rows).sort_values("qini_orders_per_1000_users", ascending=False)
    res.to_csv(ART / f"uplift_model_comparison_{tag}.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nQini area = extra orders per 1,000 users vs targeting at random (higher is better):")
    print(res.to_string(index=False))

    # ---- diagnostic: how much of the TRUE benefit group do models put in their top 30%?
    if truth_lift is not None:
        top_true = truth_lift >= np.quantile(truth_lift, 0.70)          # users with the highest true relative lift
        k30 = int(0.30 * n)
        diag = {c: float(top_true[np.argsort(-scores[c].to_numpy(), kind="stable")[:k30]].mean()) for c in scores}
        diag = pd.Series(diag, name="share_of_top30_that_are_truly_high_lift").round(3)
        diag.to_csv(ART / f"uplift_top30_diagnostic_{tag}.csv")
        print(f"\nShare of each model's top-30% that truly belongs to the highest-true-lift 30% (random = 0.30):")
        print(diag.to_string())

    # ---- measured uplift by decile for the best learner
    best = res[res.model.str.contains("-learner")].iloc[0].model
    s = scores[best].to_numpy()
    dec = pd.qcut(pd.Series(s).rank(method="first"), 10, labels=False)
    dtab = pd.DataFrame({"decile(0=lowest predicted)": range(10),
                         "mean_predicted_uplift": [s[dec == k].mean() for k in range(10)],
                         "measured_uplift": [tau_hat[dec == k].mean() for k in range(10)]})
    dtab.to_csv(ART / f"uplift_by_decile_{tag}.csv", index=False)
    print(f"\nBest learner: {best}. Predicted vs measured uplift by decile:")
    print(dtab.round(4).to_string(index=False))

    # ---- plots
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    f = np.arange(1, n + 1) / n
    for c in scores:
        G = gain_curve(scores[c].to_numpy(), tau_hat)
        ax.plot(f, G, label=c, lw=1.6 if "-learner" in c else 1.0, ls="-" if "-learner" in c else "--")
    ax.plot([0, 1], [0, gain_curve(scores["random"].to_numpy(), tau_hat)[-1]], color="k", lw=0.7, ls=":")
    ax.set_xlabel("fraction of users treated (highest predicted uplift first)")
    ax.set_ylabel("estimated incremental orders")
    ax.set_title(f"Qini-style gain curves, {tag} (cross-fitted)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(ART / f"uplift_qini_curves_{tag}.png", dpi=130); plt.close(fig)

    # ---- profit view for the best learner
    G = gain_curve(s, tau_hat)
    k = np.arange(1, n + 1)
    profit = VALUE_PER_ORDER * G - COST_PER_TREATED * k
    best_k = int(np.argmax(profit)) + 1
    treat_all = profit[-1]
    print(f"\nProfit view (ASSUMPTIONS: INR {VALUE_PER_ORDER:.0f}/incremental order, INR {COST_PER_TREATED:.0f}/treated user)")
    print(f"  treat nobody : INR 0")
    print(f"  treat everyone: INR {treat_all:,.0f}")
    print(f"  treat top {best_k / n:.0%} by {best}: INR {profit[best_k - 1]:,.0f}")
    print("  (optimistic: the best cut-off is picked on the same data it is scored on; confirm on a fresh holdout)")
    pd.DataFrame({"fraction_treated": k / n, "profit_inr": profit}).iloc[::200].to_csv(ART / f"uplift_profit_curve_{tag}.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.plot(k / n, profit); ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("fraction of users treated"); ax.set_ylabel("estimated profit (INR)")
    ax.set_title(f"Targeting by {best}: profit vs reach (assumed value/cost)")
    fig.tight_layout(); fig.savefig(ART / f"uplift_profit_curve_{tag}.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="exp_008", choices=["exp_008", "exp_009"])
    main(ap.parse_args().exp)
