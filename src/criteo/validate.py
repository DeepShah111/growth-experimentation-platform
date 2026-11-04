"""Run the SAME methods as the simulator project on REAL randomised data (Criteo Uplift v2).

    python -m src.criteo.validate --path data/criteo_1m.parquet [--max-rows 300000] [--reps 100]

Checks
  1. randomisation   : covariate balance between treatment and control (the real test of randomisation)
  2. effect          : intent-to-treat effect of `treatment` on visit and conversion, with and without
                       regression adjustment (the continuous-covariate cousin of CUPED)
  3. power           : how many users would the rare 'conversion' outcome need?
  4. uplift          : S/T/X learners, cross-fitted, Qini gain vs random, bootstrap CIs

We analyse by `treatment`, never by `exposure`: exposure happens AFTER assignment, so conditioning
on it would break randomisation (post-treatment bias).
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold

from src.config import ROOT
from src.experiments.stats_utils import required_n_per_arm
from src.engine.uplift_core import SEED, fit_predict, gain_curve, ipw_effect, qini_area

ART = ROOT / "artifacts"


# ------------------------------------------------------------------------ 1. randomisation
def balance_table(X: pd.DataFrame, t: np.ndarray) -> pd.DataFrame:
    """Standardised mean difference per feature. |SMD| < 0.05 is the usual 'balanced' rule of thumb."""
    xt, xc = X[t == 1], X[t == 0]
    smd = (xt.mean() - xc.mean()) / np.sqrt((xt.var() + xc.var()) / 2)
    pval = [stats.ttest_ind(xt[c], xc[c], equal_var=False).pvalue for c in X.columns]
    return pd.DataFrame({"smd": smd.round(4), "p_value": np.round(pval, 4)})


# ------------------------------------------------------------------------------ 2. effect
def diff_in_rates(y: np.ndarray, t: np.ndarray) -> dict:
    p1, p0 = y[t == 1].mean(), y[t == 0].mean()
    n1, n0 = (t == 1).sum(), (t == 0).sum()
    se = np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
    return {"rate_treat": p1, "rate_ctrl": p0, "abs_effect": p1 - p0, "abs_ci_half": 1.96 * se,
            "rel_lift": p1 / p0 - 1, "p_value": 2 * stats.norm.sf(abs(p1 - p0) / se)}


def lin_adjusted_effect(y: np.ndarray, t: np.ndarray, X: np.ndarray) -> dict:
    """Regression-adjusted ATE (Lin 2013): y ~ 1 + t + Xc + t*Xc, robust (HC1) SE on t."""
    Xc = X - X.mean(0)
    D = np.column_stack([np.ones(len(y)), t, Xc, t[:, None] * Xc]).astype(np.float64)
    beta, *_ = np.linalg.lstsq(D, y.astype(np.float64), rcond=None)
    e = y - D @ beta
    bread = np.linalg.inv(D.T @ D)
    De = D * e[:, None]
    cov = bread @ (De.T @ De) @ bread * len(y) / (len(y) - D.shape[1])
    se = float(np.sqrt(cov[1, 1]))
    return {"abs_effect": float(beta[1]), "abs_ci_half": 1.96 * se,
            "p_value": float(2 * stats.norm.sf(abs(beta[1]) / se))}


# -------------------------------------------------------------------------------- 4. uplift
def cross_fit_scores(X: np.ndarray, y: np.ndarray, t: np.ndarray, n_splits: int = 4, kinds: str = "STX") -> pd.DataFrame:
    cat_mask = [False] * X.shape[1]
    scores = {k: np.zeros(len(y)) for k in kinds}
    for tr, te in KFold(n_splits, shuffle=True, random_state=SEED).split(X):
        for k in kinds:
            scores[k][te] = fit_predict(k, X[tr], y[tr].astype(float), t[tr], X[te], cat_mask)
    out = pd.DataFrame({f"{k}-learner": v for k, v in scores.items()})
    out["random"] = np.random.default_rng(SEED).random(len(y))
    return out


def uplift_report(scores: pd.DataFrame, y: np.ndarray, t: np.ndarray, label: str, reps: int,
                  tau: np.ndarray | None = None) -> pd.DataFrame:
    """tau: per-user effect pseudo-outcomes. Default = IPW with a constant propensity; pass AIPW values to be robust."""
    tau = ipw_effect(y.astype(float), t) if tau is None else tau
    n = len(y)
    rng = np.random.default_rng(SEED)
    boots = {c: [] for c in scores}
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        for c in scores:
            boots[c].append(qini_area(scores[c].to_numpy()[idx], tau[idx]))
    rows = []
    for c in scores:
        a = qini_area(scores[c].to_numpy(), tau)
        lo, hi = np.percentile(boots[c], [2.5, 97.5])
        dlo, dhi = np.percentile(np.array(boots[c]) - np.array(boots["random"]), [2.5, 97.5])
        rows.append({"outcome": label, "model": c, "qini_extra_per_1000_users": round(a, 3),
                     "ci_low": round(lo, 3), "ci_high": round(hi, 3),
                     "diff_vs_random_low": round(dlo, 3), "diff_vs_random_high": round(dhi, 3)})
    return pd.DataFrame(rows).sort_values("qini_extra_per_1000_users", ascending=False)


def decile_table(score: np.ndarray, y: np.ndarray, t: np.ndarray) -> pd.DataFrame:
    tau = ipw_effect(y.astype(float), t)
    dec = pd.qcut(pd.Series(score).rank(method="first"), 10, labels=False)
    return pd.DataFrame({"decile(0=lowest predicted)": range(10),
                         "mean_predicted_uplift": [score[dec == k].mean() for k in range(10)],
                         "measured_uplift": [tau[dec == k].mean() for k in range(10)]})


# ---------------------------------------------------------------------------------- main
def run(path: str, max_rows: int | None = None, reps: int = 100, out_dir=ART) -> dict:
    out_dir.mkdir(exist_ok=True)
    df = pd.read_parquet(path)
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=SEED).reset_index(drop=True)
    feats = [c for c in df.columns if c.startswith("f")]
    t = df.treatment.to_numpy().astype(int)
    X = df[feats].to_numpy(np.float32)
    res: dict = {}

    print(f"rows={len(df):,}  treatment_ratio={t.mean():.4f}  visit_rate={df.visit.mean():.4f}  "
          f"conversion_rate={df.conversion.mean():.5f}")
    print("(published: treatment ratio ~0.85, visit ~0.047, conversion ~0.00292)")

    bal = balance_table(df[feats], t)
    bal.to_csv(out_dir / "criteo_balance.csv")
    res["balance"] = bal
    print(f"\n[1] Covariate balance: max |SMD| = {bal.smd.abs().max():.4f} (common rule of thumb: |SMD| < 0.1 counts as balanced; but under perfect randomisation at this size about +-0.003 is expected, so see diagnostics)")

    rows = []
    for outcome in ("visit", "conversion"):
        y = df[outcome].to_numpy().astype(float)
        a, b = diff_in_rates(y, t), lin_adjusted_effect(y, t, X)
        rows.append({"outcome": outcome, **{f"naive_{k}": v for k, v in a.items()},
                     "adj_abs_effect": b["abs_effect"], "adj_abs_ci_half": b["abs_ci_half"], "adj_p": b["p_value"],
                     "ci_width_ratio_adj_vs_naive": b["abs_ci_half"] / a["abs_ci_half"]})
    eff = pd.DataFrame(rows)
    eff.to_csv(out_dir / "criteo_effects.csv", index=False)
    res["effects"] = eff
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.float_format", lambda v: f"{v:.5f}")
    print("\n[2] Intent-to-treat effect of showing ads (treatment vs control):")
    print(eff.T.to_string(header=False))

    conv_ctrl = df.conversion[t == 0]
    n_ctrl = int((t == 0).sum())
    mean, sd = conv_ctrl.mean(), conv_ctrl.std()
    need = required_n_per_arm(0.10, mean, sd)
    print(f"\n[3] Power: control conversion rate {mean:.5f} with {n_ctrl:,} control users. "
          f"Detecting a 10% relative lift in conversion needs ~{need:,.0f} users per arm; visit has ~16x more events.")

    outs = {}
    for outcome in ("visit", "conversion"):
        y = df[outcome].to_numpy()
        scores = cross_fit_scores(X, y, t)
        rep = uplift_report(scores, y, t, outcome, reps)
        outs[outcome] = (scores, rep)
        print(f"\n[4] Uplift on '{outcome}': extra events per 1,000 users vs random targeting")
        print(rep.drop(columns="outcome").to_string(index=False))
    pd.concat([r for _, r in outs.values()]).to_csv(out_dir / "criteo_uplift_models.csv", index=False)
    res["uplift"] = {k: v[1] for k, v in outs.items()}

    best_scores, best_rep = outs["visit"]
    best = best_rep[best_rep.model.str.contains("-learner")].iloc[0].model
    dt = decile_table(best_scores[best].to_numpy(), df.visit.to_numpy(), t)
    dt.to_csv(out_dir / "criteo_visit_deciles.csv", index=False)
    print(f"\nBest learner on visit = {best}. Predicted vs measured uplift by decile:")
    print(dt.to_string(index=False))

    tau = ipw_effect(df.visit.to_numpy().astype(float), t)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    f = np.arange(1, len(df) + 1) / len(df)
    for c in best_scores:
        ax.plot(f, gain_curve(best_scores[c].to_numpy(), tau), label=c, lw=1.6 if "-learner" in c else 1.0,
                ls="-" if "-learner" in c else "--")
    ax.set_xlabel("fraction of users targeted (highest predicted uplift first)")
    ax.set_ylabel("estimated incremental visits"); ax.set_title("Criteo (real data): Qini-style gain, outcome = visit")
    ax.legend(fontsize=8, loc="lower right"); fig.tight_layout(); fig.savefig(out_dir / "criteo_qini_visit.png", dpi=130); plt.close(fig)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="data/criteo_1m.parquet")
    ap.add_argument("--max-rows", type=int, default=None, help="cap rows for a quick laptop run")
    ap.add_argument("--reps", type=int, default=100, help="bootstrap repetitions")
    a = ap.parse_args()
    run(a.path, a.max_rows, a.reps)
