"""Why do the naive and regression-adjusted Criteo effects disagree?  (run AFTER src.criteo.validate)

    python -m src.criteo.diagnostics --path data/criteo_1m.parquet [--max-rows 400000]

In a perfectly randomised test the treatment flag cannot be predicted from the features, and the naive
difference in means, the regression-adjusted effect and a doubly-robust (AIPW) effect all agree.
This script checks that directly:
  1. per-feature standardised mean differences (SMD)
  2. propensity check: can a model predict `treatment` from the features? (cross-validated AUC; 0.5 = no)
  3. effect estimates: naive, Lin-adjusted, AIPW (cross-fitted gradient boosting, estimated propensity)
  4. uplift deciles re-measured with AIPW pseudo-outcomes instead of a constant 85% propensity
  5. exposure rate (post-treatment; shown for context only, never used as a feature)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from src.config import ROOT
from src.criteo.validate import balance_table, cross_fit_scores, diff_in_rates, lin_adjusted_effect, uplift_report
from src.targeting.uplift import SEED, ipw_effect

ART = ROOT / "artifacts"
CLIP = (0.02, 0.98)                                   # keeps AIPW weights from exploding


def _clf():
    return HistGradientBoostingClassifier(learning_rate=0.1, max_iter=100, max_depth=4,
                                          min_samples_leaf=200, random_state=SEED)


def propensity_auc(X, t, n_splits=3) -> float:
    """Cross-validated AUC of predicting treatment from features. ~0.5 means features carry no treatment signal."""
    pred = np.zeros(len(t))
    for tr, te in KFold(n_splits, shuffle=True, random_state=SEED).split(X):
        pred[te] = _clf().fit(X[tr], t[tr]).predict_proba(X[te])[:, 1]
    return float(roc_auc_score(t, pred))


def aipw(X, y, t, n_splits=3) -> dict:
    """Cross-fitted doubly-robust effect. tau_i = mu1-mu0 + T(y-mu1)/e - (1-T)(y-mu0)/(1-e)."""
    n = len(y)
    mu1, mu0, e = np.zeros(n), np.zeros(n), np.zeros(n)
    for tr, te in KFold(n_splits, shuffle=True, random_state=SEED).split(X):
        a, b = tr[t[tr] == 1], tr[t[tr] == 0]
        mu1[te] = _clf().fit(X[a], y[a]).predict_proba(X[te])[:, 1]
        mu0[te] = _clf().fit(X[b], y[b]).predict_proba(X[te])[:, 1]
        e[te] = _clf().fit(X[tr], t[tr]).predict_proba(X[te])[:, 1]
    e = np.clip(e, *CLIP)
    tau = mu1 - mu0 + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e)
    se = tau.std(ddof=1) / np.sqrt(n)
    return {"tau_i": tau, "effect": float(tau.mean()), "ci_half": float(1.96 * se),
            "p_value": float(2 * stats.norm.sf(abs(tau.mean()) / se)), "e_hat": e}


def run(path: str, max_rows: int | None = None, out_dir=ART) -> dict:
    out_dir.mkdir(exist_ok=True)
    df = pd.read_parquet(path)
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=SEED).reset_index(drop=True)
    feats = [c for c in df.columns if c.startswith("f")]
    X, t = df[feats].to_numpy(np.float32), df.treatment.to_numpy().astype(int)
    res: dict = {}
    pd.set_option("display.width", 220, "display.float_format", lambda v: f"{v:.5f}")

    bal = balance_table(df[feats], t).sort_values("smd", key=abs, ascending=False)
    bal.to_csv(out_dir / "criteo_balance_sorted.csv")
    n = len(df)
    se_smd = np.sqrt(1 / (t == 1).sum() + 1 / (t == 0).sum())
    print(f"[1] Feature SMDs, largest first. Expected size under perfect randomisation: about +-{se_smd:.4f} (1 SE).")
    print(bal.head(6).to_string())
    print(f"    {int((bal.smd.abs() > 3 * se_smd).sum())} of {len(bal)} features sit beyond 3 SE.")
    res["balance"] = bal

    auc = propensity_auc(X, t)
    print(f"\n[2] Propensity check: AUC of predicting treatment from features = {auc:.4f}  (0.5 = perfectly randomised)")
    res["propensity_auc"] = auc

    rows = []
    for outcome in ("visit", "conversion"):
        y = df[outcome].to_numpy().astype(float)
        a, b, c = diff_in_rates(y, t), lin_adjusted_effect(y, t, X.astype(np.float64)), aipw(X, y, t)
        rows += [{"outcome": outcome, "method": "naive difference", "effect": a["abs_effect"], "ci_half": a["abs_ci_half"]},
                 {"outcome": outcome, "method": "regression-adjusted (Lin)", "effect": b["abs_effect"], "ci_half": b["abs_ci_half"]},
                 {"outcome": outcome, "method": "AIPW (boosted, est. propensity)", "effect": c["effect"], "ci_half": c["ci_half"]}]
        if outcome == "visit":
            res["aipw_visit"] = c
    eff = pd.DataFrame(rows)
    eff.to_csv(out_dir / "criteo_effects_robustness.csv", index=False)
    print("\n[3] Absolute effect of treatment (the three should agree if randomisation is clean):")
    print(eff.to_string(index=False))
    res["effects"] = eff

    y = df.visit.to_numpy().astype(float)
    s = cross_fit_scores(X, y.astype(int), t, kinds="S")["S-learner"].to_numpy()
    tau_const = ipw_effect(y, t)                      # assumes a constant 85% propensity
    tau_dr = res["aipw_visit"]["tau_i"]               # uses estimated propensity + outcome models
    dec = pd.qcut(pd.Series(s).rank(method="first"), 10, labels=False)
    tab = pd.DataFrame({"decile(0=lowest predicted)": range(10),
                        "mean_predicted_uplift": [s[dec == k].mean() for k in range(10)],
                        "measured_constant_propensity": [tau_const[dec == k].mean() for k in range(10)],
                        "measured_AIPW": [tau_dr[dec == k].mean() for k in range(10)]})
    tab.to_csv(out_dir / "criteo_visit_deciles_robust.csv", index=False)
    share = lambda col: tab[col].iloc[9] / tab[col].sum()
    print("\n[4] Visit uplift by predicted decile, measured two ways:")
    print(tab.to_string(index=False))
    print(f"    top decile's share of all incremental visits: constant-propensity {share('measured_constant_propensity'):.1%}, "
          f"AIPW {share('measured_AIPW'):.1%}")
    res["deciles"] = tab

    if "exposure" in df:
        ex = df.exposure[t == 1].mean()
        print(f"\n[5] Exposure: {ex:.1%} of treated users actually saw an ad (post-treatment; context only). "
              f"The effects above are per user ASSIGNED to be targeted, not per ad seen.")
    return res


def robust_comparison(path: str, max_rows: int | None = None, reps: int = 100, out_dir=ART,
                      outcomes=("visit", "conversion")) -> pd.DataFrame:
    """Re-run the S/T/X-learner comparison, scoring with AIPW pseudo-outcomes (robust to the small feature imbalance)."""
    out_dir.mkdir(exist_ok=True)
    df = pd.read_parquet(path)
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=SEED).reset_index(drop=True)
    feats = [c for c in df.columns if c.startswith("f")]
    X, t = df[feats].to_numpy(np.float32), df.treatment.to_numpy().astype(int)
    frames = []
    for outcome in outcomes:
        y = df[outcome].to_numpy()
        dr = aipw(X, y.astype(float), t)
        scores = cross_fit_scores(X, y.astype(int), t)
        rep = uplift_report(scores, y, t, outcome, reps, tau=dr["tau_i"])
        print(f"\nRobust (AIPW-scored) uplift comparison on '{outcome}': extra events per 1,000 users vs random")
        print(rep.drop(columns="outcome").to_string(index=False))
        frames.append(rep)
    out = pd.concat(frames)
    out.to_csv(out_dir / "criteo_uplift_models_robust.csv", index=False)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="data/criteo_1m.parquet")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--compare", action="store_true", help="re-run the S/T/X comparison with robust AIPW scoring")
    ap.add_argument("--reps", type=int, default=100)
    a = ap.parse_args()
    if a.compare:
        robust_comparison(a.path, a.max_rows, a.reps)
    else:
        run(a.path, a.max_rows)
