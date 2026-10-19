"""First-pass analysis: SRM check, naive effect, CUPED effect, compared with the hidden truth.

Run: python -m src.experiments.first_look
Deliberately a *starting point*. Later steps add power, peeking, multiple-testing and the
Criteo validation.
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from src.config import DB_PATH, ROOT, SQL_DIR


def rel_lift(y_t: np.ndarray, y_c: np.ndarray) -> tuple[float, float, float]:
    """Relative lift of treatment over control, a rough 95% CI half-width (delta method), p-value."""
    mt, mc = y_t.mean(), y_c.mean()
    se_t, se_c = y_t.std(ddof=1) / np.sqrt(len(y_t)), y_c.std(ddof=1) / np.sqrt(len(y_c))
    lift = mt / mc - 1
    se_rel = np.sqrt((se_t / mc) ** 2 + (mt * se_c / mc**2) ** 2)
    return lift, 1.96 * se_rel, stats.ttest_ind(y_t, y_c, equal_var=False).pvalue


def cuped(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Variance reduction with a pre-experiment covariate. Mean is preserved, so lift stays unbiased."""
    var_x = np.var(x, ddof=1)
    if var_x == 0:                      # no pre-period signal in this slice: nothing to adjust
        return y.copy()
    theta = np.cov(y, x, ddof=1)[0, 1] / var_x
    return y - theta * (x - x.mean())


def analyse(df: pd.DataFrame, metric: str, cov: str) -> dict:
    is_t = (df.variant == "treatment").to_numpy()
    y, x = df[metric].to_numpy(float), df[cov].to_numpy(float)
    lift, hw, p = rel_lift(y[is_t], y[~is_t])
    ya = cuped(y, x)
    lift_a, hw_a, p_a = rel_lift(ya[is_t], ya[~is_t])
    chi_p = stats.chisquare([is_t.sum(), (~is_t).sum()]).pvalue
    return {"treat_n": int(is_t.sum()), "ctrl_n": int((~is_t).sum()), "SRM_p": chi_p,
            "naive_lift": lift, "naive_CI±": hw, "naive_p": p,
            "cuped_lift": lift_a, "cuped_CI±": hw_a, "cuped_p": p_a}


def main() -> None:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    sql = (SQL_DIR / "marts" / "experiment_user_metrics.sql").read_text()
    truth = con.execute("SELECT * FROM answer_key.ground_truth").df().set_index("experiment_id")
    meta = con.execute("SELECT experiment_id, name, primary_metric FROM experiments").df().set_index("experiment_id")

    rows, frames = [], {}
    for exp_id in truth.index:
        df = con.execute(sql, {"exp_id": exp_id}).df()
        frames[exp_id] = df
        metric = "revenue" if meta.loc[exp_id, "primary_metric"] == "revenue_per_user" else "orders"
        res = analyse(df, f"{metric}_in", f"{metric}_pre")
        rows.append({"exp": exp_id, "name": meta.loc[exp_id, "name"], "metric": metric, **res,
                     "effect_kind": truth.loc[exp_id, "effect_kind"],
                     # heterogeneous experiments have no single true lift (it differs by user)
                     "true_nominal": np.nan if truth.loc[exp_id, "effect_kind"] in ("by_profile", "by_engagement")
                     else truth.loc[exp_id, "nominal_relative_lift"]})
    pd.set_option("display.width", 250, "display.max_columns", 30,
                  "display.float_format", lambda v: f"{v:.4f}")
    table = pd.DataFrame(rows)
    (ROOT / "artifacts").mkdir(exist_ok=True)
    table.to_csv(ROOT / "artifacts" / "first_look_verdicts.csv", index=False)
    print(table.to_string(index=False))
    print("\nSRM_p < 0.001 -> arm sizes do not match the planned 50/50 split: do NOT trust that result.")

    print("\nexp_005 (onboarding_flow, true +18% for NEW users only) split by is_new:")
    for flag, g in frames["exp_005"].groupby("is_new"):
        r = analyse(g, "orders_in", "orders_pre")
        print(f"  is_new={flag}: n={r['treat_n'] + r['ctrl_n']:>6}  naive_lift={r['naive_lift']:+.3f} ± {r['naive_CI±']:.3f}")

    print("\nexp_008 (winback_offer_profile). Hidden truth: lift = channel effect (organic +4%, paid +30%, referral -10%)")
    print("  + device effect (android 0, ios -5%, web +15%). Observed lift by channel and by device:")
    d8 = frames["exp_008"]
    for col in ("channel", "device"):
        for lab, g in d8.groupby(col):
            r = analyse(g, "orders_in", "orders_pre")
            print(f"  {col}={lab:9s} n={r['treat_n'] + r['ctrl_n']:>6}  naive_lift={r['naive_lift']:+.3f} ± {r['naive_CI±']:.3f}")

    print("\nexp_009 (personalised_offer_latent; hidden truth: casual +30%, medium +10%, loyal -3%).")
    print("Segments are hidden, so we can only split by what we observe (new vs existing, active before):")
    d9 = frames["exp_009"].copy()
    d9["group"] = np.where(d9.is_new, "new (joined during experiment)",
                           np.where(d9.sessions_pre == 0, "existing, inactive before", "existing, active before"))
    for lab, g in d9.groupby("group"):
        r = analyse(g, "orders_in", "orders_pre")
        print(f"  {lab:32s} n={r['treat_n'] + r['ctrl_n']:>6}  naive_lift={r['naive_lift']:+.3f} ± {r['naive_CI±']:.3f}")


if __name__ == "__main__":
    main()
