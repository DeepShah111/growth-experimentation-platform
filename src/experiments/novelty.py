"""Novelty-effect analysis: does the lift fade as users get used to the change?

Input: the per-user-week table from sql/marts/experiment_user_weekly.sql
       (columns: exposure_week, user_id, variant, n_orders).

Why not just compare the whole-window average?  A change that is +12% in week 1 and +1% in week 4 averages to
roughly +7%, and shipping on that number over-promises. The right question is the lift week by week.

Run:  python -m src.experiments.novelty --exp exp_004
"""
from __future__ import annotations

import argparse

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from src.config import DB_PATH, ROOT, SQL_DIR


def usable_weeks(df: pd.DataFrame, min_users: int = 500) -> list[int]:
    """Exposure weeks where BOTH arms have at least `min_users` users (later weeks can be nearly empty)."""
    n = df.groupby(["exposure_week", "variant"]).size().unstack(fill_value=0)
    ok = n[(n.get("treatment", 0) >= min_users) & (n.get("control", 0) >= min_users)]
    return [int(w) for w in ok.index]


def weekly_effects(df: pd.DataFrame, min_users: int = 500) -> pd.DataFrame:
    """Treatment-vs-control orders per user per exposure week, with a 95% CI on the relative lift."""
    rows = []
    for wk in usable_weeks(df, min_users):
        g = df[df.exposure_week == wk]
        t = g.loc[g.variant == "treatment", "n_orders"].to_numpy(float)
        c = g.loc[g.variant == "control", "n_orders"].to_numpy(float)
        mt, mc = t.mean(), c.mean()
        se_t, se_c = t.std(ddof=1) / np.sqrt(len(t)), c.std(ddof=1) / np.sqrt(len(c))
        lift = mt / mc - 1
        se_rel = np.sqrt((se_t / mc) ** 2 + (mt * se_c / mc**2) ** 2)      # delta method
        rows.append({"exposure_week": int(wk), "n_treat": len(t), "n_ctrl": len(c),
                     "rate_treat": mt, "rate_ctrl": mc, "abs_effect": mt - mc,
                     "rel_lift": lift, "ci_half": 1.96 * se_rel,
                     "p_value": stats.ttest_ind(t, c, equal_var=False).pvalue})
    return pd.DataFrame(rows)


def decay_test(df: pd.DataFrame, min_users: int = 500) -> dict:
    """Does the treatment effect change from week to week? (negative = fading, the novelty signature)

    The same users appear in several weeks, so weeks are NOT independent samples. We therefore reduce each user to
    ONE number: the slope of their weekly orders against exposure week (ordinary least squares). The average slope
    difference between treatment and control is the change in effect per week, tested with a Welch t-test.
    Only users observed in every usable week are included. Absolute scale (orders per user-week per week).
    """
    weeks = usable_weeks(df, min_users)
    if len(weeks) < 3:
        raise ValueError(f"need at least 3 usable weeks for a trend test, got {weeks}")
    wide = (df[df.exposure_week.isin(weeks)]
            .pivot_table(index=["user_id", "variant"], columns="exposure_week", values="n_orders", aggfunc="first")
            .dropna().reset_index())
    w = np.asarray(weeks, float)
    wc = w - w.mean()
    slope = wide[weeks].to_numpy(float) @ wc / (wc @ wc)
    is_t = (wide.variant == "treatment").to_numpy()
    t, c = slope[is_t], slope[~is_t]
    diff = t.mean() - c.mean()
    se = np.sqrt(t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c))
    return {"weeks_used": weeks, "users_in_all_weeks": len(wide),
            "effect_change_per_week": diff, "ci_half": 1.96 * se,
            "p_value": float(stats.ttest_ind(t, c, equal_var=False).pvalue)}


def load(exp_id: str) -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    return con.execute((SQL_DIR / "marts" / "experiment_user_weekly.sql").read_text(), {"exp_id": exp_id}).df()


def main(exp_id: str = "exp_004") -> None:
    df = load(exp_id)
    pd.set_option("display.width", 200, "display.float_format", lambda v: f"{v:.4f}")
    wk = weekly_effects(df)
    (ROOT / "artifacts").mkdir(exist_ok=True)
    wk.to_csv(ROOT / "artifacts" / f"novelty_weekly_{exp_id}.csv", index=False)
    print(f"{exp_id}: lift by week since the user's own assignment (orders per user-week)")
    print(wk.to_string(index=False))
    r = decay_test(df)
    print(f"\nTrend test over weeks {r['weeks_used']} (users seen in all of them: {r['users_in_all_weeks']:,}):")
    print(f"  change in effect per week = {r['effect_change_per_week']:+.4f} +/- {r['ci_half']:.4f} orders/user-week, "
          f"p = {r['p_value']:.4f}   (negative = fading)")
    print("  Weeks are not independent, so each user is reduced to one slope before comparing arms.")
    print("  A non-significant result means 'cannot tell at this sample size', not 'no novelty effect'.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="exp_004")
    main(ap.parse_args().exp)
