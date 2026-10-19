"""Revenue metrics are heavy-tailed: a few big baskets or big spenders dominate the mean and widen the interval.

This module compares ways of estimating the relative lift in revenue per user for one experiment (default exp_007):
  naive      difference of means
  CUPED      same, after removing the part explained by pre-period revenue
  winsorized outcomes above a cap are replaced by the cap (cap = pooled quantile, SAME cap for both arms)
  bootstrap  percentile interval for the naive ratio (no normality assumption)
and splits the effect into 'more buyers' (purchase rate) and 'bigger baskets' (revenue per buyer).

Caution: winsorizing changes WHAT is measured (a capped mean), so its lift is not the same quantity as the naive one.
It is a variance-for-bias trade; it is shown next to the uncapped estimates, never instead of them.

Run:  python -m src.experiments.revenue --exp exp_007
"""
from __future__ import annotations

import argparse

import duckdb
import numpy as np
import pandas as pd

from src.config import DB_PATH, ROOT, SQL_DIR
from src.experiments.stats_utils import cuped_adjust


def tail_profile(y: np.ndarray) -> dict:
    """How concentrated is the outcome?  Shares are of total revenue."""
    y = np.sort(np.asarray(y, float))[::-1]
    total = y.sum()
    n = len(y)
    top = lambda frac: float(y[: max(1, int(round(frac * n)))].sum() / total)   # noqa: E731
    mu, sd = y.mean(), y.std(ddof=1)
    return {"users": n, "share_zero": float((y == 0).mean()), "mean": float(mu), "sd": float(sd),
            "cv": float(sd / mu), "top_1pct_share": top(0.01), "top_10pct_share": top(0.10),
            "p99": float(np.quantile(y, 0.99)), "max": float(y[0])}


def winsorize(y: np.ndarray, cap: float) -> np.ndarray:
    return np.minimum(y, cap)


def _ratio_delta(yt: np.ndarray, yc: np.ndarray) -> tuple[float, float]:
    mt, mc = yt.mean(), yc.mean()
    se_t, se_c = yt.std(ddof=1) / np.sqrt(len(yt)), yc.std(ddof=1) / np.sqrt(len(yc))
    return mt / mc - 1, 1.96 * np.sqrt((se_t / mc) ** 2 + (mt * se_c / mc**2) ** 2)


def bootstrap_ratio(yt: np.ndarray, yc: np.ndarray, reps: int = 1000, seed: int = 11) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    out = np.empty(reps)
    for i in range(reps):
        out[i] = yt[rng.integers(0, len(yt), len(yt))].mean() / yc[rng.integers(0, len(yc), len(yc))].mean() - 1
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(lo), float(hi)


def compare_estimators(df: pd.DataFrame, cap_quantile: float = 0.995, reps: int = 1000) -> pd.DataFrame:
    is_t = (df.variant == "treatment").to_numpy()
    y, x = df.revenue_in.to_numpy(float), df.revenue_pre.to_numpy(float)
    rows = []

    def add(name, v, note=""):
        lift, hw = _ratio_delta(v[is_t], v[~is_t])
        rows.append({"estimator": name, "lift": lift, "ci_low": lift - hw, "ci_high": lift + hw,
                     "ci_width": 2 * hw, "note": note})

    add("naive mean ratio", y)
    add("CUPED (pre-period revenue)", cuped_adjust(y, x))
    cap = float(np.quantile(y, cap_quantile))
    add(f"winsorized at pooled p{cap_quantile * 100:g} (cap {cap:,.0f})", winsorize(y, cap),
        "capped mean: a different quantity")
    lo, hi = bootstrap_ratio(y[is_t], y[~is_t], reps)
    rows.append({"estimator": f"bootstrap percentile ({reps} resamples)", "lift": y[is_t].mean() / y[~is_t].mean() - 1,
                 "ci_low": lo, "ci_high": hi, "ci_width": hi - lo, "note": "no normality assumption"})
    return pd.DataFrame(rows)


def decompose(df: pd.DataFrame) -> pd.DataFrame:
    """Revenue per user = purchase rate x revenue per purchasing user. Descriptive: buyers are a post-treatment group."""
    is_t = (df.variant == "treatment").to_numpy()
    y = df.revenue_in.to_numpy(float)
    buy = (y > 0).astype(float)
    rows = []
    for name, arm in (("treatment", is_t), ("control", ~is_t)):
        rows.append({"arm": name, "users": int(arm.sum()), "purchase_rate": buy[arm].mean(),
                     "revenue_per_buyer": y[arm][buy[arm] == 1].mean(), "revenue_per_user": y[arm].mean()})
    t = pd.DataFrame(rows).set_index("arm")
    lift_buy = t.loc["treatment", "purchase_rate"] / t.loc["control", "purchase_rate"] - 1
    lift_basket = t.loc["treatment", "revenue_per_buyer"] / t.loc["control", "revenue_per_buyer"] - 1
    t.loc["relative lift"] = [np.nan, lift_buy, lift_basket, t.loc["treatment", "revenue_per_user"] / t.loc["control", "revenue_per_user"] - 1]
    return t


def main(exp_id: str = "exp_007") -> None:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute((SQL_DIR / "marts" / "experiment_user_metrics.sql").read_text(), {"exp_id": exp_id}).df()
    truth = con.execute("SELECT nominal_relative_lift, effect_kind FROM answer_key.ground_truth WHERE experiment_id = ?",
                        [exp_id]).fetchone()
    pd.set_option("display.width", 200, "display.max_columns", 20, "display.float_format", lambda v: f"{v:,.4f}")
    prof = tail_profile(df.revenue_in.to_numpy(float))
    print(f"{exp_id}: revenue per user over the experiment window ({prof['users']:,} users)")
    print(f"  users with zero revenue: {prof['share_zero']:.1%} | mean {prof['mean']:,.1f} | sd {prof['sd']:,.1f} (cv {prof['cv']:.2f})")
    print(f"  top 1% of users hold {prof['top_1pct_share']:.1%} of revenue, top 10% hold {prof['top_10pct_share']:.1%}; "
          f"p99 {prof['p99']:,.0f}, max {prof['max']:,.0f}")
    est = compare_estimators(df)
    print("\nRelative lift in revenue per user by estimator:")
    print(est.to_string(index=False))
    dec = decompose(df)
    print("\nWhere does the lift come from? (purchase rate x revenue per buyer)")
    print(dec.to_string())
    print("Descriptive only: buyers are a post-treatment group (the offer may change who buys), so the split is not causal.")
    print(f"\nAnswer key (for judging only): nominal lift {truth[0]:+.3f} ({truth[1]}); built to move basket value, not conversion.")
    out = ROOT / "artifacts"; out.mkdir(exist_ok=True)
    est.to_csv(out / f"revenue_estimators_{exp_id}.csv", index=False)
    dec.to_csv(out / f"revenue_decomposition_{exp_id}.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="exp_007")
    main(ap.parse_args().exp)
