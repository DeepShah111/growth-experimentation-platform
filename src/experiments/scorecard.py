"""Experiment decision scorecard: from raw data to a ship / iterate / hold call, with the reasons.

Every experiment goes through the same gates, in this order:
  1. TRUST      sample-ratio check. A failure means the assignment is broken: no analysis, DO NOT TRUST.
  2. EVIDENCE   CUPED lift on the primary metric, tested with Holm correction across all trusted experiments.
  3. POWER      minimum detectable lift (MDE). If the result is not significant AND the MDE is larger than the
                smallest lift worth shipping, the answer is INCONCLUSIVE, not "no effect".
  4. SAFETY     guardrail = the other money metric (revenue for order tests, orders for revenue tests):
                a significantly negative guardrail blocks shipping.
  5. VALIDITY   overlap with another experiment on the same surface is only a blocker if the interaction test finds
                an interaction; a fading effect (significantly negative weekly trend) means extend and re-read.
  6. ECONOMICS  for experiments where every treated user costs money (the win-back offer), a statistically significant
                lift is not enough: if the lift is significantly UNPROFITABLE at the assumed unit economics, do not ship
                it to everyone (target a subset instead). The economics are assumptions set in src/config.py.

Decisions: DO NOT TRUST | DO NOT SHIP (harmful) | SHIP | ITERATE | INCONCLUSIVE (underpowered) | NO EFFECT FOUND.
The thresholds below are choices, not facts; change them in one place. The hidden answer key is used ONLY afterwards,
to judge the decisions, never inside `decide`.

Run:  python -m src.experiments.scorecard
"""
from __future__ import annotations

import argparse

import duckdb
import numpy as np
import pandas as pd

from src.config import COST_PER_TREATED, DB_PATH, ROOT, SQL_DIR, VALUE_PER_ORDER
from src.experiments.first_look import analyse
from src.experiments.interaction import interaction_effect
from src.experiments.novelty import decay_test, usable_weeks, weekly_effects
from src.engine.decision import ALPHA, INTERACTION_ALPHA, MIN_EFFECT, SRM_ALPHA, TREND_ALPHA, decide  # noqa: F401  (re-exported)
from src.experiments.stats_utils import holm, holm_adjust, mde_rel, net_value_per_user  # noqa: F401

COST_PER_TREATED_USER = {"exp_008": COST_PER_TREATED}   # experiments where each treated user costs money (INR)


# ---------------------------------------------------------------------------------- building the table
def _sql(name: str) -> str:
    return (SQL_DIR / name).read_text()


def build(con: duckdb.DuckDBPyConnection, min_users_week: int = 500, min_effect: float | None = None) -> pd.DataFrame:
    bar = MIN_EFFECT if min_effect is None else min_effect
    exps = con.execute("SELECT experiment_id, name, primary_metric FROM experiments ORDER BY experiment_id").df()
    pairs = con.execute(_sql("analysis/overlapping_experiments.sql")).df()
    partner = {}
    for _, p in pairs.iterrows():
        cells = con.execute(_sql("analysis/experiment_interaction.sql"), {"exp_a": p.exp_a, "exp_b": p.exp_b}).df()
        ip = interaction_effect(cells)["p_value"]
        partner[p.exp_a] = (p.exp_b, ip)
        partner[p.exp_b] = (p.exp_a, ip)

    rows = []
    for _, e in exps.iterrows():
        df = con.execute(_sql("marts/experiment_user_metrics.sql"), {"exp_id": e.experiment_id}).df()
        rev_primary = e.primary_metric == "revenue_per_user"
        prim, sec = ("revenue", "orders") if rev_primary else ("orders", "revenue")
        res = analyse(df, f"{prim}_in", f"{prim}_pre")
        gua = analyse(df, f"{sec}_in", f"{sec}_pre")
        y = df[f"{prim}_in"].to_numpy(float); x = df[f"{prim}_pre"].to_numpy(float)
        n_arm = (res["treat_n"] + res["ctrl_n"]) / 2
        rho = float(np.corrcoef(y, x)[0, 1]) if x.std() > 0 else 0.0
        row = {"experiment": e.experiment_id, "name": e["name"], "primary_metric": prim,
               "n_treat": res["treat_n"], "n_ctrl": res["ctrl_n"], "srm_p": res["SRM_p"],
               "lift": res["cuped_lift"], "ci_low": res["cuped_lift"] - res["cuped_CI±"],
               "ci_high": res["cuped_lift"] + res["cuped_CI±"], "p_value": res["cuped_p"],
               "mde": float(mde_rel(n_arm, y.mean(), y.std(ddof=1), rho=rho)),
               "guardrail_metric": sec, "guardrail_lift": gua["cuped_lift"],
               "guardrail_breach": bool(gua["cuped_lift"] + gua["cuped_CI±"] < 0)}
        row["underpowered"] = row["mde"] > bar
        cost = COST_PER_TREATED_USER.get(e.experiment_id)
        row["net_value"] = row["net_ci_low"] = row["net_ci_high"] = np.nan
        if cost is not None and not rev_primary:
            is_t = (df.variant == "treatment").to_numpy()
            row["net_value"], row["net_ci_low"], row["net_ci_high"] = net_value_per_user(
                y[is_t], y[~is_t], VALUE_PER_ORDER, cost)
        ov = partner.get(e.experiment_id)
        row["overlap_with"], row["interaction_p"] = (ov if ov else ("", None))
        wk = con.execute(_sql("marts/experiment_user_weekly.sql"), {"exp_id": e.experiment_id}).df()
        weeks = usable_weeks(wk, min_users_week)
        row.update({"trend_slope": None, "trend_p": None, "latest_week_lift": None, "latest_week_ci_half": None})
        if len(weeks) >= 3:
            dt = decay_test(wk, min_users_week)
            we = weekly_effects(wk, min_users_week).iloc[-1]
            row.update({"trend_slope": dt["effect_change_per_week"], "trend_p": dt["p_value"],
                        "latest_week_lift": float(we.rel_lift), "latest_week_ci_half": float(we.ci_half)})
        rows.append(row)

    t = pd.DataFrame(rows)
    ok = (t.srm_p >= SRM_ALPHA).to_numpy()
    t["p_adj"] = np.nan
    t["significant"] = False
    if ok.sum():
        pv = t.loc[ok, "p_value"].to_numpy()
        t.loc[ok, "p_adj"] = holm_adjust(pv)
        t.loc[ok, "significant"] = holm(pv, ALPHA)
    out = [decide(r, min_effect=bar) for r in t.to_dict("records")]
    t["decision"] = [o[0] for o in out]
    t["reasons"] = [" | ".join(o[1]) for o in out]
    return t


def _fmt(t: pd.DataFrame) -> pd.DataFrame:
    f = t[["experiment", "name", "primary_metric", "srm_p", "lift", "ci_low", "ci_high", "p_adj", "mde", "decision"]].copy()
    for c in ("lift", "ci_low", "ci_high", "mde"):
        f[c] = f[c].map(lambda v: f"{v:+.1%}" if c != "mde" else f"{v:.1%}")
    f["srm_p"] = f.srm_p.map(lambda v: f"{v:.1e}" if v < 0.001 else f"{v:.3f}")
    f["p_adj"] = f.p_adj.map(lambda v: "n/a" if pd.isna(v) else f"{v:.3g}")
    return f


def main(min_effect: float | None = None) -> None:
    bar = MIN_EFFECT if min_effect is None else min_effect
    con = duckdb.connect(str(DB_PATH), read_only=True)
    dq = con.execute(_sql("analysis/data_quality_events.sql")).df()
    t = build(con, min_effect=bar)
    art = ROOT / "artifacts"; art.mkdir(exist_ok=True)
    t.to_csv(art / "scorecard.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 60)
    print(f"EXPERIMENT SCORECARD  (worth-shipping bar {bar:.0%}, Holm alpha {ALPHA}, SRM alpha {SRM_ALPHA})")
    print(f"Data quality of the event stream: {1 - dq.distinct_events.sum() / dq.raw_events.sum():.2%} duplicates removed, "
          f"{dq.late_events.sum() / dq.distinct_events.sum():.2%} of events arrived more than a day late.\n")
    print(_fmt(t).to_string(index=False))
    print("\nReasons:")
    for r in t.itertuples():
        print(f"  {r.experiment} {r.decision}: {r.reasons}")
    truth = con.execute("SELECT experiment_id AS experiment, effect_kind, nominal_relative_lift AS nominal_lift, planted_issue "
                        "FROM answer_key.ground_truth ORDER BY 1").df()
    print("\nANSWER KEY, for judging the decisions only (nothing above used it):")
    print(truth.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-effect", type=float, default=None, help="smallest relative lift worth shipping, e.g. 0.03")
    main(ap.parse_args().min_effect)
