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
from scipy import stats

from src.config import COST_PER_TREATED, DB_PATH, ROOT, SQL_DIR, VALUE_PER_ORDER
from src.experiments.first_look import analyse
from src.experiments.interaction import interaction_effect
from src.experiments.novelty import decay_test, usable_weeks, weekly_effects
from src.experiments.stats_utils import holm, holm_adjust, mde_rel

SRM_ALPHA = 0.001          # sample-ratio check: below this the split is broken
ALPHA = 0.05               # family-wise error rate for the Holm correction
MIN_EFFECT = 0.03          # smallest relative lift worth shipping (a business assumption)
TREND_ALPHA = 0.05         # weekly-trend test for a fading effect
INTERACTION_ALPHA = 0.05
COST_PER_TREATED_USER = {"exp_008": COST_PER_TREATED}   # experiments where each treated user costs money (INR)


def net_value_per_user(y_treat: np.ndarray, y_ctrl: np.ndarray, value_per_order: float, cost_per_treated: float):
    """Profit per user of rolling the change out to everyone: value x (extra orders per user) - cost. Welch 95% CI."""
    d = y_treat.mean() - y_ctrl.mean()
    se = np.sqrt(y_treat.var(ddof=1) / len(y_treat) + y_ctrl.var(ddof=1) / len(y_ctrl))
    return value_per_order * d - cost_per_treated, value_per_order * (d - 1.96 * se) - cost_per_treated, \
        value_per_order * (d + 1.96 * se) - cost_per_treated


# ---------------------------------------------------------------------------------- the decision rule
def decide(r: dict) -> tuple[str, list[str]]:
    """Pure function of one scorecard row. Returns (decision, reasons)."""
    if r["srm_p"] < SRM_ALPHA:
        return "DO NOT TRUST", [f"sample ratio mismatch (p = {r['srm_p']:.1e}): arms are {r['n_treat']:,} vs {r['n_ctrl']:,}, "
                                "so assignment is broken and the result cannot be read"]
    notes = []
    if r["srm_p"] < 0.10:
        notes.append(f"sample-ratio p = {r['srm_p']:.3f} is borderline; worth a look at the assignment code")
    if r["significant"] and r["lift"] < 0:
        return "DO NOT SHIP (harmful)", [f"significantly negative lift {r['lift']:+.1%} (Holm-adjusted p = {r['p_adj']:.3g})"]
    if r["significant"]:
        blockers = []
        if r["guardrail_breach"]:
            blockers.append(f"guardrail metric is significantly negative ({r['guardrail_lift']:+.1%})")
        if r["interaction_p"] is not None and r["interaction_p"] < INTERACTION_ALPHA:
            blockers.append(f"overlapping experiment {r['overlap_with']} interacts with it (p = {r['interaction_p']:.3f})")
        if r["trend_p"] is not None and r["trend_slope"] < 0 and r["trend_p"] < TREND_ALPHA:
            blockers.append(f"effect is fading week over week (trend p = {r['trend_p']:.3f}): extend and re-read")
        if r.get("net_ci_high") is not None and not np.isnan(r["net_ci_high"]) and r["net_ci_high"] < 0:
            blockers.append(f"the lift is real but unprofitable at the assumed economics: INR {r['net_value']:+.2f} per user "
                            f"(95% CI {r['net_ci_low']:+.2f} to {r['net_ci_high']:+.2f}); do not ship to everyone, target a subset (see src.targeting)")
        if blockers:
            return "ITERATE", blockers
        notes.insert(0, f"lift {r['lift']:+.1%} (95% CI {r['ci_low']:+.1%} to {r['ci_high']:+.1%}), Holm-adjusted p = {r['p_adj']:.3g}")
        if r.get("net_value") is not None and not np.isnan(r["net_value"]):
            notes.append(f"profit at the assumed economics: INR {r['net_value']:+.2f} per user (95% CI {r['net_ci_low']:+.2f} to {r['net_ci_high']:+.2f})")
        if r["ci_low"] < MIN_EFFECT:
            notes.append(f"the interval's low end ({r['ci_low']:+.1%}) is below the {MIN_EFFECT:.0%} worth-shipping bar")
        if r["overlap_with"] and r["interaction_p"] is not None:
            notes.append(f"overlaps {r['overlap_with']}; no interaction detected (p = {r['interaction_p']:.2f}), which is not proof there is none")
        if r["latest_week_lift"] is not None and r["latest_week_ci_half"] is not None and r["latest_week_lift"] - r["latest_week_ci_half"] <= 0:
            notes.append(f"latest full week lift {r['latest_week_lift']:+.1%} +/- {r['latest_week_ci_half']:.1%} no longer clears zero "
                         "(not a significant fade, but watch it)")
        return "SHIP", notes
    # not significant
    if r["underpowered"]:
        return "INCONCLUSIVE (underpowered)", [f"lift {r['lift']:+.1%} not significant, and this sample cannot detect lifts below "
                                              f"{r['mde']:.1%} (worth-shipping bar is {MIN_EFFECT:.0%}): unknown, not zero"] + notes
    return "NO EFFECT FOUND", [f"lift {r['lift']:+.1%} (95% CI {r['ci_low']:+.1%} to {r['ci_high']:+.1%}); the test could have "
                              f"detected {r['mde']:.1%}"] + notes


# ---------------------------------------------------------------------------------- building the table
def _sql(name: str) -> str:
    return (SQL_DIR / name).read_text()


def build(con: duckdb.DuckDBPyConnection, min_users_week: int = 500) -> pd.DataFrame:
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
        row["underpowered"] = row["mde"] > MIN_EFFECT
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
    out = [decide(r) for r in t.to_dict("records")]
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
    global MIN_EFFECT
    if min_effect is not None:
        MIN_EFFECT = min_effect
    con = duckdb.connect(str(DB_PATH), read_only=True)
    dq = con.execute(_sql("analysis/data_quality_events.sql")).df()
    t = build(con)
    art = ROOT / "artifacts"; art.mkdir(exist_ok=True)
    t.to_csv(art / "scorecard.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 60)
    print(f"EXPERIMENT SCORECARD  (worth-shipping bar {MIN_EFFECT:.0%}, Holm alpha {ALPHA}, SRM alpha {SRM_ALPHA})")
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
