"""Interactive pages of the platform: change a parameter and the decision, the plan or the profit changes.

Everything here calls the same pure functions the REST API calls (src.engine), on real data first
(Hillstrom e-mail test, Criteo ads test) and on the simulator, where the true answer is known.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import charts as C  # noqa: E402

from src.config import EXPERIMENTS, SQL_DIR  # noqa: E402
from src.engine import planner, sequential  # noqa: E402
from src.engine.analyze import AnalysisConfig, AnalysisError, analyze_experiment  # noqa: E402
from src.engine.datasets import (CRITEO, HILLSTROM, build_demo_warehouse, criteo_available, experiment_in_sql,  # noqa: E402
                                 guess_columns, hillstrom_available, load_criteo, load_hillstrom, load_simulated,
                                 read_upload, simulated_spec)
from src.engine.sqlguard import lock_down, validate_readonly  # noqa: E402
from src.engine.targeting import policy_summary, profit_curve, qini_table, score_uplift, thin  # noqa: E402

PAGES = ["Analyze an experiment", "Plan an experiment", "Peeking simulator", "Live targeting", "SQL workbench"]
UPLOAD = "Upload my own CSV"
SIM = "Simulated app (known truth)"
EXP_META = {e.experiment_id: e for e in EXPERIMENTS}
COLOUR = {"SHIP": "green", "ITERATE": "orange", "INCONCLUSIVE (underpowered)": "gray", "NO EFFECT FOUND": "gray",
          "DO NOT TRUST": "red", "DO NOT SHIP (harmful)": "red"}
PRIMARY = {"exp_007": "revenue_per_user"}          # every other simulated experiment is judged on orders per user


# ------------------------------------------------------------------------------------------ cached data
@st.cache_resource(show_spinner="Building the demo warehouse (12,000 simulated users)...")
def warehouse():
    con, truth = build_demo_warehouse(return_truth=True)
    lock_down(con)
    return con, truth


@st.cache_data(show_spinner=False)
def hillstrom_df() -> pd.DataFrame:
    return load_hillstrom()


@st.cache_data(show_spinner=False)
def criteo_df() -> pd.DataFrame:
    return load_criteo(max_rows=200_000)


@st.cache_data(show_spinner=False)
def sim_df(exp_id: str) -> pd.DataFrame:
    con, _ = warehouse()
    return load_simulated(con.cursor(), exp_id)


def source_options() -> list[str]:
    opts = []
    if hillstrom_available():
        opts.append(HILLSTROM.label)
    if criteo_available():
        opts.append(CRITEO.label)
    return [*opts, SIM, UPLOAD]


def explain(what: str, why: str, how: str | None = None):
    with st.container(border=True):
        st.markdown(f"**What this does.** {what}")
        st.markdown(f"**Why it matters.** {why}")
        if how:
            st.markdown(f"**How to use it.** {how}")


def _economics_defaults(source: str, outcome: str, exp: str | None = None) -> tuple[float, float, str]:
    """(value per unit of outcome, cost per treated user, currency). Assumptions the user can change, not measured facts."""
    if source == HILLSTROM.label:
        return {"spend": (0.30, 0.15), "conversion": (30.0, 0.15), "visit": (0.5, 0.15)}.get(outcome, (1.0, 0.15)) + ("USD",)
    if source == CRITEO.label:
        return {"visit": (1.0, 0.05), "conversion": (20.0, 0.05)}.get(outcome, (1.0, 0.05)) + ("units",)
    if source == SIM:
        cost = 3.0 if exp in ("exp_008", "exp_009") else 0.0          # only the win-back offers cost money to send
        return {"orders_in": (50.0, cost), "revenue_in": (0.12, cost)}.get(outcome, (1.0, cost)) + ("INR",)
    return (1.0, 0.0, "units")


def _source_of(spec) -> str:
    return HILLSTROM.label if spec.key == "hillstrom" else CRITEO.label if spec.key == "criteo" else SIM if spec.key.startswith("sim:") else UPLOAD


# ================================================================================================ ANALYZE
def _pick_data(key: str):
    """Returns (df, spec, extra) for the dataset chosen on the Analyze page, or None if an upload is still missing."""
    source = st.selectbox("Dataset", source_options(), key=f"{key}_src",
                          help="Real randomised data first. The simulator is a test bench: its true effects are known.")
    if source == HILLSTROM.label:
        return hillstrom_df(), HILLSTROM, {}
    if source == CRITEO.label:
        return criteo_df(), CRITEO, {}
    if source == SIM:
        exp = st.selectbox("Experiment", list(EXP_META), key=f"{key}_exp",
                           format_func=lambda e: f"{e}  {EXP_META[e].name}")
        spec = simulated_spec(exp, PRIMARY.get(exp, "orders_per_user"))
        return sim_df(exp), spec, {"exp": exp}
    up = st.file_uploader("CSV with one row per user (an arm column, an outcome column, optional pre-experiment columns)",
                          type=["csv"], key=f"{key}_up")
    if up is None:
        st.info("Upload a CSV to map its columns. Nothing is stored. Up to 1,000,000 rows.")
        return None
    try:
        df = read_upload(up.getvalue())
    except ValueError as exc:
        st.error(str(exc))
        return None
    g = guess_columns(df)
    c1, c2 = st.columns(2)
    variant = c1.selectbox("Arm column", list(df.columns), index=list(df.columns).index(g["variant"]), key=f"{key}_v")
    arms = sorted(df[variant].astype(str).unique())
    if len(arms) > 12:
        st.error(f"'{variant}' has {len(arms)} distinct values; pick the column that holds the arm label.")
        return None
    control = c2.selectbox("Control arm", arms, index=arms.index(g["control"]) if g["control"] in arms else 0, key=f"{key}_c")
    nums = [c for c in df.columns if c != variant and pd.api.types.is_numeric_dtype(df[c])]
    if not nums:
        st.error("no numeric columns to use as an outcome")
        return None
    outcome = st.selectbox("Outcome column", nums, index=nums.index(g["outcome"]) if g["outcome"] in nums else 0, key=f"{key}_o")
    covs = st.multiselect("Pre-experiment columns (to reduce noise)", [c for c in nums if c != outcome], key=f"{key}_cv",
                          help="Only columns measured BEFORE assignment. A column affected by the treatment would bias the result.")
    from src.engine.datasets import DatasetSpec
    spec = DatasetSpec(key="upload", label=UPLOAD, variant=variant, control=control, outcomes={outcome: "your outcome"},
                       covariates=tuple(covs), real=True, description="Your own table.", caveats="Randomisation is assumed, not checked, "
                       "beyond the sample-ratio test.")
    return df, spec, {"upload_covs": covs}


def _render_result(res: dict, cfg: AnalysisConfig, spec, extra: dict):
    srm = res["srm"]
    c1, c2 = st.columns([1, 2])
    c1.metric("Sample-ratio check (p-value)", f"{srm['p_value']:.3g}",
              delta="split looks right" if res["srm_ok"] else "SPLIT IS BROKEN: do not read the results", delta_color="normal" if res["srm_ok"] else "inverse")
    c2.dataframe(pd.DataFrame({"users": srm["counts"], "expected": {k: round(v) for k, v in srm["expected"].items()}}).T,
                 width="stretch")
    arms = res["arms"]
    for r in arms.itertuples():
        with st.container(border=True):
            top = st.columns([1.7, 1.1, 1.7, 1.1, 1.2])
            top[0].markdown(f"### {r.arm}  \n:{COLOUR[r.decision]}[**{r.decision}**]")
            top[1].metric("Lift vs control", f"{r.lift:+.1%}", help="Relative change in the average outcome per user")
            top[2].metric("95% interval", f"{r.ci_low:+.1%} to {r.ci_high:+.1%}")
            top[3].metric("Adjusted p (Holm)", "n/a" if pd.isna(r.p_adj) else (f"{r.p_adj:.2g}"))
            top[4].metric("Detectable lift", "n/a" if pd.isna(r.mde) else f"{r.mde:.1%}", help="The smallest relative lift this sample could reliably detect, given its size and variance")
            st.write(r.reasons)
            if cfg.value_per_unit is not None and pd.notna(r.net_value):
                st.caption(f"Profit if shipped to everyone: **{cfg.currency} {r.net_value:+.3f} per user** "
                           f"(95% interval {r.net_ci_low:+.3f} to {r.net_ci_high:+.3f}), at the economics you set.")
    st.altair_chart(C.arms_forest_chart(arms, cfg.min_effect), width="stretch")
    st.caption("Dashed green line = the smallest lift you said is worth shipping. An interval that is entirely to the right of it is a clear win.")
    for n in res["notes"]:
        st.caption(f"Note: {n}")
    with st.expander("All numbers"):
        show = arms[["arm", "n_treat", "n_ctrl", "mean_treat", "mean_ctrl", "naive_lift", "naive_p", "adj_lift", "adj_p", "p_adj",
                     "ci_width_ratio", "mde", "decision"]]
        st.dataframe(show, width="stretch")
        st.download_button("Download as CSV", arms.to_csv(index=False), file_name="experiment_decision.csv", key="an_dl")
    if extra.get("exp"):
        with st.expander("Reveal the planted truth (simulator only)"):
            _, truth = warehouse()
            t = truth[truth.experiment_id == extra["exp"]].iloc[0]
            st.write(f"**Built-in effect:** {t.effect_kind}, nominal relative lift {t.nominal_relative_lift:+.1%}.")
            st.write(f"**Planted issue:** {t.planted_issue or 'none'}.  \n**Note:** {t.notes}")


def page_analyze():
    st.header("Analyze an experiment")
    explain("Takes any randomised experiment table and returns a trust-checked decision per treatment arm: sample-ratio check, "
            "relative lift with an interval, noise reduction from pre-experiment columns, multiple-testing correction, power, "
            "guardrail and profit.",
            "A lift is only worth acting on if the split was fair, the sample could detect it and it pays for itself. "
            "This is the same engine the REST API runs.",
            "Pick a dataset, then move the controls on the left. The decision on the right updates immediately.")
    picked = _pick_data("an")
    if picked is None:
        return
    df, spec, extra = picked
    left, right = st.columns([1, 2.2], gap="large")
    with left:
        st.subheader("Controls")
        outcome = st.selectbox("Outcome", list(spec.outcomes), key="an_out", format_func=lambda o: f"{o}: {spec.outcomes[o]}")
        covs = st.multiselect("Adjust for pre-experiment columns", [c for c in (spec.covariates or ()) if c != outcome],
                              default=[c for c in spec.covariates if c != outcome],
                              key=f"an_cov_{spec.key}", help="Multi-covariate CUPED: same lift, usually a narrower interval.")
        guard_opts = ["none", *[g for g in spec.guardrails if g != outcome]]
        guard = st.selectbox("Guardrail metric (must not get worse)", guard_opts, key="an_guard")
        alpha = st.slider("Significance level (family-wise)", 0.01, 0.10, 0.05, 0.01, key="an_alpha")
        bar = st.slider("Smallest lift worth shipping", 0, 30, 3, 1, format="%d%%", key="an_bar") / 100
        use_econ = st.checkbox("Include unit economics", value=spec.key != "upload", key="an_econ")
        v0, c0, cur0 = _economics_defaults(_source_of(spec), outcome, extra.get("exp"))
        value = cost = None
        currency = cur0
        if use_econ:
            value = st.number_input(f"Value of one unit of '{outcome}'", min_value=0.0, value=float(v0), step=float(max(v0 / 10, 0.01)),
                                    key=f"an_val_{spec.key}_{outcome}", help="Money per unit of the outcome: margin per order, per conversion, or per dollar of spend")
            cost = st.number_input("Cost of treating one user", min_value=0.0, value=float(c0), step=0.05, key=f"an_cost_{spec.key}")
            currency = st.text_input("Currency label", cur0, max_chars=5, key="an_cur")
        arms_all = sorted(df[spec.variant].astype(str).unique())
        split = None
        if st.checkbox("The planned split was NOT equal", key="an_split"):
            shares = {a: st.number_input(f"Planned share of '{a}' (%)", 0.0, 100.0, round(100 / len(arms_all), 1), 1.0, key=f"an_sh_{a}") for a in arms_all}
            if sum(shares.values()) > 0:
                split = {a: s / 100 for a, s in shares.items()}
    with right:
        cfg = AnalysisConfig(control=spec.control, alpha=alpha, min_effect=bar, planned_split=split, value_per_unit=value,
                             cost_per_treated=cost or 0.0, currency=currency, guardrail=None if guard == "none" else guard)
        try:
            res = analyze_experiment(df, spec.variant, outcome, covs, cfg)
        except AnalysisError as exc:
            st.error(str(exc))
            return
        st.caption(f"{spec.description}  \n*Caveat:* {spec.caveats}")
        _render_result(res, cfg, spec, extra)


# ================================================================================================ PLAN
def page_plan():
    st.header("Plan an experiment")
    explain("Works out how many users and how many days a test needs before you run it, and what you could detect with the traffic you have.",
            "Most 'inconclusive' results are tests that were never able to see the effect. Planning turns that into a decision made in advance.",
            "Set the baseline and the lift you care about. Everything updates as you move a slider.")
    presets = {"Type my own numbers": None}
    if hillstrom_available():
        h = hillstrom_df()
        ctrl = h[h.segment == HILLSTROM.control]
        presets["Hillstrom control: conversion rate"] = ("rate", float(ctrl.conversion.mean()), None)
        presets["Hillstrom control: visit rate"] = ("rate", float(ctrl.visit.mean()), None)
        presets["Hillstrom control: spend per user"] = ("mean", float(ctrl.spend.mean()), float(ctrl.spend.std()))
    pick = st.selectbox("Start from", list(presets), key="pl_pre")
    pre = presets[pick]
    c1, c2 = st.columns(2, gap="large")
    with c1:
        kind_label = st.radio("What are you measuring?", ["A rate (conversion, click)", "An average per user (orders, revenue)"],
                              index=0 if pre is None or pre[0] == "rate" else 1, key=f"pl_kind_{pick}")
        kind = "rate" if kind_label.startswith("A rate") else "mean"
        if kind == "rate":
            base = st.number_input("Baseline rate (control)", 0.0001, 0.9999, float(pre[1]) if pre and pre[0] == "rate" else 0.05, 0.001,
                                   format="%.4f", key=f"pl_base_{pick}")
            sd = None
        else:
            base = st.number_input("Baseline average per user", 0.0001, 1e9, float(pre[1]) if pre and pre[0] == "mean" else 10.0, key=f"pl_mean_{pick}")
            sd = st.number_input("Standard deviation per user", 0.0001, 1e9, float(pre[2]) if pre and pre[0] == "mean" and pre[2] else base * 2.0,
                                 key=f"pl_sd_{pick}", help="Revenue and counts are skewed: use the real SD from past data")
        lift = st.slider("Smallest relative lift you want to detect", 1, 100, 10, 1, format="%d%%", key="pl_lift") / 100
        arms = st.slider("Number of arms (including control)", 2, 6, 2, 1, key="pl_arms")
    with c2:
        alpha = st.slider("Significance level", 0.01, 0.10, 0.05, 0.01, key="pl_alpha")
        power = st.slider("Power (chance of detecting a real effect)", 0.5, 0.95, 0.8, 0.05, key="pl_power")
        rho = st.slider("Correlation with a pre-experiment covariate (CUPED)", 0.0, 0.9, 0.0, 0.05, key="pl_rho",
                        help="0 = no adjustment. 0.5 cuts the needed sample by about 25%.")
        daily = st.number_input("Users entering the test per day", 1.0, 1e8, 2000.0, 100.0, key="pl_daily")
    try:
        p = planner.plan(kind, base, lift, sd, alpha, power, rho, daily, arms)
    except ValueError as exc:
        st.error(str(exc))
        return
    m = st.columns(4)
    m[0].metric("Users per arm", f"{p['n_per_arm']:,}")
    m[1].metric("Users in total", f"{p['total_users']:,}")
    m[2].metric("Days of traffic", f"{p['days']:,}")
    m[3].metric("Run for (whole weeks)", f"{p['weeks']} week{'s' if p['weeks'] != 1 else ''}", help="Round up to full weeks: weekday and weekend behaviour differ")
    if arms > 2:
        st.caption(f"With {arms} arms the significance level is split across {arms - 1} comparisons with control (Bonferroni): {p['alpha_used']:.3f} each.")
    if p["weeks"] > 8:
        st.warning("That is a long test. Accept a larger minimum lift, add a covariate, or cut the number of arms.")
    lifts = np.linspace(lift / 4, lift * 2.5, 30)
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("**Power at this sample size**")
        st.altair_chart(C.live_power_chart(planner.power_curve(kind, base, p["n_per_arm"], lifts, sd, p["alpha_used"], rho), lift), width="stretch")
        st.caption("Dashed amber = the lift you asked for. Green = 80% power. Smaller true effects than you planned for are likely to be missed.")
    with b:
        st.markdown("**Days needed for each minimum lift**")
        dd = planner.days_curve(kind, base, np.linspace(max(lift / 3, 0.005), lift * 2.5, 25), daily, sd, alpha, power, rho, arms)
        st.altair_chart(C.days_chart(dd), width="stretch")
        st.caption("Halving the lift you want to detect roughly quadruples the days.")
    st.subheader("Or: what can I detect in a fixed time?")
    days = st.slider("Days available", 1, 90, 14, 1, key="pl_days")
    n_arm = daily * days / arms
    mde = planner.mde_for_n(kind, base, n_arm, sd, p["alpha_used"], power, rho)
    st.write(f"With {days} days you reach about **{n_arm:,.0f} users per arm**, so the smallest lift you can reliably detect is "
             f"**{mde:.1%}** (relative to the baseline). Anything smaller will look 'not significant' even if it is real.")


# ================================================================================================ PEEKING
def page_peeking():
    st.header("Peeking simulator")
    explain("Runs thousands of A/A tests (no real effect) and counts how often a stopping rule still declares a win.",
            "Checking a dashboard daily and stopping at the first p < 0.05 turns a 5% false-positive rate into something much larger.",
            "Move the number of looks. Compare the naive rule with the boundaries that keep the error at 5%.")
    c1, c2 = st.columns(2)
    looks = c1.slider("How many times you look at the result", 1, 30, 10, 1, key="pk_looks")
    alpha = c2.slider("Significance level", 0.01, 0.10, 0.05, 0.01, key="pk_alpha")
    t0 = time.time()
    res = sequential.peeking_study(looks, alpha, n_sims=30_000)
    st.altair_chart(C.peeking_bars(res), width="stretch")
    naive = float(res.false_positive_rate.iloc[1]) if len(res) > 1 else float(res.false_positive_rate.iloc[0])
    st.write(f"With **{looks} looks** and the naive rule, about **{naive:.0%}** of tests with no real effect are called wins "
             f"(target {alpha:.0%}). Simulated in {time.time() - t0:.1f}s from 30,000 A/A tests.")
    st.markdown("**False wins as the number of looks grows**")
    st.altair_chart(C.peeking_line(sequential.fpr_by_looks(30, alpha, 15_000)), width="stretch")
    st.caption("Fixes: decide the sample size in advance (Plan page), look once, or use a sequential boundary such as O'Brien-Fleming that is strict early and relaxed late.")


# ================================================================================================ TARGETING
@st.cache_data(show_spinner="Training and cross-fitting the uplift models (a few seconds)...", max_entries=6)
def _score(source: str, treat_arm: str, outcome: str, exp: str | None) -> dict:
    if source == HILLSTROM.label:
        df, spec, control = hillstrom_df(), HILLSTROM, HILLSTROM.control
    elif source == CRITEO.label:
        df, spec, control = criteo_df(), CRITEO, CRITEO.control
    else:
        df, spec, control = sim_df(exp), simulated_spec(exp, "orders_per_user"), "control"
    res = score_uplift(df, spec.variant, treat_arm, control, outcome, spec.features, spec.categorical, max_rows=40_000)
    res.pop("frame")
    res["qini_tbl"] = qini_table(res, reps=60)
    return res


def page_targeting():
    st.header("Live targeting")
    explain("Trains uplift models (S, T and X learners, cross-fitted so every user is scored by a model that never saw them) and shows "
            "what it would earn to treat only the users the model picks, at the value and cost you set.",
            "A treatment that wins on average can still lose money, and a model can look good by luck. The page shows both honestly.",
            "Choose the data, then move value and cost. The profit curve and the recommendation update instantly.")
    opts = [o for o in source_options() if o != UPLOAD]
    source = st.selectbox("Dataset", opts, key="tg_src")
    exp = None
    if source == HILLSTROM.label:
        treat = st.radio("Treatment arm", ["Mens E-Mail", "Womens E-Mail"], horizontal=True, key="tg_arm")
        outcome = st.selectbox("Outcome", ["spend", "conversion", "visit"], key="tg_out_h")
    elif source == CRITEO.label:
        treat = "treatment"
        outcome = st.selectbox("Outcome", ["visit", "conversion"], key="tg_out_c")
    else:
        exp = st.radio("Experiment", ["exp_008", "exp_009"], horizontal=True, key="tg_exp",
                       format_func=lambda e: {"exp_008": "exp_008 learnable heterogeneity", "exp_009": "exp_009 hidden segments (hard)"}[e])
        treat = "treatment"
        outcome = st.selectbox("Outcome", ["orders_in", "revenue_in"], key="tg_out_s")
    res = _score(source, treat, outcome, exp)
    st.caption(f"{res['n']:,} users scored (random sample, half-and-half cross-fitting). Average effect of the treatment: "
               f"{res['ate']:+.4f} per user.")
    v0, c0, cur = _economics_defaults(source, outcome, exp)
    a, b = st.columns(2)
    value = a.slider(f"Value of one unit of '{outcome}' ({cur})", 0.0, float(max(v0 * 4, 1.0)), float(v0), float(max(v0 / 50, 0.005)), key=f"tg_val_{source}_{outcome}_{exp}")
    cost = b.slider(f"Cost of treating one user ({cur})", 0.0, float(max(c0 * 6, 1.0)), float(c0), 0.01, key=f"tg_cost_{source}_{exp}")
    model = st.radio("Targeting model", ["S-learner", "T-learner", "X-learner"], horizontal=True, index=1, key="tg_model")
    summ = policy_summary(res, model, value, cost)
    q = res["qini_tbl"].set_index("model")
    works = bool(q.loc[model, "beats_random"])
    h = summ["holdout"]
    m = st.columns(4)
    m[0].metric("Best reach (in-sample)", f"{summ['best_share']:.0%} of users")
    m[1].metric(f"Profit per 1,000, targeted ({cur})", f"{h['profit_holdout']:,.1f}", help="Cut-off chosen on one half of users, scored on the other half")
    m[2].metric(f"Profit per 1,000, treat everyone ({cur})", f"{h['treat_everyone_holdout']:,.1f}")
    m[3].metric("Treat nobody", "0.0")
    gain = h["profit_holdout"] - h["treat_everyone_holdout"]
    if not works:
        st.warning(f"**The {model} does not clearly beat random targeting on this data** (its interval versus random includes zero). "
                   "Do not claim a targeting win; decide between treating everyone and treating nobody using the economics above.")
    elif gain > 0:
        st.success(f"Targeting the model's top {h['share_treated_on_B']:.0%} earns about **{gain:,.1f} {cur} more per 1,000 users** than treating everyone, "
                   "on users the cut-off was not chosen on.")
    else:
        st.info("The model ranks users better than random, but at these economics treating everyone is just as good or better.")
    curve = thin(profit_curve(res["scores"][model].to_numpy(), res["tau"], value, cost))
    st.altair_chart(C.live_profit_chart(curve, cur), width="stretch")
    st.caption("Green dot = best reach on this sample (optimistic: picked on the data it is scored on). Red diamond = treat everyone.")
    st.markdown("**Does any model beat random targeting?**")
    st.altair_chart(C.qini_live_chart(res["qini_tbl"]), width="stretch")
    st.caption("Extra outcome per 1,000 users vs the average user, with a bootstrap interval. A model only counts if it clears the 'random' row clearly.")


# ================================================================================================ SQL
def _run_custom(con, sql: str, explain_plan: bool) -> pd.DataFrame:
    clean = validate_readonly(sql)
    cur = con.cursor()
    if explain_plan and not clean.lower().startswith("explain"):
        clean = "EXPLAIN " + clean
    if clean.lower().startswith(("select", "with", "from")):
        return cur.execute(f"SELECT * FROM ({clean}) AS q LIMIT 1000").df()
    return cur.execute(clean).df()


def page_sql():
    st.header("SQL workbench")
    explain("A real DuckDB warehouse of 12,000 simulated users (users, sessions, events, orders, experiments, assignments) that runs the "
            "project's SQL live, runs a whole experiment analysis in pure SQL, and lets you write your own read-only queries.",
            "SQL is where experiment data starts. Every number on the other pages comes from queries like these.",
            "Pick a tab. The custom box is read-only, runs with file and network access switched off, and returns at most 1,000 rows.")
    con, _ = warehouse()
    t1, t2, t3 = st.tabs(["Project queries", "Experiment analysis in pure SQL", "Write your own"])
    with t1:
        from src.warehouse.run_sql import run_query
        names = sorted(f.stem for f in (SQL_DIR / "analysis").glob("*.sql") if f.stem != "experiment_in_sql")
        q = st.selectbox("Query", names, key="sq_q")
        st.code((SQL_DIR / "analysis" / f"{q}.sql").read_text(), language="sql")
        t0 = time.time()
        try:
            out = run_query(con.cursor(), q)
            st.caption(f"{len(out):,} rows in {time.time() - t0:.2f}s on the demo warehouse")
            st.dataframe(out.head(200), width="stretch")
        except Exception as exc:  # noqa: BLE001 - show any engine error instead of a stack trace
            st.error(f"query failed: {exc}")
    with t2:
        st.write("The per-user table, the CUPED adjustment and the lift with its interval, computed with `COVAR_SAMP`, `VAR_SAMP` and "
                 "window-free aggregates. The Python engine must give the same answer.")
        exp = st.selectbox("Experiment", list(EXP_META), key="sq_exp", format_func=lambda e: f"{e}  {EXP_META[e].name}")
        sql = experiment_in_sql()
        with st.expander("Show the SQL"):
            st.code(sql, language="sql")
        sql_row = con.cursor().execute(sql, {"exp_id": exp}).df().iloc[0]
        df = sim_df(exp)
        py = analyze_experiment(df, "variant", "orders_in", ["orders_pre"], AnalysisConfig(control="control"))["arms"].iloc[0]
        cmp = pd.DataFrame({"SQL": [sql_row.cuped_lift, sql_row.cuped_half_width, sql_row.naive_lift],
                            "Python engine": [py.adj_lift, py.adj_half, py.naive_lift]},
                           index=["CUPED lift", "CUPED 95% half-width", "Naive lift"])
        st.dataframe(cmp.style.format("{:.6f}"), width="stretch")
        agree = bool(np.allclose(cmp["SQL"], cmp["Python engine"], rtol=1e-9, atol=1e-12))
        (st.success if agree else st.error)("SQL and Python agree to 9 digits." if agree else "SQL and Python disagree: investigate.")
        st.caption("Orders per user is used as the metric here. n = " f"{int(sql_row.n_treat):,} treatment, {int(sql_row.n_ctrl):,} control.")
    with t3:
        schema = con.cursor().execute("SELECT table_name, column_name, data_type FROM information_schema.columns "
                                      "WHERE table_schema = 'main' ORDER BY table_name, ordinal_position").df()
        with st.expander("Tables and columns"):
            st.dataframe(schema.groupby("table_name").agg(columns=("column_name", lambda s: ", ".join(s))).reset_index(), width="stretch")
        default = ("SELECT device, COUNT(*) AS sessions,\n       ROUND(AVG(duration_sec), 1) AS avg_seconds\n"
                   "FROM sessions\nGROUP BY device\nORDER BY sessions DESC")
        sql = st.text_area("Your query (read-only)", default, height=170, key="sq_sql")
        want_plan = st.checkbox("Show the execution plan (EXPLAIN) instead of the result", key="sq_plan")
        if st.button("Run", key="sq_run", type="primary"):
            try:
                out = _run_custom(con, sql, want_plan)
                st.dataframe(out, width="stretch")
                st.caption(f"{len(out):,} rows shown")
            except ValueError as exc:
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001
                st.error(f"query failed: {exc}")


# ================================================================================================ router
def render(page: str):
    {"Analyze an experiment": page_analyze, "Plan an experiment": page_plan, "Peeking simulator": page_peeking,
     "Live targeting": page_targeting, "SQL workbench": page_sql}[page]()
