"""Growth Experimentation Platform: results viewer. Reads the small files in artifacts/ (no database needed).

Run locally:  streamlit run app/streamlit_app.py
Deploy:       push the repo (with artifacts/ and docs/) to GitHub, then create the app on share.streamlit.io
Set GEP_ARTIFACTS to point the app at a different artifacts folder (used by the tests).
"""
import os
from pathlib import Path

import matplotlib
import pandas as pd
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ART = Path(os.environ.get("GEP_ARTIFACTS", ROOT / "artifacts"))
MEMOS = ROOT / "docs" / "memos"
st.set_page_config(page_title="Growth Experimentation Platform", layout="wide")

PAGES = ["Overview", "Decision scorecard", "Decision memos", "Experiment verdicts", "Novelty & overlap", "Revenue tails",
         "Power", "Peeking", "Multiple testing", "Uplift targeting", "Platform validation", "Real data (Criteo)"]


def read(name: str):
    f = ART / name
    return pd.read_csv(f) if f.exists() else None


def table(name: str, df: pd.DataFrame | None = None, **kw):
    df = read(name) if df is None else df
    if df is None:
        st.info(f"`{name}` not found. Run the pipeline first.")
    else:
        st.dataframe(df, width="stretch", hide_index=True, **kw)


def image(name: str, caption: str = ""):
    f = ART / name
    if f.exists():
        st.image(str(f), caption=caption, width="stretch")


def pct(x, plus=True):
    return "n/a" if pd.isna(x) else (f"{x:+.1%}" if plus else f"{x:.1%}")


DECISION_COLOUR = {"SHIP": "#d4edda", "ITERATE": "#fff3cd", "INCONCLUSIVE (underpowered)": "#fff3cd",
                   "NO EFFECT FOUND": "#e2e3e5", "DO NOT TRUST": "#f8d7da", "DO NOT SHIP (harmful)": "#f8d7da"}


st.sidebar.title("Growth Experimentation Platform")
page = st.sidebar.radio("Section", PAGES)
st.sidebar.caption("The main dataset is simulated (a fictional grocery-delivery app). Criteo is the real-data check.")

# ------------------------------------------------------------------------------------------------ overview
if page == "Overview":
    st.title("Growth Experimentation Platform")
    st.write("""
**The problem.** A fictional grocery-delivery app has nine experiments to read and a win-back offer that costs money to send.
The growth team needs three answers: **which results can we trust, what should ship, and who should get the offer?**

**How it works.** A simulated app generates the data, so the true effect of every experiment is known. A SQL warehouse
(DuckDB) cleans the events; a scorecard applies trust gates, statistics, power and economics to every experiment; uplift
models decide who to target; and a real randomised dataset (Criteo) checks the methods on real data.

**Important.** The main dataset is simulated and the assumptions are mine (inside public benchmark ranges). Profit figures use
assumed unit economics. This is a prototype on 100,000 simulated users, not a production system.
""")
    st.subheader("What the platform found")
    st.markdown("""
- A planted assignment bug turns a **true zero effect into a fake +19%** (exp_006); the scorecard refuses to read it.
- **Peeking at results every day** raises false positives from about 5.8% (one look) to **25.0%**.
- The win-back offer (exp_008) has a **significant +7.1% lift but loses money if shown to everyone** (INR -1.29 per user at the assumed economics): the right call is to target a subset.
- Across 100 fresh simulated worlds the platform's estimates are **unbiased** and its 95% intervals cover the truth, slightly conservatively.
- Honest misses: a slowly fading effect (exp_004) and an effect that exists only in one segment (exp_005) are not caught by the overall scorecard.
""")
    st.write("Use the sidebar: scorecard, memos, novelty, revenue tails, power, peeking, uplift, validation, Criteo.")

# ------------------------------------------------------------------------------------------------ scorecard
elif page == "Decision scorecard":
    st.header("Decision scorecard: what ships, what does not")
    sc = read("scorecard.csv")
    if sc is None:
        st.info("`scorecard.csv` not found. Run `python -m src.experiments.scorecard`.")
    else:
        counts = sc.decision.value_counts()
        cols = st.columns(4)
        for col, label in zip(cols, ["SHIP", "ITERATE", "INCONCLUSIVE (underpowered)", "DO NOT TRUST"]):
            col.metric(label, int(counts.get(label, 0)))
        view = pd.DataFrame({
            "experiment": sc.experiment, "name": sc.name, "decision": sc.decision,
            "lift (CUPED)": sc.lift.map(pct), "95% CI": [f"{pct(a)} to {pct(b)}" for a, b in zip(sc.ci_low, sc.ci_high)],
            "Holm p": sc.p_adj.map(lambda v: "n/a" if pd.isna(v) else f"{v:.3g}"),
            "min detectable lift": sc.mde.map(lambda v: pct(v, plus=False)),
            "sample-ratio p": sc.srm_p.map(lambda v: f"{v:.1e}" if v < 0.001 else f"{v:.3f}")})
        styled = view.style.map(lambda v: f"background-color: {DECISION_COLOUR.get(v, 'white')}; color: black", subset=["decision"])
        st.dataframe(styled, width="stretch", hide_index=True)
        pick = st.selectbox("Why? Pick an experiment", list(sc.experiment))
        row = sc[sc.experiment == pick].iloc[0]
        st.markdown(f"**{pick} ({row['name']}): {row.decision}**")
        for reason in str(row.reasons).split(" | "):
            st.markdown(f"- {reason}")
        st.caption("Gates, in order: sample-ratio trust check, CUPED lift with Holm correction, power versus the smallest lift worth "
                   "shipping (3%, an assumption), guardrail metric, overlap/interaction and fading checks, and economics for "
                   "experiments whose treatment costs money.")
        with st.expander("How did the scorecard do against the hidden truth? (my reading, not a computed score)"):
            st.table(pd.DataFrame({
                "experiment": ["exp_001", "exp_002", "exp_003", "exp_004", "exp_005", "exp_006", "exp_007", "exp_008", "exp_009"],
                "hidden truth": ["constant +8%", "constant +1.5%", "exactly 0", "novelty: +12% fading to +1%", "+18% for new users only",
                                 "0, planted assignment bug", "constant +6% (basket value)", "varies by channel/device", "hidden-segment effect"],
                "judgement": ["right", "defensible", "defensible", "MISS: weekly noise too large to see the fade", "MISS: segment-only effect",
                              "right", "right", "right: real lift, unprofitable for everyone", "defensible"]}))

# ------------------------------------------------------------------------------------------------ memos
elif page == "Decision memos":
    st.header("Decision memos")
    files = sorted(MEMOS.glob("*.md")) if MEMOS.exists() else []
    if not files:
        st.info("No memos found in docs/memos/.")
    else:
        choice = st.selectbox("Memo", files, format_func=lambda p: p.stem.replace("_", " "))
        st.markdown(choice.read_text())

elif page == "Experiment verdicts":
    st.header("Naive vs CUPED estimates against the hidden truth")
    table("first_look_verdicts.csv")
    st.caption("SRM_p < 0.001 means the arm sizes do not match the planned split: do not trust that result "
               "(exp_006 is the planted bug). CUPED cannot repair broken randomisation.")

# ------------------------------------------------------------------------------------------------ novelty
elif page == "Novelty & overlap":
    st.header("Does the lift fade? Do overlapping experiments interact?")
    nv = read("novelty_weekly_exp_004.csv")
    st.subheader("exp_004 (homepage redesign): lift by week since each user's own assignment")
    if nv is None:
        st.info("`novelty_weekly_exp_004.csv` not found. Run `python -m src.experiments.novelty`.")
    else:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.errorbar(nv.exposure_week, nv.rel_lift, yerr=nv.ci_half, marker="o", capsize=4)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(nv.exposure_week); ax.set_xlabel("exposure week"); ax.set_ylabel("relative lift in orders")
        st.pyplot(fig)
        plt.close(fig)
        table("novelty_weekly_exp_004.csv", nv.round(4))
        st.caption("The lift looks smaller in week 3 but the intervals overlap, and the trend test cannot confirm a fade at this "
                   "sample size (p = 0.19). 'Not significant' here means 'cannot tell', not 'no novelty effect'.")
    st.subheader("exp_001 x exp_004: interaction")
    it = read("interaction_exp_001_exp_004.csv")
    if it is None:
        st.info("`interaction_exp_001_exp_004.csv` not found. Run `python -m src.experiments.interaction`.")
    else:
        table("interaction_exp_001_exp_004.csv", it.round(4))
        st.caption("Interaction = effect of exp_001 when exp_004 is on minus when it is off. Not significant does not prove the effects add.")

elif page == "Revenue tails":
    st.header("Heavy-tailed revenue (exp_007)")
    st.write("Most users spend nothing and a few spend a lot, so the mean is noisy. Four estimators of the revenue lift:")
    est = read("revenue_estimators_exp_007.csv")
    if est is not None:
        table("revenue_estimators_exp_007.csv", est.round(4))
    else:
        st.info("`revenue_estimators_exp_007.csv` not found. Run `python -m src.experiments.revenue`.")
    st.caption("Winsorizing measures a capped mean, a different quantity; it is shown beside the uncapped estimates, never instead of them.")
    with st.expander("Descriptive split: more buyers or bigger baskets?"):
        table("revenue_decomposition_exp_007.csv")
        st.caption("Buyers are a post-treatment group, so this split is descriptive, not causal.")

elif page == "Power":
    st.header("How many users do we need?")
    table("power_exp002.csv")
    image("power_curve.png", "Empirical power (solid/dashed) vs analytic formula (dotted)")
    st.caption("exp_002 was underpowered by design: 'not significant' there means 'unknown'.")
    with st.expander("Power validation grid"):
        table("power_validation.csv")

elif page == "Peeking":
    st.header("Checking results early inflates false wins")
    table("peeking_false_positive_rate.csv")
    image("peeking_inflation.png")

elif page == "Multiple testing":
    st.header("Many experiments, many chances for a false win")
    table("multiple_testing_verdicts.csv")
    table("family_wise_error.csv")

# ------------------------------------------------------------------------------------------------ uplift
elif page == "Uplift targeting":
    exp = st.selectbox("Experiment", ["exp_008", "exp_009"],
                       format_func=lambda e: {"exp_008": "exp_008: effect depends on observable channel/device",
                                              "exp_009": "exp_009: effect depends on a hidden segment (hard)"}[e])
    table(f"uplift_model_comparison_{exp}.csv")
    image(f"uplift_qini_curves_{exp}.png")
    with st.expander("Predicted vs measured uplift by decile"):
        table(f"uplift_by_decile_{exp}.csv")
    image(f"uplift_profit_curve_{exp}.png", "Profit under ASSUMED economics (INR 50/order, INR 3/treated user)")
    if exp == "exp_008":
        st.subheader("Repeated over 10 seeds, with an honest profit cut-off")
        ms = read("uplift_multiseed_summary_exp_008.csv")
        if ms is None:
            st.info("Run `python -m src.targeting.multiseed --exp exp_008` to create the multi-seed tables.")
        else:
            table("uplift_multiseed_summary_exp_008.csv", ms.round(2))
            st.caption("Qini = extra orders per 1,000 users vs random. Profit per 1,000 users (INR); the hold-out figure picks the cut-off "
                       "on half the users and scores it on the other half, so it is lower than the in-sample one.")
            pb = read("uplift_paired_bootstrap_exp_008.csv")
            if pb is not None:
                st.markdown("**Differences between learners (paired bootstrap)**")
                table("uplift_paired_bootstrap_exp_008.csv", pb.round(2))
                st.caption("S beating T is well supported; S beating X is marginal; X versus T cannot be separated. All three beat random targeting by a wide margin.")

# ------------------------------------------------------------------------------------------------ validation
elif page == "Platform validation":
    st.header("Can the platform's own statistics be trusted?")
    st.write("The whole analysis was repeated in 100 freshly simulated worlds (20,000 users each) for the experiments with a "
             "constant, known effect. exp_006 (planted assignment bug) is a positive control and should look broken.")
    va = read("validation_summary.csv")
    if va is None:
        st.info("`validation_summary.csv` not found. Run `python -m src.experiments.validation`.")
    else:
        view = va[["experiment", "estimator", "truth", "mean_estimate", "bias", "bias_se", "se_ratio", "coverage", "coverage_low",
                   "coverage_high", "reject_rate", "cuped_ci_width_vs_naive"]].round(4)
        table("validation_summary.csv", view)
        st.markdown("""
**How to read it.** Bias within about 2 standard errors of zero means unbiased. A standard-error ratio near 1 means honest
intervals (above 1 is conservative). Coverage should be near 95%. The reject rate is power for real effects and the false-positive
rate for exp_003 (truth 0). In this run the four clean experiments are unbiased, intervals cover 94% to 100% of the time and are
slightly conservative, CUPED narrows the intervals by about 7% to 9%, and the positive control exp_006 is flagged (coverage 0% to 5%).
""")

else:
    st.header("Real randomised data: Criteo Uplift")
    if not (ART / "criteo_effects_robustness.csv").exists():
        st.info("Criteo results not generated yet. Run the Criteo notebooks in Colab.")
    else:
        st.write("Effect of showing ads, estimated three ways. A perfectly randomised test would give the same answer "
                 "for all three. The naive estimate is higher, which signals a small covariate imbalance in the data.")
        table("criteo_effects_robustness.csv")
        with st.expander("Feature balance (standardised mean differences; ~0.003 expected under perfect randomisation)"):
            table("criteo_balance_sorted.csv")
        st.subheader("Who responds? Uplift by predicted decile (visit)")
        table("criteo_visit_deciles_robust.csv")
        st.caption("'measured_AIPW' is the robust measurement. The constant-propensity column overstates the top decile.")
        if (ART / "criteo_uplift_models_robust.csv").exists():
            st.subheader("Model comparison (robust scoring)")
            table("criteo_uplift_models_robust.csv")
        image("criteo_qini_visit.png", "Gain curve scored with constant propensity; see the robust table for numbers to quote")
