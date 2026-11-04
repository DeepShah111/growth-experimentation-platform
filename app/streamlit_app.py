"""Growth Experimentation Platform: results viewer.

Reads only the small files in artifacts/, sql/ and docs/ (no database, no data download).
Run locally:  streamlit run app/streamlit_app.py
Set GEP_ARTIFACTS to point the app at a different artifacts folder (used by the tests).
"""
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))
import charts as C  # noqa: E402
import live_pages as LIVE  # noqa: E402  (interactive pages: analyse / plan / peeking / targeting / SQL)

ART = Path(os.environ.get("GEP_ARTIFACTS", ROOT / "artifacts"))
MEMOS = ROOT / "docs" / "memos"
st.set_page_config(page_title="Growth Experimentation Platform", layout="wide")

PAGES = ["Overview", *LIVE.PAGES, "How it works", "SQL layer", "Decision scorecard", "Trust checks", "Statistics lab", "Who gets the offer?",
         "Real data (Criteo)", "Platform validation", "Decision memos", "Limits & honesty"]


# ------------------------------------------------------------------------------------------------ helpers
def read(name: str):
    f = ART / name
    return pd.read_csv(f) if f.exists() else None


def missing(name: str, how: str = ""):
    st.info(f"`{name}` was not found in artifacts/. {how}".strip())


def chart(ch, caption: str | None = None):
    st.altair_chart(ch, width="stretch")
    if caption:
        st.caption(caption)


def explain(what: str, why: str, how_to_read: str | None = None):
    with st.container(border=True):
        st.markdown(f"**What this shows.** {what}")
        st.markdown(f"**Why it matters.** {why}")
        if how_to_read:
            st.markdown(f"**How to read it.** {how_to_read}")


def pct(x, d=1, plus=True):
    return "n/a" if pd.isna(x) else (f"{x:+.{d}%}" if plus else f"{x:.{d}%}")


def fmt_p(p):
    if pd.isna(p):
        return "n/a"
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def fmt_inr(x):
    return f"INR {x:+,.0f}"


def show_image(name: str, caption: str = ""):
    f = ROOT / "assets" / name
    if f.exists():
        st.image(str(f), caption=caption, width="stretch")


def scorecard():
    return read("scorecard.csv")


def get_row(df, col, val):
    r = df[df[col] == val]
    return r.iloc[0] if len(r) else None


# A roomier navigation panel: wider, larger labels, a clear highlight on the page you are on.
st.markdown("""
<style>
section[data-testid="stSidebar"] { width: 23rem !important; min-width: 23rem !important; }
section[data-testid="stSidebar"] > div:first-child { width: 23rem !important; }
[data-testid="stSidebar"] h1 { font-size: 1.7rem; line-height: 1.25; padding-bottom: .4rem; }
[data-testid="stSidebar"] div[role="radiogroup"] { gap: .2rem; }
[data-testid="stSidebar"] div[role="radiogroup"] label { padding: .55rem .75rem; border-radius: 10px; width: 100%; }
[data-testid="stSidebar"] div[role="radiogroup"] label p { font-size: 1.08rem; }
[data-testid="stSidebar"] div[role="radiogroup"] label:hover { background: rgba(120, 140, 255, .14); }
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) { background: rgba(120, 140, 255, .26); font-weight: 600; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { font-size: .85rem; }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("Growth Experimentation Platform")
page = st.sidebar.radio("Section", PAGES)
st.sidebar.caption("Live pages run on REAL randomised data (Hillstrom e-mail test, Criteo ads test) and on a simulated app whose true "
                   "effects are known, so the platform can be checked against the truth.")

# ================================================================================================ OVERVIEW
if page in LIVE.PAGES:
    LIVE.render(page)

elif page == "Overview":
    st.title("Growth Experimentation Platform")
    st.markdown("##### Which experiment results can we trust, what should ship, and who should get the offer?")
    st.info("Which experiment results can be trusted, what should ship, and who should get the offer. The platform runs on **real randomised "
            "data** (Hillstrom e-mail test, Criteo ads test) and on a **simulated grocery app with nine planted experiments**, where the true "
            "answer is known and the analysis can be graded. Open **Analyze an experiment** to try it: change a setting and the decision changes.")
    sc, pk = scorecard(), read("peeking_false_positive_rate.csv")
    k1, k2, k3, k4 = st.columns(4)
    if sc is not None:
        k1.metric("Experiments analysed", len(sc))
        k2.metric("Caught as untrustworthy", int((sc.decision == "DO NOT TRUST").sum()), help="Planted assignment bug in exp_006")
    if pk is not None:
        daily = pk[pk.strategy.str.contains("every day") & pk.strategy.str.contains("naive")].false_positive_rate
        once = pk[pk.strategy.str.contains("1 look")].false_positive_rate
        if len(daily) and len(once):
            k3.metric("False wins from daily peeking", pct(daily.iloc[0], plus=False), delta=f"vs {pct(once.iloc[0], plus=False)} with one look", delta_color="off")
    if sc is not None:
        r8 = get_row(sc, "experiment", "exp_008")
        if r8 is not None and pd.notna(r8.get("net_value")):
            k4.metric("Offer sent to everyone", f"INR {r8.net_value:,.2f} per user", help="Assumed INR 50 per order, INR 3 per user shown the offer")

    st.subheader("The three questions")
    a, b, c = st.columns(3)
    with a:
        with st.container(border=True):
            st.markdown("### 1. Can we trust it?")
            st.write("Before reading any result, check that the experiment was run properly: users split as planned, enough users, no peeking, no duplicate or late data.")
            st.caption("See: Trust checks, Statistics lab")
    with b:
        with st.container(border=True):
            st.markdown("### 2. What should ship?")
            st.write("A scorecard turns the statistics into one call per experiment: SHIP, ITERATE, INCONCLUSIVE or DO NOT TRUST, with the reasons and the money.")
            st.caption("See: Decision scorecard, Decision memos")
    with c:
        with st.container(border=True):
            st.markdown("### 3. Who gets the offer?")
            st.write("Models predict which users the offer helps. Sending it to everyone loses money; sending it to the right users earns it.")
            st.caption("See: Who gets the offer?")

    if sc is not None:
        st.subheader("Every experiment at a glance")
        chart(C.forest_chart(sc), "Each line is the measured lift with its 95% interval. Green = ship, amber = fix first, grey = cannot tell, red = broken. "
                                  "The dashed green line is the 3% lift worth shipping (an assumption).")

    st.subheader("How it is built")
    st.write("Raw events are cleaned in SQL, turned into one row per user, and then either read with statistics or used to train uplift models. "
             "Open **How it works** for the full picture, or **SQL layer** to see the actual queries.")
    with st.expander("Show the architecture diagram"):
        show_image("architecture.png")

# ================================================================================================ HOW IT WORKS
elif page == "How it works":
    st.header("How it works: from raw events to a decision")
    st.write("Follow the data from the top to the bottom. Statistics (no model) and machine learning branch off the same cleaned SQL tables.")
    show_image("ml_pipeline.png", "Data and model flow")
    st.subheader("Step by step")
    st.markdown("""
| # | Step | What happens | Where |
|---|---|---|---|
| 1 | **Simulate** | 100,000 users over 84 days, 9 experiments with hidden true effects, planted problems (duplicate events, late events, an assignment bug) | `src/simulation` |
| 2 | **Load** | Parquet files become DuckDB tables; the hidden answer key sits in a separate schema | `sql/schema` |
| 3 | **Clean** | Remove exact duplicate events (0.60%), flag events that arrive more than a day late (1.51%); 5,275,310 raw events become 5,243,847 | `sql/staging` |
| 4 | **Report** | 11 SQL reports: funnel, retention, sessions, running totals, top-N, RFM, streaks, data quality | `sql/analysis` |
| 5 | **Model the data** | One row per user: outcomes after assignment, behaviour in the 28 days before, profile | `sql/marts` |
| 6 | **Statistics** | Sample-ratio check, CUPED, correction for many tests, power, novelty, interaction, economics, decision | `src/experiments` |
| 7 | **Machine learning** | S, T and X uplift learners with gradient boosting, cross-fitted | `src/targeting` |
| 8 | **Evaluate** | Honest scoring on randomised data, 10 seeds, hold-out profit | `src/targeting` |
| 9 | **Real data** | Criteo ad experiment with the same checks | `src/criteo` |
""")
    st.subheader("The machine-learning models in plain words")
    st.markdown("""
| Model | The question it answers | How it works |
|---|---|---|
| **S-learner** | How many extra orders does the offer cause for this user? | One model sees the user's features plus a "got the offer" flag. Uplift = its prediction with the flag on minus off. |
| **T-learner** | Same | Two models: one trained on users who got the offer, one on users who did not. Uplift = the difference. |
| **X-learner** | Same | Uses those two models to estimate each user's missing outcome, then learns the effect from that and blends the results. |
| **CUPED** | How do we read an experiment with less noise? | Not a learned model: a simple adjustment using each user's behaviour before the experiment. |
| **Doubly-robust (AIPW)** | What is the real effect when assignment is slightly unbalanced? (Criteo) | Combines an outcome model and a propensity model so that one can be wrong without ruining the answer. |
""")
    st.caption("All three uplift learners use gradient boosting (150 trees, depth 4) on 8 features known before the experiment. "
               "Settings were fixed, not tuned. No deep learning is used.")

# ================================================================================================ SQL LAYER
elif page == "SQL layer":
    st.header("The SQL layer")
    explain("The queries that clean the raw events, build one row per user, and answer business questions. They run in DuckDB.",
            "Everything downstream (statistics and machine learning) reads the clean tables, so a mistake here would quietly poison every result. "
            "Each query has a unit test that checks it against a hand-computed answer.",
            "Pick a query. You see what it answers, the SQL itself, and the result on the 100,000-user simulation.")
    items = {
        "Cleaning: duplicates and late events": ("staging/stg_events_clean.sql", "ROW_NUMBER() to remove exact duplicates, flag late arrivals", "data_quality"),
        "Purchase funnel by device": ("analysis/funnel_by_device.sql", "Conditional aggregation on de-duplicated events", "funnel"),
        "Cohort retention": ("analysis/weekly_retention.sql", "Signup cohorts, DATE_TRUNC, retention by week", "retention"),
        "Sessions rebuilt from raw events": ("analysis/sessionization_check.sql", "LAG() gap detection and a running SUM", "sessions"),
        "Revenue trends": ("analysis/daily_revenue_trends.sql", "Running total, 7-day moving average, LAG for day and week change", "revenue"),
        "Top products per category": ("analysis/top_products_per_category.sql", "DENSE_RANK() with QUALIFY: top-N per group, ties share a rank", "products"),
        "Customer segments (RFM)": ("analysis/rfm_segments.sql", "NTILE() scoring and a share-of-total window", "rfm"),
        "Activity streaks": ("analysis/activity_streaks.sql", "Gaps and islands: longest run of consecutive days", "streaks"),
        "Per-user experiment table (the model's input)": ("marts/experiment_user_metrics.sql", "Joins, a 28-day pre-period, sorted by user for repeatable results", None),
    }
    questions = {"data_quality": "How many events are duplicated or arrive late, and is it stable over time?",
                 "funnel": "Where do users drop out between viewing, carting and buying, and does it differ by device?",
                 "retention": "Of the users who sign up, how many are still active after 1, 2, 3 ... weeks?",
                 "sessions": "If we ignore the app's own session ids, can we rebuild sessions from timestamps alone?",
                 "revenue": "How is revenue trending, smoothing out the weekly rhythm?",
                 "products": "Which products are carted most inside each category?",
                 "rfm": "Which customer groups drive the revenue?",
                 "streaks": "How many days in a row do users keep coming back?", None: "What does each assigned user look like before and after the experiment? (This table feeds the statistics and the models.)"}
    choice = st.selectbox("Query", list(items))
    path, technique, key = items[choice]
    st.markdown(f"**Question it answers:** {questions[key]}")
    st.markdown(f"**SQL technique:** {technique}")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**The SQL**")
        f = ROOT / "sql" / path
        st.code(f.read_text() if f.exists() else "-- file not found in this deployment", language="sql")
    with c2:
        st.markdown("**The result**")
        if key == "data_quality":
            d = read("sql/data_quality_events.csv")
            if d is None:
                missing("sql/data_quality_events.csv", "Run python -m src.warehouse.run_sql")
            else:
                tot = d.raw_events.sum(); dist = d.distinct_events.sum()
                m1, m2 = st.columns(2)
                m1.metric("Duplicates removed", pct((tot - dist) / tot, 2, plus=False)); m2.metric("Late (> 1 day)", pct(d.late_events.sum() / dist, 2, plus=False))
                chart(C.dq_chart(d), "Flat over time, as a healthy pipeline should be. A sudden jump would signal a tracking problem.")
        elif key == "funnel":
            d = read("sql/funnel_by_device.csv")
            chart(C.funnel_chart(d), "Web converts far worse at the last step.") if d is not None else missing("sql/funnel_by_device.csv")
        elif key == "retention":
            d = read("sql/weekly_retention.csv")
            chart(C.retention_chart(d), "Most new users drop out in the first week; a long tail stays.") if d is not None else missing("sql/weekly_retention.csv")
        elif key == "sessions":
            d = read("sql/sessionization_check.csv")
            if d is None:
                missing("sql/sessionization_check.csv")
            else:
                chart(C.sessions_chart(d), "Rebuilding sessions from gaps gives slightly fewer sessions than the app logged: some logged sessions start within 30 minutes of the previous one.")
        elif key == "revenue":
            d = read("sql/daily_revenue_trends.csv")
            chart(C.revenue_trend_chart(d), "A weekly rhythm and a gentle downward drift; the final partial day is hidden.") if d is not None else missing("sql/daily_revenue_trends.csv")
        elif key == "products":
            d = read("sql/top_products_per_category.csv")
            if d is None:
                missing("sql/top_products_per_category.csv")
            else:
                cat = st.selectbox("Category", sorted(d.category.unique()))
                sub = d[d.category == cat].sort_values("rank_in_category")
                import altair as alt
                chart(alt.Chart(sub).mark_bar(color=C.BLUE).encode(y=alt.Y("product_id:N", sort=list(sub.product_id), title="Product id"), x=alt.X("cart_adds:Q", title="Times added to cart"),
                                                                  tooltip=["product_id", "cart_adds", "rank_in_category"]).properties(height=30 * len(sub) + 40))
        elif key == "rfm":
            d = read("sql/rfm_segments.csv")
            if d is None:
                missing("sql/rfm_segments.csv")
            else:
                chart(C.rfm_chart(d), "A small group of 'champions' brings in about half of all revenue. Segment names are rule outputs on simulated data.")
        elif key == "streaks":
            d = read("sql/activity_streaks.csv")
            chart(C.streaks_chart(d), "Most users are active for only 1 to 3 days in a row (streaks above 20 days are rare and not shown).") if d is not None else missing("sql/activity_streaks.csv")
        else:
            st.info("This query builds the table the statistics and the models read: one row per assigned user, with outcomes after assignment (`orders_in`, `revenue_in`) "
                    "and behaviour from the 28 days before (`orders_pre`, `revenue_pre`, `sessions_pre`) plus profile fields.")
    st.caption("Not covered: query tuning and EXPLAIN plans, and nothing here says anything about performance at production scale.")

# ================================================================================================ SCORECARD
elif page == "Decision scorecard":
    st.header("Decision scorecard: what ships, what does not")
    sc = scorecard()
    if sc is None:
        missing("scorecard.csv", "Run python -m src.experiments.scorecard")
    else:
        explain("One row per experiment: how much the main metric moved (with its 95% interval) and the call the platform makes.",
                "A p-value alone does not decide what to ship. The scorecard also checks that the experiment can be trusted, that it had enough users, that a guardrail metric is not hurt, and that the money works.",
                "Look at where the line sits against the dashed green 3% bar. Green means ship, amber means fix something first, grey means we cannot tell, red means do not trust it.")
        counts = sc.decision.value_counts()
        cols = st.columns(4)
        for col, label in zip(cols, ["SHIP", "ITERATE", "INCONCLUSIVE (underpowered)", "DO NOT TRUST"]):
            col.metric(label, int(counts.get(label, 0)))
        chart(C.forest_chart(sc))
        st.subheader("Why did it decide that?")
        pick = st.selectbox("Pick an experiment", list(sc.experiment), format_func=lambda e: f"{e}  {sc[sc.experiment == e].iloc[0]['name']}")
        row = get_row(sc, "experiment", pick)
        colour = {"SHIP": "green", "ITERATE": "orange", "DO NOT TRUST": "red", "DO NOT SHIP (harmful)": "red"}.get(row.decision, "gray")
        st.markdown(f"### :{colour}[{row.decision}]")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Lift in main metric", pct(row.lift)); m2.metric("95% interval", f"{pct(row.ci_low)} to {pct(row.ci_high)}")
        m3.metric("Users per arm", f"{int(row.n_treat):,} / {int(row.n_ctrl):,}"); m4.metric("Smallest lift it could detect", pct(row.mde, plus=False))
        for reason in str(row.reasons).split(" | "):
            st.markdown(f"- {reason}")
        with st.expander("How the scorecard decides (the gates, in order)"):
            st.markdown("""
1. **Trust**: do the two groups have the planned sizes? If not, the assignment is broken and nothing else is read.
2. **Evidence**: the lift after variance reduction (CUPED), judged with a correction for reading many experiments at once.
3. **Power**: if nothing is significant, could the test even have seen a lift worth shipping? If not, the answer is *unknown*, not *zero*.
4. **Safety**: a second money metric (the guardrail) must not be significantly worse.
5. **Validity**: overlapping experiments must not interact; the effect must not be fading week over week.
6. **Economics**: if every treated user costs money, a real lift can still lose money overall.

The 3% worth-shipping bar and the alpha levels are choices, not facts. They live in one place in the code.
""")
        with st.expander("How did the scorecard do against the hidden truth? (my reading, not a computed score)"):
            st.table(pd.DataFrame({
                "experiment": ["exp_001", "exp_002", "exp_003", "exp_004", "exp_005", "exp_006", "exp_007", "exp_008", "exp_009"],
                "hidden truth": ["constant +8%", "constant +1.5%", "exactly 0", "novelty: +12% fading to +1%", "+18% for new users only", "0, planted assignment bug",
                                 "constant +6% (basket value)", "varies by channel and device", "hidden-segment effect"],
                "judgement": ["right", "defensible", "defensible", "MISS: weekly noise too large to see the fade", "MISS: segment-only effect", "right", "right",
                              "right: real lift, unprofitable for everyone", "defensible"]}))
            st.caption("The two misses are real limits, documented rather than hidden.")

# ================================================================================================ TRUST CHECKS
elif page == "Trust checks":
    st.header("Can we trust the data and the experiments?")
    t1, t2, t3 = st.tabs(["Were users split as planned?", "Naive vs adjusted vs the truth", "Is the data clean?"])
    with t1:
        explain("The share of users who landed in the treatment group, for every experiment. The plan was 50/50.",
                "If the split is off, the assignment is broken and any lift is meaningless. This is called a sample-ratio mismatch.",
                "Dots sit on the dashed 50% line when all is well. The red dot (exp_006) is the planted bug: low-activity treated users were dropped, which fakes a lift.")
        sc = scorecard()
        chart(C.srm_chart(sc)) if sc is not None else missing("scorecard.csv")
        st.caption("exp_004 is amber: its split is slightly unusual (p = 0.062) but not broken.")
    with t2:
        explain("For each experiment: what a simple comparison gives (naive), what the variance-reduced estimate gives (CUPED), and the true effect we built in.",
                "It checks the methods against the truth. CUPED gives the same answer with tighter intervals, but it cannot repair a broken experiment.",
                "Green diamonds are the truth. Look at exp_006: the truth is 0, yet both estimates say +13% to +19%. exp_004 and exp_005 are special cases (an effect that fades, and an effect only for new users).")
        fl = read("first_look_verdicts.csv")
        chart(C.truth_chart(fl)) if fl is not None else missing("first_look_verdicts.csv", "Run python -m src.experiments.first_look")
    with t3:
        explain("The share of duplicate events and late-arriving events, day by day, found with SQL.",
                "Duplicates inflate counts and late events change yesterday's numbers after the report is sent. A monitor like this catches pipeline problems early.",
                "The lines should be flat. The simulator plants about 0.6% duplicates and 1.5% late events, and the monitor finds both.")
        d = read("sql/data_quality_events.csv")
        chart(C.dq_chart(d)) if d is not None else missing("sql/data_quality_events.csv")

# ================================================================================================ STATISTICS LAB
elif page == "Statistics lab":
    st.header("Statistics lab: the classic ways A/B tests go wrong")
    tabs = st.tabs(["Power", "Peeking", "Many experiments", "Novelty & overlap", "Revenue tails"])
    with tabs[0]:
        explain("The chance an experiment detects a real effect, for different effect sizes and numbers of users. Lines are the formula, dots are measured by simulation.",
                "A test with too few users can miss a real effect, and 'not significant' then means 'unknown', not 'no effect'.",
                "Higher is better; the dashed green line is the usual 80% target. Solid lines use CUPED (fewer users needed), dashed lines do not.")
        d, p2 = read("power_validation.csv"), read("power_exp002.csv")
        if d is not None:
            chart(C.power_chart(d), "The 0% line is the false-positive rate of about 5%.")
        else:
            missing("power_validation.csv", "Run python -m src.experiments.studies power")
        if p2 is not None:
            r = p2.iloc[0]
            a, b, c = st.columns(3)
            a.metric("exp_002: users per arm available", f"{int(r.users_per_arm_available):,}")
            b.metric("Needed without CUPED", f"{int(r.needed_no_cuped):,}")
            c.metric("Needed with CUPED", f"{int(r.needed_with_cuped):,}")
            st.caption(f"With the users available, the smallest lift exp_002 could detect is about {pct(r.mde_available_no_cuped, plus=False)} ({pct(r.mde_available_with_cuped, plus=False)} with CUPED), "
                       "far above its real +1.5%: it was built to be underpowered.")
    with tabs[1]:
        explain("What happens when you look at an A/B test again and again and stop at the first 'significant' result, when there is NO real effect at all.",
                "Every extra look is another chance for noise to look like a win. This is the most common way teams fool themselves.",
                "The correct plan is 5% false wins. Peeking daily gives 25%. Honest fixes (stricter thresholds when you peek) bring it back near 5%.")
        d, d2 = read("peeking_false_positive_rate.csv"), read("peeking_vs_looks.csv")
        if d is not None:
            chart(C.peeking_bars(d), "Dashed green line = 5%, the rate you intended.")
        else:
            missing("peeking_false_positive_rate.csv", "Run python -m src.experiments.studies peeking")
        if d2 is not None:
            chart(C.peeking_line(d2), "The more often you peek, the more false wins.")
    with tabs[2]:
        explain("The chance of at least one false 'win' when you run many experiments that have no real effect, with and without a correction.",
                "Run 20 harmless experiments and one will look like a winner about 6 times in 10 by chance. A correction (Holm) keeps the whole family at about 5%.",
                "Red = no correction (measured), pink = the textbook formula, green = with Holm.")
        d = read("family_wise_error.csv")
        chart(C.fwer_chart(d)) if d is not None else missing("family_wise_error.csv", "Run python -m src.experiments.studies multiple")
        mv = read("multiple_testing_verdicts.csv")
        if mv is not None:
            with st.expander("Which experiments survive each correction?"):
                show = mv[["experiment", "name", "cuped_lift", "p_value", "srm_ok", "raw_p<0.05", "bonferroni", "holm", "benjamini_hochberg"]].copy()
                show["cuped_lift"] = show.cuped_lift.map(pct); show["p_value"] = show.p_value.map(fmt_p)
                st.dataframe(show, width="stretch", hide_index=True)
    with tabs[3]:
        explain("Two checks on one experiment pair. Left: does the homepage redesign's lift fade as users get used to it (novelty)? Right: does the checkout button work the same with and without the redesign (interaction)?",
                "A novelty bump looks great in week one and disappears. Two overlapping experiments can distort each other.",
                "Weekly lifts that shrink suggest novelty. If the two bars are about equal, the experiments do not interfere.")
        a, b = st.columns(2)
        with a:
            st.markdown("**Homepage redesign (exp_004): lift by week**")
            d = read("novelty_weekly_exp_004.csv")
            if d is not None:
                chart(C.weekly_chart(d), "Week 3 is lower, but the intervals overlap and the trend test cannot confirm a fade at this size (p = 0.19). 'Cannot tell' is not 'no novelty'.")
            else:
                missing("novelty_weekly_exp_004.csv", "Run python -m src.experiments.novelty")
        with b:
            st.markdown("**Checkout button (exp_001) with and without exp_004**")
            it = read("interaction_exp_001_exp_004.csv")
            if it is not None:
                chart(C.interaction_chart(it.iloc[0]), f"Interaction {it.iloc[0].interaction:+.4f} +/- {it.iloc[0].ci_half:.4f} orders per user (p = {it.iloc[0].p_value:.2f}). No interaction detected, which is not proof there is none.")
            else:
                missing("interaction_exp_001_exp_004.csv", "Run python -m src.experiments.interaction")
    with tabs[4]:
        explain("Revenue per user is zero for most users and huge for a few. Four ways to estimate the revenue lift of the free-delivery threshold (exp_007).",
                "A few big spenders make the average noisy. Different estimators trade noise against what exactly they measure.",
                "Each line is one estimator's 95% interval; the dashed green line is the true +6%. All four agree. Winsorizing caps big values, so it measures a slightly different quantity.")
        d = read("revenue_estimators_exp_007.csv")
        if d is not None:
            chart(C.estimators_chart(d, 0.06))
        else:
            missing("revenue_estimators_exp_007.csv", "Run python -m src.experiments.revenue")
        st.markdown("**How concentrated is revenue?** In this run, 74.7% of users spent nothing, and the top 1% of users hold 20.9% of all revenue (the top 10% hold 76.1%).")
        dec = read("revenue_decomposition_exp_007.csv")
        if dec is not None:
            with st.expander("More buyers or bigger baskets? (descriptive, not causal)"):
                st.dataframe(dec, width="stretch", hide_index=True)
                st.caption("Buyers are a post-treatment group, so this split describes the data but does not prove a cause.")

# ================================================================================================ WHO GETS THE OFFER
elif page == "Who gets the offer?":
    st.header("Who should get the win-back offer? (exp_008)")
    explain("The offer raised orders overall, but it helps some users a lot and hurts others. Models predict each user's response from what we knew before the experiment.",
            "Sending the offer to everyone is the lazy choice, and at the assumed costs it loses money. Sending it only to the users it helps earns money.",
            "Work through the four tabs: which model is best, who benefits, whether it pays, and whether the result is stable.")
    t1, t2, t3, t4 = st.tabs(["Which model?", "Who benefits?", "Does it pay?", "Is it stable?"])
    with t1:
        d = read("uplift_model_comparison_exp_008.csv")
        if d is None:
            missing("uplift_model_comparison_exp_008.csv", "Run python -m src.targeting.uplift --exp exp_008")
        else:
            chart(C.uplift_models_chart(d), "Extra orders per 1,000 users compared with picking users at random. Intervals that stay right of zero beat random.")
            st.markdown("All three learners beat random targeting and the simple rule 'new users first' by a wide margin. "
                        "**Do not read too much into the order among the three learners:** the intervals overlap (see 'Is it stable?').")
    with t2:
        d = read("uplift_by_decile_exp_008.csv")
        if d is None:
            missing("uplift_by_decile_exp_008.csv")
        else:
            explain("Users sorted by the model's score into ten equal groups, and what the offer really did to each group in the randomised experiment.",
                    "The model is only useful if the users it ranks high really respond more. The bottom group is actively harmed by the offer.",
                    "Bars are measured extra orders per user (green = helps, red = hurts). The dashed white line is what the model predicted.")
            chart(C.decile_chart(d))
    with t3:
        curve = read("uplift_profit_curve_exp_008.csv")
        if curve is None:
            missing("uplift_profit_curve_exp_008.csv")
        else:
            explain("Profit if you send the offer to the best-scored users first, as a function of how many users you reach. You can change the money assumptions below.",
                    "The economics decide the answer. With the defaults, reaching everyone loses money and reaching the best fifth earns it.",
                    "Green dot = the best share to reach. The red diamond (when shown) = sending to everyone. Move the sliders to see when sending to everyone becomes worthwhile.")
            s1, s2 = st.columns(2)
            value = s1.slider("Profit per extra order (INR)", 20, 150, 50, step=5)
            cost = s2.slider("Cost of showing the offer to one user (INR)", 0.0, 10.0, 3.0, step=0.5)
            pc = C.profit_curve(curve, value, cost, n_users=100_000)
            best = pc.loc[pc.profit.idxmax()]
            m1, m2, m3 = st.columns(3)
            m1.metric("Best share to reach", pct(best.fraction_treated, 0, plus=False)); m2.metric("Profit at that share", fmt_inr(best.profit))
            last = pc.iloc[-1]
            if last.fraction_treated >= 0.999:
                m3.metric("Profit if sent to everyone", fmt_inr(last.profit))
            chart(C.profit_chart(pc), "Optimistic: the best share is picked on the same data it is scored on. The honest version is in 'Is it stable?'. Assumed economics, not measured margins.")
    with t4:
        ms, runs, pb = read("uplift_multiseed_summary_exp_008.csv"), read("uplift_multiseed_runs_exp_008.csv"), read("uplift_paired_bootstrap_exp_008.csv")
        if ms is None or runs is None:
            missing("uplift_multiseed_summary_exp_008.csv", "Run python -m src.targeting.multiseed --exp exp_008")
        else:
            explain("The whole evaluation repeated with 10 different random seeds, and the profit cut-off picked on one half of the users and scored on the other half.",
                    "One run can be a lucky draw. Repeating it shows how stable the result is, and the hold-out profit removes the optimism of scoring on the same data.",
                    "Left: each dot is one seed (the white tick is the average). Right: grey bars are optimistic, blue bars are honest.")
            a, b = st.columns(2)
            with a:
                chart(C.seeds_chart(runs))
            with b:
                chart(C.holdout_chart(ms))
            best_n = ms[ms.model == "S-learner"].best_learner_in_n_seeds
            if len(best_n):
                st.markdown(f"The S-learner was the best of the three in **{int(best_n.iloc[0])} of 10 seeds**. But that is only the algorithm's randomness. "
                            "Sampling noise from having 100,000 users is what the intervals below show.")
            if pb is not None:
                st.markdown("**Is one learner really better than another?** (paired bootstrap)")
                chart(C.paired_chart(pb), "Intervals that stay right of zero mean the first learner is clearly better. S beating T is well supported, S beating X is marginal, and X versus T cannot be separated.")

# ================================================================================================ CRITEO
elif page == "Real data (Criteo)":
    st.header("Real randomised data: the Criteo ad experiment")
    eff = read("criteo_effects_robustness.csv")
    if eff is None:
        missing("criteo_effects_robustness.csv", "Run the Criteo commands (see the README).")
    else:
        explain("A real advertising experiment: users were randomly chosen to be shown ads or not, and we measure website visits and purchases. We use a 1,000,000-user sample.",
                "Simulated data can only prove so much. Here the same checks run on real data, where the true answer is unknown.",
                "Check balance first, then compare three ways of estimating the effect, then see who responds.")
        t1, t2, t3 = st.tabs(["Was it balanced?", "The effect of the ads", "Who responds?"])
        with t1:
            bal = read("criteo_balance_sorted.csv")
            if bal is not None:
                chart(C.balance_chart(bal), "All 12 features are slightly unbalanced between the groups (largest 0.052; perfect randomisation would give about 0.003). Small, but systematic: it means the naive estimate is a little biased.")
        with t2:
            o = st.radio("Outcome", ["visit", "conversion"], horizontal=True)
            chart(C.criteo_effects_chart(eff, o), "Naive = raw difference, Lin = regression-adjusted, AIPW = doubly-robust. Only about 3.6% of treated users actually saw an ad, so effects are per user assigned.")
            r = eff[eff.outcome == o].set_index("method").effect
            naive_key = [k for k in r.index if "naive" in k.lower()]
            aipw_key = [k for k in r.index if "aipw" in k.lower()]
            if naive_key and aipw_key:
                n_, a_ = r[naive_key[0]], r[aipw_key[0]]
                st.markdown(f"The adjusted (doubly-robust) effect is **{(1 - a_ / n_):.0%} lower** than the naive one ({a_ * 100:.2f} vs {n_ * 100:.2f} percentage points): part of the naive number was bias from the small imbalance.")
        with t3:
            dec = read("criteo_visit_deciles_robust.csv")
            if dec is not None:
                explain("Users ranked by a model, and the measured extra visits per user in each group.",
                        "The top group is where advertising money is best spent. A first-pass measurement overstated it; the robust one is the number to quote.",
                        "Blue = robust measurement, grey = first pass.")
                chart(C.criteo_deciles_chart(dec), "The top decile gains about 6 percentage points of visits and holds about 72% of all incremental visits (the first-pass method said 82%).")
            mod = read("criteo_uplift_models_robust.csv")
            if mod is not None:
                with st.expander("Model comparison (robust scoring)"):
                    st.dataframe(mod.round(3), width="stretch", hide_index=True)
                    st.caption("The S > X > T order repeats but the intervals overlap, so it is suggestive only.")

# ================================================================================================ VALIDATION
elif page == "Platform validation":
    st.header("Can the platform's own statistics be trusted?")
    explain("The whole analysis was repeated in 100 freshly simulated worlds of 20,000 users each, for the experiments whose true effect is known.",
            "A platform that gives confident answers must be checked itself: is it unbiased, and does its '95% interval' really contain the truth about 95% of the time?",
            "Red = the planted-bug experiment, included on purpose: the checks should flag it, and they do.")
    s, runs = read("validation_summary.csv"), read("validation_runs.csv")
    if s is None:
        missing("validation_summary.csv", "Run python -m src.experiments.validation")
    else:
        clean = s[s.experiment != "exp_006"]
        unbiased = (clean.bias.abs() < 2 * clean.bias_se).all()
        k1, k2, k3 = st.columns(3)
        k1.metric("Clean experiments unbiased?", "Yes" if unbiased else "No")
        k2.metric("Interval coverage (clean)", f"{clean.coverage.min():.0%} to {clean.coverage.max():.0%}", help="Target: about 95%")
        k3.metric("Planted bug flagged?", "Yes" if (s[s.experiment == "exp_006"].coverage < 0.5).all() else "No")
        a, b = st.columns(2)
        with a:
            st.markdown("**Do the 95% intervals contain the truth?**")
            chart(C.coverage_chart(s), "Dot = share of worlds where the interval contained the truth; line = its own uncertainty. Target: the green line at 95%.")
        with b:
            st.markdown("**Is the average estimate correct?**")
            chart(C.bias_chart(s), "Average error with +/- 2 standard errors. Lines that cross the green zero line are unbiased.")
        if runs is not None:
            st.markdown("**One experiment, 100 simulated worlds**")
            exp = st.selectbox("Experiment", [e for e in sorted(runs.experiment.unique())], format_func=lambda e: f"{e}" + ("  (planted bug)" if e == "exp_006" else ""))
            chart(C.estimates_hist(runs, exp), "Each bar counts simulated worlds; the green line is the true effect. For clean experiments the estimates centre on it; CUPED (blue) is tighter than naive (grey).")
        st.markdown("Intervals are slightly **conservative** (standard-error ratio 0.91 to 1.22): they cover a little more often than 95%, never less. CUPED narrows intervals by about 7% to 9%. "
                    "Why they are conservative was not investigated.")

# ================================================================================================ MEMOS
elif page == "Decision memos":
    st.header("Decision memos")
    st.write("What a decision-maker would actually read: a recommendation first, then the evidence, the checks, the risks and the next step.")
    files = sorted(MEMOS.glob("*.md")) if MEMOS.exists() else []
    if not files:
        st.info("No memos found in docs/memos/.")
    else:
        nice = {"exp_001_checkout_button_ship_memo": "Ship memo: checkout button (exp_001)", "exp_008_winback_targeting_memo": "Targeting memo: win-back offer (exp_008)"}
        choice = st.radio("Memo", files, format_func=lambda p: nice.get(p.stem, p.stem.replace("_", " ")), horizontal=True)
        with st.container(border=True):
            st.markdown(choice.read_text())

# ================================================================================================ LIMITS
else:
    st.header("Limits and honesty")
    st.write("What this project does not show, stated plainly.")
    st.markdown("""
- **The simulator is a test bench, not evidence about the real world.** Its answers can be checked against the truth only because I wrote the truth. The real randomised data are the Hillstrom e-mail test (64,000 customers, one retailer, 2008) and a sample of Criteo's ads test (anonymised features).
- **Profit figures use assumed economics** (INR 50 per extra order, INR 3 per user shown the offer). They are not measured margins.
- **The scorecard has known blind spots.** It reads each experiment's overall effect, so an effect that exists only for one segment (exp_005) or one that slowly fades (exp_004) can be missed.
- **The scorecard's thresholds are choices** (a 3% worth-shipping bar, alpha levels), not facts.
- **The learner ranking is not settled.** All three beat random targeting. S beats T clearly; S versus X is marginal; X versus T cannot be separated.
- **Hyper-parameters were fixed, not tuned.** No other model family was benchmarked.
- **100,000 simulated users** on a laptop-sized database says nothing about performance at production scale. Query tuning was not covered.
- **No marketplace effects.** Users do not affect each other in the simulation (no shared couriers or supply), which real delivery apps cannot ignore.
- **Prototype, not production.** There is no live data feed, scoring API or monitoring.
""")
