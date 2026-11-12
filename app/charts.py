"""Chart builders for the results viewer. Pure functions: DataFrame in, Altair chart out (no Streamlit here),
so every chart can be tested and rendered to an image on its own."""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

GREEN, AMBER, RED, GREY, BLUE, PURPLE = "#2e9e5b", "#e0a100", "#d64545", "#8a94a6", "#4c9be8", "#9b6bd6"
DECISION_COLOUR = {"SHIP": GREEN, "ITERATE": AMBER, "INCONCLUSIVE (underpowered)": GREY, "NO EFFECT FOUND": GREY,
                   "DO NOT TRUST": RED, "DO NOT SHIP (harmful)": RED}
H = 300


def _rule(x, color="#888", dash=(4, 4)):
    return alt.Chart(pd.DataFrame({"x": [x]})).mark_rule(color=color, strokeDash=list(dash)).encode(x="x:Q")


def drop_partial_last_day(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """The simulated window ends mid-day, so the final row is a partial day. Hide it if it is under half a normal day."""
    if len(df) > 8 and df[col].iloc[-1] < 0.5 * df[col].iloc[-8:-1].median():
        return df.iloc[:-1]
    return df


def _hrule(y, color="#888", dash=(4, 4)):
    return alt.Chart(pd.DataFrame({"y": [y]})).mark_rule(color=color, strokeDash=list(dash)).encode(y="y:Q")


# ------------------------------------------------------------------------------------------------ scorecard
def forest_chart(sc: pd.DataFrame, bar: float = 0.03) -> alt.Chart:
    d = sc.copy()
    d["label"] = d.experiment + "  " + d.name
    d["lift_pct"], d["lo_pct"], d["hi_pct"] = d.lift, d.ci_low, d.ci_high
    d["trusted"] = np.where(d.decision == "DO NOT TRUST", "not readable", "readable")
    order = list(d.label)
    present = [k for k in DECISION_COLOUR if k in set(d.decision)]
    colour = alt.Color("decision:N", scale=alt.Scale(domain=present, range=[DECISION_COLOUR[k] for k in present]),
                       legend=alt.Legend(title="Decision", orient="bottom"))
    y = alt.Y("label:N", sort=order, title=None)
    tip = ["experiment", "name", "decision", alt.Tooltip("lift:Q", format="+.1%"), alt.Tooltip("ci_low:Q", format="+.1%"),
           alt.Tooltip("ci_high:Q", format="+.1%")]
    op = alt.condition(alt.datum.decision == "DO NOT TRUST", alt.value(0.35), alt.value(1.0))
    rng = alt.Chart(d).mark_rule(strokeWidth=3).encode(x=alt.X("ci_low:Q", title="Lift in the main metric (95% interval)", axis=alt.Axis(format="%")),
                                                       x2="ci_high:Q", y=y, color=colour, opacity=op, tooltip=tip)
    pts = alt.Chart(d).mark_point(filled=True, size=110).encode(x="lift:Q", y=y, color=colour, opacity=op, tooltip=tip)
    bad = d[d.decision == "DO NOT TRUST"]
    flag = alt.Chart(bad).mark_text(align="left", dx=8, color=RED, fontSize=11).encode(x="ci_high:Q", y=y, text=alt.value("do not trust: broken assignment"))
    zero = _rule(0, "#999", (0, 0))
    barline = _rule(bar, GREEN, (5, 4))
    txt = alt.Chart(pd.DataFrame({"x": [bar], "t": [f"{bar:.0%} worth-shipping bar"]})).mark_text(
        align="left", dx=4, dy=-6, color=GREEN, fontSize=11, baseline="bottom").encode(x="x:Q", text="t:N", y=alt.value(0))
    return (zero + barline + rng + pts + txt + flag).properties(height=34 * len(d) + 60)


def srm_chart(sc: pd.DataFrame) -> alt.Chart:
    d = sc.copy()
    d["label"] = d.experiment + "  " + d["name"]
    d["share"] = d.n_treat / (d.n_treat + d.n_ctrl)
    d["status"] = np.where(d.srm_p < 0.001, "FAIL: arms do not match the planned 50/50", np.where(d.srm_p < 0.1, "borderline", "ok"))
    scale = alt.Scale(domain=["ok", "borderline", "FAIL: arms do not match the planned 50/50"], range=[GREEN, AMBER, RED])
    y = alt.Y("label:N", sort=list(d.label), title=None)
    base = alt.Chart(d).encode(y=y)
    pts = base.mark_point(filled=True, size=140).encode(x=alt.X("share:Q", scale=alt.Scale(domain=[0.40, 0.60]), title="Share of users in the treatment arm", axis=alt.Axis(format="%")), color=alt.Color("status:N", scale=scale, legend=alt.Legend(title="Sample-ratio check", orient="bottom")),
                                                        tooltip=["experiment", alt.Tooltip("share:Q", format=".1%"), alt.Tooltip("srm_p:Q", format=".2e")])
    return (_rule(0.5, "#999") + pts).properties(height=30 * len(d) + 50)


def truth_chart(fl: pd.DataFrame) -> alt.Chart:
    rows = []
    for _, r in fl.iterrows():
        label = f"{r['exp']}  {r['name']}"
        rows.append({"label": label, "who": "Naive", "value": r["naive_lift"], "lo": r["naive_lift"] - r["naive_CI±"], "hi": r["naive_lift"] + r["naive_CI±"]})
        rows.append({"label": label, "who": "CUPED (adjusted)", "value": r["cuped_lift"], "lo": r["cuped_lift"] - r["cuped_CI±"], "hi": r["cuped_lift"] + r["cuped_CI±"]})
        if pd.notna(r["true_nominal"]):
            rows.append({"label": label, "who": "Hidden truth", "value": r["true_nominal"], "lo": np.nan, "hi": np.nan})
    d = pd.DataFrame(rows)
    order = list(dict.fromkeys(d.label))
    who = ["Hidden truth", "Naive", "CUPED (adjusted)"]
    y = alt.Y("label:N", sort=order, title=None)
    off = alt.YOffset("who:N", sort=who)
    col = alt.Color("who:N", scale=alt.Scale(domain=who, range=[GREEN, GREY, BLUE]), legend=alt.Legend(title=None, orient="bottom"))
    err = alt.Chart(d.dropna(subset=["lo"])).mark_rule(strokeWidth=2).encode(
        x=alt.X("lo:Q", title="Estimated lift (95% interval)", axis=alt.Axis(format="%")), x2="hi:Q", y=y, yOffset=off, color=col)
    pts = alt.Chart(d).mark_point(filled=True, size=90).encode(
        x="value:Q", y=y, yOffset=off, color=col, shape=alt.Shape("who:N", scale=alt.Scale(domain=who, range=["diamond", "circle", "square"]), legend=None),
        tooltip=["label", "who", alt.Tooltip("value:Q", format="+.1%")])
    return (_rule(0, "#999", (0, 0)) + err + pts).properties(height=44 * len(order) + 60)


# ------------------------------------------------------------------------------------------------ statistics lab
def peeking_bars(df: pd.DataFrame) -> alt.Chart:
    d = df.copy(); d["over"] = d.false_positive_rate > 0.075
    bars = alt.Chart(d).mark_bar().encode(
        x=alt.X("false_positive_rate:Q", title="Share of A/A tests (no real effect) declared a 'win'", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 0.31])),
        y=alt.Y("strategy:N", sort="-x", title=None, axis=alt.Axis(labelLimit=330)),
        color=alt.condition(alt.datum.false_positive_rate > 0.075, alt.value(RED), alt.value(BLUE)),
        tooltip=["strategy", alt.Tooltip("false_positive_rate:Q", format=".1%")])
    txt = alt.Chart(d).mark_text(align="left", dx=5, color="#e6e6e6", fontWeight="bold").encode(x="false_positive_rate:Q", y=alt.Y("strategy:N", sort="-x", axis=alt.Axis(labelLimit=330)), text=alt.Text("false_positive_rate:Q", format=".1%"))
    return (bars + txt + _rule(0.05, GREEN)).properties(height=40 * len(d) + 90)


def peeking_line(df: pd.DataFrame) -> alt.Chart:
    line = alt.Chart(df).mark_line(point=True, color=RED).encode(
        x=alt.X("n_looks:Q", title="How many times you peek and stop at the first 'significant' result"),
        y=alt.Y("false_positive_rate:Q", title="False-positive rate", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, max(0.3, float(df.false_positive_rate.max()) * 1.1)])),
        tooltip=["n_looks", alt.Tooltip("false_positive_rate:Q", format=".1%")])
    return (line + _hrule(0.05, GREEN)).properties(height=H)


def power_chart(df: pd.DataFrame) -> alt.Chart:
    d = df.copy(); d["lift"] = (d.true_lift * 100).round(1).astype(str) + "% true lift"
    d["method"] = np.where(d.cuped, "with CUPED", "without CUPED")
    lines = alt.Chart(d).mark_line().encode(
        x=alt.X("n_per_arm:Q", title="Users per arm", axis=alt.Axis(format="~s")),
        y=alt.Y("analytic_power:Q", title="Chance of detecting the effect (power)", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("lift:N", title=None), strokeDash=alt.StrokeDash("method:N", title=None))
    pts = alt.Chart(d).mark_point(filled=True, size=35).encode(x="n_per_arm:Q", y="empirical_power:Q", color="lift:N",
                                                              tooltip=["lift", "method", "n_per_arm", alt.Tooltip("empirical_power:Q", format=".0%"), alt.Tooltip("analytic_power:Q", format=".0%")])
    return (lines + pts + _hrule(0.8, GREEN)).properties(height=H + 20)


def fwer_chart(df: pd.DataFrame) -> alt.Chart:
    d = df.rename(columns={"P(>=1 false win), no correction": "Measured, no correction", "theory 1-0.95^m": "Theory, no correction", "with Holm": "With Holm correction"})
    long = d.melt(id_vars="n_null_experiments", value_vars=["Measured, no correction", "Theory, no correction", "With Holm correction"], var_name="series", value_name="p")
    scale = alt.Scale(domain=["Measured, no correction", "Theory, no correction", "With Holm correction"], range=[RED, "#e9a3a3", GREEN])
    ch = alt.Chart(long).mark_line(point=True).encode(
        x=alt.X("n_null_experiments:Q", title="Number of experiments with NO real effect"),
        y=alt.Y("p:Q", title="Chance of at least one false 'win'", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("series:N", scale=scale, title=None, legend=alt.Legend(orient="bottom")), tooltip=["series", "n_null_experiments", alt.Tooltip("p:Q", format=".1%")])
    return (ch + _hrule(0.05, GREEN)).properties(height=H)


def weekly_chart(df: pd.DataFrame) -> alt.Chart:
    d = df.copy(); d["lo"], d["hi"] = d.rel_lift - d.ci_half, d.rel_lift + d.ci_half
    x = alt.X("exposure_week:O", title="Weeks since each user first saw the change", axis=alt.Axis(labelAngle=0))
    err = alt.Chart(d).mark_rule(strokeWidth=3, color=BLUE).encode(x=x, y=alt.Y("lo:Q", title="Relative lift in orders (95% interval)", axis=alt.Axis(format="%")), y2="hi:Q")
    pts = alt.Chart(d).mark_point(filled=True, size=130, color=BLUE).encode(x=x, y="rel_lift:Q", tooltip=["exposure_week", alt.Tooltip("rel_lift:Q", format="+.1%"), alt.Tooltip("ci_half:Q", format=".1%")])
    return (_hrule(0, "#999", (0, 0)) + err + pts).properties(height=H)


def interaction_chart(row: pd.Series) -> alt.Chart:
    d = pd.DataFrame({"when": ["exp_004 switched ON", "exp_004 switched OFF"], "effect": [row.effect_A_when_B_on, row.effect_A_when_B_off]})
    return alt.Chart(d).mark_bar(size=60, color=BLUE).encode(x=alt.X("when:N", title=None, sort=None), y=alt.Y("effect:Q", title="Extra orders per user from exp_001"),
                                                            tooltip=["when", alt.Tooltip("effect:Q", format="+.4f")]).properties(height=H - 60)


def estimators_chart(df: pd.DataFrame, truth: float | None = None) -> alt.Chart:
    err = alt.Chart(df).mark_rule(strokeWidth=3, color=BLUE).encode(y=alt.Y("estimator:N", title=None, sort=None), x=alt.X("ci_low:Q", title="Relative lift in revenue per user (95% interval)", axis=alt.Axis(format="%")), x2="ci_high:Q")
    pts = alt.Chart(df).mark_point(filled=True, size=120, color=BLUE).encode(y=alt.Y("estimator:N", sort=None), x="lift:Q", tooltip=["estimator", alt.Tooltip("lift:Q", format="+.1%"), alt.Tooltip("ci_width:Q", format=".1%")])
    layers = [err, pts]
    if truth is not None:
        layers.append(_rule(truth, GREEN))
    return alt.layer(*layers).properties(height=44 * len(df) + 50)


# ------------------------------------------------------------------------------------------------ uplift
def uplift_models_chart(df: pd.DataFrame) -> alt.Chart:
    d = df.rename(columns={"qini_orders_per_1000_users": "qini"}).copy()
    d["kind"] = np.where(d.model.isin(["S-learner", "T-learner", "X-learner"]), "ML model", "baseline")
    d = d[~d.model.str.startswith("ref:")].sort_values("qini", ascending=False)
    y = alt.Y("model:N", sort=list(d.model), title=None)
    col = alt.Color("kind:N", scale=alt.Scale(domain=["ML model", "baseline"], range=[BLUE, GREY]), legend=alt.Legend(title=None, orient="bottom"))
    err = alt.Chart(d).mark_rule(strokeWidth=3).encode(y=y, x=alt.X("ci_low:Q", title="Extra orders per 1,000 users vs random targeting (95% interval)"), x2="ci_high:Q", color=col)
    pts = alt.Chart(d).mark_point(filled=True, size=130).encode(y=y, x="qini:Q", color=col, tooltip=["model", alt.Tooltip("qini:Q", format=".1f"), alt.Tooltip("ci_low:Q", format=".1f"), alt.Tooltip("ci_high:Q", format=".1f")])
    return (_rule(0, "#999", (0, 0)) + err + pts).properties(height=46 * len(d) + 140)


def decile_chart(df: pd.DataFrame) -> alt.Chart:
    d = df.rename(columns={"decile(0=lowest predicted)": "decile"}).copy(); d["decile"] = d.decile + 1
    d["effect"] = np.where(d.measured_uplift >= 0, "offer helps (extra orders)", "offer hurts (fewer orders)")
    bars = alt.Chart(d).mark_bar().encode(
        x=alt.X("decile:O", title="Users ranked by the model, 1 = least likely to respond, 10 = most likely"),
        y=alt.Y("measured_uplift:Q", title="Measured extra orders per user"),
        color=alt.Color("effect:N", scale=alt.Scale(domain=["offer helps (extra orders)", "offer hurts (fewer orders)"], range=[GREEN, RED]), legend=alt.Legend(title=None, orient="bottom")),
        tooltip=["decile", alt.Tooltip("measured_uplift:Q", format="+.3f"), alt.Tooltip("mean_predicted_uplift:Q", format="+.3f")])
    pred = alt.Chart(d).mark_line(point=alt.OverlayMarkDef(color="white", size=60), color="white", strokeDash=[4, 3]).encode(x="decile:O", y="mean_predicted_uplift:Q")
    return (bars + pred + _hrule(0, "#999", (0, 0))).properties(height=H)


def profit_curve(curve: pd.DataFrame, value: float, cost: float, n_users: int, base_value: float = 50.0, base_cost: float = 3.0):
    """Recompute the profit curve for any unit economics from the one saved at (base_value, base_cost)."""
    d = curve.copy()
    gain_orders = (d.profit_inr + base_cost * d.fraction_treated * n_users) / base_value        # cumulative incremental orders
    d["profit"] = value * gain_orders - cost * d.fraction_treated * n_users
    return d


def profit_chart(d: pd.DataFrame) -> alt.Chart:
    best = d.loc[d.profit.idxmax()]
    line = alt.Chart(d).mark_line(color=BLUE, strokeWidth=3).encode(
        x=alt.X("fraction_treated:Q", title="Share of users who get the offer (best-scored first)", axis=alt.Axis(format="%")),
        y=alt.Y("profit:Q", title="Profit (INR, whole user base)", axis=alt.Axis(format="~s")), tooltip=[alt.Tooltip("fraction_treated:Q", format=".0%"), alt.Tooltip("profit:Q", format=",.0f")])
    mk = alt.Chart(pd.DataFrame([best])).mark_point(filled=True, size=160, color=GREEN).encode(x="fraction_treated:Q", y="profit:Q")
    layers = [_hrule(0, "#999", (0, 0)), line, mk]
    if d.fraction_treated.iloc[-1] >= 0.999:                       # older artifacts stop at 99.8%: do not mislabel their last point
        layers.append(alt.Chart(pd.DataFrame([d.iloc[-1]])).mark_point(filled=True, size=160, color=RED, shape="diamond").encode(x="fraction_treated:Q", y="profit:Q"))
    return alt.layer(*layers).properties(height=H)


def seeds_chart(runs: pd.DataFrame) -> alt.Chart:
    d = runs[runs.model != "random"]
    pts = alt.Chart(d).mark_point(filled=True, size=70, opacity=0.7).encode(
        y=alt.Y("model:N", title=None), x=alt.X("qini:Q", title="Extra orders per 1,000 users vs random, one dot per seed"), color=alt.Color("model:N", legend=None), tooltip=["seed", "model", alt.Tooltip("qini:Q", format=".1f")])
    mean = alt.Chart(d).mark_tick(color="white", thickness=3, size=28).encode(y="model:N", x="mean(qini):Q")
    return (_rule(0, "#999", (0, 0)) + pts + mean).properties(height=150)


def holdout_chart(summ: pd.DataFrame) -> alt.Chart:
    d = summ[summ.model != "random"].melt(id_vars="model", value_vars=["profit_in_sample_per_1000", "profit_holdout_per_1000"], var_name="how", value_name="profit")
    d["how"] = d["how"].map({"profit_in_sample_per_1000": "same users (optimistic)", "profit_holdout_per_1000": "held-out half (honest)"})
    return alt.Chart(d).mark_bar().encode(x=alt.X("model:N", title=None, axis=alt.Axis(labelAngle=0)), y=alt.Y("profit:Q", title="Profit per 1,000 users (INR, assumed economics)"),
                                          color=alt.Color("how:N", title=None, scale=alt.Scale(domain=["same users (optimistic)", "held-out half (honest)"], range=[GREY, BLUE]), legend=alt.Legend(orient="bottom")), xOffset="how:N",
                                          tooltip=["model", "how", alt.Tooltip("profit:Q", format=",.0f")]).properties(height=H)


def paired_chart(pb: pd.DataFrame) -> alt.Chart:
    err = alt.Chart(pb).mark_rule(strokeWidth=3, color=BLUE).encode(y=alt.Y("difference:N", title=None, sort=None), x=alt.X("ci_low:Q", title="Difference in extra orders per 1,000 users (95% interval)"), x2="ci_high:Q")
    pts = alt.Chart(pb).mark_point(filled=True, size=130, color=BLUE).encode(y=alt.Y("difference:N", sort=None), x="mean:Q", tooltip=["difference", alt.Tooltip("mean:Q", format="+.1f")])
    return (_rule(0, "#999", (0, 0)) + err + pts).properties(height=44 * len(pb) + 50)


# ------------------------------------------------------------------------------------------------ real data
def balance_chart(df: pd.DataFrame) -> alt.Chart:
    d = df.rename(columns={"Unnamed: 0": "feature"}).copy(); d["abs_smd"] = d.smd.abs()
    return alt.Chart(d).mark_bar(color=BLUE).encode(y=alt.Y("feature:N", sort="-x", title=None), x=alt.X("abs_smd:Q", title="Difference between arms (standardised); 0 = perfectly balanced"),
                                                    tooltip=["feature", alt.Tooltip("smd:Q", format="+.4f"), alt.Tooltip("p_value:Q", format=".2g")]).properties(height=28 * len(d) + 40)


def criteo_effects_chart(df: pd.DataFrame, outcome: str) -> alt.Chart:
    d = df[df.outcome == outcome].copy()
    d["effect_pp"], d["half_pp"] = d.effect * 100, d.ci_half * 100
    d["lo"], d["hi"] = d.effect_pp - d.half_pp, d.effect_pp + d.half_pp
    y = alt.Y("method:N", sort=None, title=None)
    err = alt.Chart(d).mark_rule(strokeWidth=3).encode(y=y, x=alt.X("lo:Q", title=f"Effect of the ads on '{outcome}' (percentage points, 95% interval)"), x2="hi:Q", color=alt.Color("method:N", legend=None))
    pts = alt.Chart(d).mark_point(filled=True, size=140).encode(y=y, x="effect_pp:Q", color=alt.Color("method:N", legend=None), tooltip=["method", alt.Tooltip("effect_pp:Q", format=".3f")])
    return (_rule(0, "#999", (0, 0)) + err + pts).properties(height=44 * len(d) + 50)


def criteo_deciles_chart(df: pd.DataFrame) -> alt.Chart:
    d = df.rename(columns={"decile(0=lowest predicted)": "decile"}).copy(); d["decile"] = d.decile + 1
    long = d.melt(id_vars="decile", value_vars=["measured_AIPW", "measured_constant_propensity"], var_name="how", value_name="uplift")
    long["how"] = long.how.map({"measured_AIPW": "Robust measurement (doubly-robust)", "measured_constant_propensity": "First-pass measurement (overstates the top)"})
    bars = alt.Chart(long).mark_bar().encode(x=alt.X("decile:O", title="Users ranked by the model, 10 = most likely to respond"), y=alt.Y("uplift:Q", title="Extra visits per user"),
                                             color=alt.Color("how:N", scale=alt.Scale(range=[BLUE, GREY]), title=None, legend=alt.Legend(orient="bottom")), xOffset="how:N",
                                             tooltip=["decile", "how", alt.Tooltip("uplift:Q", format="+.4f")])
    return bars.properties(height=H)


# ------------------------------------------------------------------------------------------------ platform validation
def coverage_chart(s: pd.DataFrame) -> alt.Chart:
    d = s.copy(); d["row"] = d.experiment + " (" + d.estimator + ")"
    d["role"] = np.where(d.experiment == "exp_006", "planted bug (should look broken)", "clean experiment")
    y = alt.Y("row:N", sort=None, title=None)
    col = alt.Color("role:N", scale=alt.Scale(domain=["clean experiment", "planted bug (should look broken)"], range=[BLUE, RED]), legend=alt.Legend(title=None, orient="bottom"))
    err = alt.Chart(d).mark_rule(strokeWidth=3).encode(y=y, x=alt.X("coverage_low:Q", title="How often the 95% interval contained the truth (with its own uncertainty)", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])), x2="coverage_high:Q", color=col)
    pts = alt.Chart(d).mark_point(filled=True, size=110).encode(y=y, x="coverage:Q", color=col, tooltip=["row", alt.Tooltip("coverage:Q", format=".0%")])
    return (_rule(0.95, GREEN) + err + pts).properties(height=30 * len(d) + 50)


def bias_chart(s: pd.DataFrame) -> alt.Chart:
    d = s.copy(); d["row"] = d.experiment + " (" + d.estimator + ")"; d["lo"], d["hi"] = d.bias - 2 * d.bias_se, d.bias + 2 * d.bias_se
    d["role"] = np.where(d.experiment == "exp_006", "planted bug (should look broken)", "clean experiment")
    y = alt.Y("row:N", sort=None, title=None)
    col = alt.Color("role:N", scale=alt.Scale(domain=["clean experiment", "planted bug (should look broken)"], range=[BLUE, RED]), legend=alt.Legend(title=None, orient="bottom"))
    err = alt.Chart(d).mark_rule(strokeWidth=3).encode(y=y, x=alt.X("lo:Q", title="Average error of the estimate (estimate minus truth), with +/- 2 standard errors", axis=alt.Axis(format="%")), x2="hi:Q", color=col)
    pts = alt.Chart(d).mark_point(filled=True, size=110).encode(y=y, x="bias:Q", color=col, tooltip=["row", alt.Tooltip("bias:Q", format="+.2%")])
    return (_rule(0, GREEN, (0, 0)) + err + pts).properties(height=30 * len(d) + 50)


def estimates_hist(runs: pd.DataFrame, experiment: str) -> alt.Chart:
    d = runs[runs.experiment == experiment].copy(); truth = float(d.truth.iloc[0])
    d["estimator"] = d.estimator.map({"naive": "Naive", "cuped": "CUPED"})
    h = alt.Chart(d).mark_bar(opacity=0.6).encode(x=alt.X("lift:Q", bin=alt.Bin(maxbins=24), title="Estimated lift in one simulated world", axis=alt.Axis(format="%")),
                                                  y=alt.Y("count():Q", title="Number of simulated worlds", stack=None),
                                                  color=alt.Color("estimator:N", scale=alt.Scale(domain=["Naive", "CUPED"], range=[GREY, BLUE]), legend=alt.Legend(title=None, orient="bottom")))
    return (h + _rule(truth, GREEN, (0, 0))).properties(height=H)


# ------------------------------------------------------------------------------------------------ SQL results
def funnel_chart(df: pd.DataFrame) -> alt.Chart:
    long = df.melt(id_vars="device", value_vars=["cart_rate", "cart_to_checkout", "checkout_to_purchase", "session_conversion"], var_name="step", value_name="rate")
    long["step"] = long.step.map({"cart_rate": "session -> cart", "cart_to_checkout": "cart -> checkout", "checkout_to_purchase": "checkout -> purchase", "session_conversion": "session -> purchase (overall)"})
    return alt.Chart(long).mark_bar().encode(x=alt.X("step:N", title=None, sort=None, axis=alt.Axis(labelAngle=-15)), y=alt.Y("rate:Q", title="Conversion", axis=alt.Axis(format="%")),
                                             color=alt.Color("device:N", title="Device"), xOffset="device:N", tooltip=["device", "step", alt.Tooltip("rate:Q", format=".1%")]).properties(height=H)


def retention_chart(df: pd.DataFrame) -> alt.Chart:
    agg = df.groupby("weeks_since_signup").agg(active=("active_users", "sum"), size=("cohort_size", "sum")).reset_index()
    agg["retention"] = agg.active / agg["size"]
    return alt.Chart(agg).mark_line(point=True, color=BLUE, strokeWidth=3).encode(x=alt.X("weeks_since_signup:Q", title="Weeks since signup week", axis=alt.Axis(tickMinStep=1)), y=alt.Y("retention:Q", title="Share of new users still active", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
                                                                                 tooltip=["weeks_since_signup", alt.Tooltip("retention:Q", format=".1%")]).properties(height=H)


def sessions_chart(df: pd.DataFrame) -> alt.Chart:
    long = df.melt(id_vars="device", value_vars=["logged_sessions", "rebuilt_sessions"], var_name="source", value_name="sessions")
    long["source"] = long.source.map({"logged_sessions": "logged by the app", "rebuilt_sessions": "rebuilt from raw events with LAG()"})
    return alt.Chart(long).mark_bar().encode(x=alt.X("device:N", title=None, axis=alt.Axis(labelAngle=0)), y=alt.Y("sessions:Q", title="Sessions"), color=alt.Color("source:N", title=None, legend=alt.Legend(orient="bottom")),
                                             xOffset="source:N", tooltip=["device", "source", alt.Tooltip("sessions:Q", format=",")]).properties(height=H)


def revenue_trend_chart(df: pd.DataFrame) -> alt.Chart:
    d = drop_partial_last_day(df, "revenue").copy(); d["day"] = pd.to_datetime(d.day)
    long = d.melt(id_vars="day", value_vars=["revenue", "revenue_7d_avg"], var_name="series", value_name="inr")
    long["series"] = long.series.map({"revenue": "daily revenue", "revenue_7d_avg": "7-day moving average"})
    dom = ["daily revenue", "7-day moving average"]
    return alt.Chart(long).mark_line().encode(x=alt.X("day:T", title=None), y=alt.Y("inr:Q", title="Revenue (INR)", axis=alt.Axis(format="~s")), color=alt.Color("series:N", title=None, scale=alt.Scale(domain=dom, range=[GREY, BLUE]), legend=alt.Legend(orient="bottom")),
                                              strokeWidth=alt.StrokeWidth("series:N", legend=None, scale=alt.Scale(domain=dom, range=[1, 3.5]))).properties(height=H)


def rfm_chart(df: pd.DataFrame) -> alt.Chart:
    y = alt.Y("segment:N", sort=list(df.sort_values("share_of_revenue", ascending=False).segment), title=None)
    bars = alt.Chart(df).mark_bar(color=BLUE).encode(y=y, x=alt.X("share_of_revenue:Q", title="Share of total revenue", axis=alt.Axis(format="%")),
                                                     tooltip=["segment", alt.Tooltip("users:Q", format=","), alt.Tooltip("share_of_revenue:Q", format=".1%"), alt.Tooltip("avg_orders:Q", format=".2f")])
    txt = alt.Chart(df).mark_text(align="left", dx=4, color="#e6e6e6").encode(y=y, x="share_of_revenue:Q", text=alt.Text("share_of_revenue:Q", format=".0%"))
    return (bars + txt).properties(height=34 * len(df) + 40)


def streaks_chart(df: pd.DataFrame) -> alt.Chart:
    df = df[df.longest_streak <= 20]                                  # longer streaks are very rare
    return alt.Chart(df).mark_bar(color=BLUE).encode(x=alt.X("longest_streak:O", title="Longest run of consecutive active days"), y=alt.Y("users:Q", title="Users"), tooltip=["longest_streak", alt.Tooltip("users:Q", format=",")]).properties(height=H)


def dq_chart(df: pd.DataFrame) -> alt.Chart:
    d = drop_partial_last_day(df, "raw_events").copy(); d["day"] = pd.to_datetime(d.day)
    long = d.melt(id_vars="day", value_vars=["duplicate_rate", "late_rate"], var_name="problem", value_name="rate")
    long["problem"] = long.problem.map({"duplicate_rate": "duplicate events removed", "late_rate": "events arriving > 1 day late"})
    return alt.Chart(long).mark_line().encode(x=alt.X("day:T", title=None), y=alt.Y("rate:Q", title="Share of events", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 0.04])),
                                              color=alt.Color("problem:N", title=None, scale=alt.Scale(range=[AMBER, RED]), legend=alt.Legend(orient="bottom")), tooltip=["day:T", "problem", alt.Tooltip("rate:Q", format=".2%")]).properties(height=H)


# ------------------------------------------------------------------------------------------ live platform pages
def arms_forest_chart(arms: pd.DataFrame, bar: float = 0.03) -> alt.Chart:
    """One interval per treatment arm vs control, coloured by the platform's decision."""
    d = arms.copy()
    present = [k for k in DECISION_COLOUR if k in set(d.decision)]
    colour = alt.Color("decision:N", scale=alt.Scale(domain=present, range=[DECISION_COLOUR[k] for k in present]),
                       legend=alt.Legend(title="Decision", orient="bottom"))
    y = alt.Y("arm:N", sort=list(d.arm), title=None)
    tip = ["arm", "decision", alt.Tooltip("lift:Q", format="+.1%"), alt.Tooltip("ci_low:Q", format="+.1%"), alt.Tooltip("ci_high:Q", format="+.1%")]
    rng = alt.Chart(d).mark_rule(strokeWidth=3).encode(x=alt.X("ci_low:Q", title="Relative lift vs control (95% interval)", axis=alt.Axis(format="%")),
                                                       x2="ci_high:Q", y=y, color=colour, tooltip=tip)
    pts = alt.Chart(d).mark_point(filled=True, size=120).encode(x="lift:Q", y=y, color=colour, tooltip=tip)
    return (_rule(0, "#999", (0, 0)) + _rule(bar, GREEN, (5, 4)) + rng + pts).properties(height=60 * len(d) + 150)   # Streamlit full-width mode: height is the TOTAL (plot + axis + legend)


def live_power_chart(df: pd.DataFrame, target_lift: float | None = None) -> alt.Chart:
    line = alt.Chart(df).mark_line(color=BLUE, strokeWidth=3).encode(
        x=alt.X("true_lift:Q", title="True relative lift", axis=alt.Axis(format="%")),
        y=alt.Y("power:Q", title="Chance of detecting it (power)", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        tooltip=[alt.Tooltip("true_lift:Q", format=".1%"), alt.Tooltip("power:Q", format=".0%")])
    layers = [_hrule(0.8, GREEN), line]
    if target_lift is not None:
        layers.append(_rule(target_lift, AMBER, (5, 4)))
    return alt.layer(*layers).properties(height=H)


def days_chart(df: pd.DataFrame) -> alt.Chart:
    line = alt.Chart(df).mark_line(color=PURPLE, strokeWidth=3, point=True).encode(
        x=alt.X("detectable_lift:Q", title="Smallest relative lift you want to detect", axis=alt.Axis(format="%")),
        y=alt.Y("days:Q", title="Days of traffic needed", scale=alt.Scale(type="log")),
        tooltip=[alt.Tooltip("detectable_lift:Q", format=".1%"), "days", alt.Tooltip("n_per_arm:Q", format=",")])
    return line.properties(height=H)


def live_profit_chart(d: pd.DataFrame, currency: str = "") -> alt.Chart:
    """Profit vs share treated; green dot = best reach, red diamond = treat everyone. Columns: fraction_treated, profit."""
    best = d.loc[d.profit.idxmax()]
    line = alt.Chart(d).mark_line(color=BLUE, strokeWidth=3).encode(
        x=alt.X("fraction_treated:Q", title="Share of users treated (best-scored first)", axis=alt.Axis(format="%")),
        y=alt.Y("profit:Q", title=f"Estimated profit ({currency or 'units'}, this sample)", axis=alt.Axis(format="~s")),
        tooltip=[alt.Tooltip("fraction_treated:Q", format=".0%"), alt.Tooltip("profit:Q", format=",.0f")])
    mk = alt.Chart(pd.DataFrame([best])).mark_point(filled=True, size=160, color=GREEN).encode(x="fraction_treated:Q", y="profit:Q")
    end = alt.Chart(pd.DataFrame([d.iloc[-1]])).mark_point(filled=True, size=160, color=RED, shape="diamond").encode(x="fraction_treated:Q", y="profit:Q")
    return alt.layer(_hrule(0, "#999", (0, 0)), line, mk, end).properties(height=H)


def qini_live_chart(df: pd.DataFrame) -> alt.Chart:
    """Qini area per model with bootstrap interval. Columns: model, qini_per_1000_users, ci_low, ci_high."""
    d = df.copy()
    d["kind"] = np.where(d.model.str.endswith("-learner"), "ML model", "baseline")
    y = alt.Y("model:N", sort=list(d.model), title=None)
    col = alt.Color("kind:N", scale=alt.Scale(domain=["ML model", "baseline"], range=[BLUE, GREY]), legend=alt.Legend(title=None, orient="bottom"))
    err = alt.Chart(d).mark_rule(strokeWidth=3).encode(y=y, x=alt.X("ci_low:Q", title="Extra outcome per 1,000 users vs the average user (95% interval)"), x2="ci_high:Q", color=col)
    pts = alt.Chart(d).mark_point(filled=True, size=130).encode(y=y, x="qini_per_1000_users:Q", color=col,
                                                                 tooltip=["model", alt.Tooltip("qini_per_1000_users:Q", format=".1f")])
    return (_rule(0, "#999", (0, 0)) + err + pts).properties(height=46 * len(d) + 140)
