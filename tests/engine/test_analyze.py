"""The generic analysis engine, checked against worked examples and planted truths."""
import numpy as np
import pandas as pd
import pytest

from src.engine.analyze import AnalysisConfig, AnalysisError, analyze_experiment, srm_test, to_jsonable
from src.experiments.stats_utils import cuped_adjust, regression_adjust


def make(n=20_000, lifts=(0.10, 0.0), seed=0, sizes=None, with_cov=True):
    """Arms: control + one per lift. Outcome = count with mean 1.0 * (1 + lift); a pre-period covariate explains most variance."""
    rng = np.random.default_rng(seed)
    arms = ["control"] + [f"t{i}" for i in range(len(lifts))]
    sizes = sizes or [n] * len(arms)
    rows = []
    for arm, size, lift in zip(arms, sizes, (0.0, *lifts)):
        level = rng.gamma(2.0, 0.5, size)
        pre = rng.poisson(level * 2.0)
        y = rng.poisson(level * (1 + lift) * 2.0)
        rows.append(pd.DataFrame({"arm": arm, "y": y.astype(float), "pre": pre.astype(float) if with_cov else 0.0}))
    return pd.concat(rows, ignore_index=True)


def test_known_lift_is_recovered_and_interval_covers_it():
    r = analyze_experiment(make(), "arm", "y", ["pre"], AnalysisConfig(control="control"))["arms"].set_index("arm")
    assert r.loc["t0", "ci_low"] < 0.10 < r.loc["t0", "ci_high"]
    assert r.loc["t0", "decision"] == "SHIP" and r.loc["t1", "decision"] in {"NO EFFECT FOUND", "INCONCLUSIVE (underpowered)"}


def test_covariate_adjustment_narrows_the_interval_and_keeps_the_lift():
    df = make()
    a = analyze_experiment(df, "arm", "y", [], AnalysisConfig())["arms"].set_index("arm")
    b = analyze_experiment(df, "arm", "y", ["pre"], AnalysisConfig())["arms"].set_index("arm")
    assert b.loc["t0", "ci_width_ratio"] < 0.9
    assert abs(a.loc["t0", "naive_lift"] - b.loc["t0", "naive_lift"]) < 1e-12      # naive is untouched
    assert abs(b.loc["t0", "adj_lift"] - 0.10) < 0.03


def test_regression_adjust_matches_cuped_for_one_covariate_and_keeps_the_mean():
    rng = np.random.default_rng(1)
    x = rng.normal(size=4000); y = 3 * x + rng.normal(size=4000)
    adj, r2 = regression_adjust(y, x)
    assert np.allclose(adj, cuped_adjust(y, x)) and abs(adj.mean() - y.mean()) < 1e-10
    assert adj.var() < 0.2 * y.var() and 0.8 < r2 < 0.95
    adj2, _ = regression_adjust(y, np.c_[x, np.ones_like(x)])                     # a constant column is ignored, not an error
    assert np.allclose(adj, adj2)


def test_cuped_keeps_the_mean_and_cuts_the_variance_on_the_experiment_data():
    d = make()
    y, x = d.y.to_numpy(), d["pre"].to_numpy()
    ya = cuped_adjust(y, x)
    assert abs(ya.mean() - y.mean()) < 1e-9 and ya.var() < 0.85 * y.var()      # Poisson noise caps the gain here


def test_sample_ratio_mismatch_blocks_everything_even_a_huge_lift():
    df = make(lifts=(0.5,), sizes=[20_000, 15_000])
    r = analyze_experiment(df, "arm", "y", ["pre"], AnalysisConfig())
    assert not r["srm_ok"] and r["arms"].decision.iloc[0] == "DO NOT TRUST" and np.isnan(r["arms"].p_adj.iloc[0])


def test_planned_unequal_split_is_respected():
    df = make(lifts=(0.0,), sizes=[30_000, 10_000])                                # a true 75/25 split
    assert not analyze_experiment(df, "arm", "y", [], AnalysisConfig())["srm_ok"]
    ok = analyze_experiment(df, "arm", "y", [], AnalysisConfig(planned_split={"control": 0.75, "t0": 0.25}))
    assert ok["srm_ok"]


def test_srm_test_matches_a_hand_calculation():
    out = srm_test({"a": 5200, "b": 4800})                                          # chi2 = 2*(200^2/5000) = 16 -> p = 6.3e-5
    assert abs(out["p_value"] - 6.334e-5) < 2e-7


def test_holm_is_applied_across_the_treatment_arms():
    from src.experiments.stats_utils import holm_adjust
    df = make(lifts=(0.04, 0.04, 0.04, 0.04), n=15_000, seed=3)
    r = analyze_experiment(df, "arm", "y", ["pre"], AnalysisConfig())["arms"]
    assert np.allclose(r.p_adj.to_numpy(), holm_adjust(r.p_value.to_numpy()))
    assert (r.p_adj >= r.p_value - 1e-15).all() and (r.p_adj <= 1).all() and (r.p_adj > r.p_value).any()


def test_underpowered_non_significant_result_is_inconclusive_not_no_effect():
    df = make(n=300, lifts=(0.02,), seed=5)
    r = analyze_experiment(df, "arm", "y", ["pre"], AnalysisConfig(min_effect=0.02))["arms"].iloc[0]
    assert r.decision == "INCONCLUSIVE (underpowered)" and r.mde > 0.02


def test_guardrail_breach_blocks_a_win():
    rng = np.random.default_rng(7)
    n = 30_000
    arm = np.repeat(["control", "t"], n)
    y = np.r_[rng.poisson(1.0, n), rng.poisson(1.15, n)].astype(float)
    g = np.r_[rng.poisson(2.0, n), rng.poisson(1.7, n)].astype(float)                # the guardrail got 15% worse
    df = pd.DataFrame({"arm": arm, "y": y, "g": g})
    r = analyze_experiment(df, "arm", "y", [], AnalysisConfig(guardrail="g")).get("arms").iloc[0]
    assert r.decision == "ITERATE" and "guardrail" in r.reasons


def test_unprofitable_win_is_iterate_and_profitable_win_ships():
    df = make(lifts=(0.10,), n=30_000)
    cheap = analyze_experiment(df, "arm", "y", ["pre"], AnalysisConfig(value_per_unit=10.0, cost_per_treated=1.0))["arms"].iloc[0]
    dear = analyze_experiment(df, "arm", "y", ["pre"], AnalysisConfig(value_per_unit=10.0, cost_per_treated=50.0))["arms"].iloc[0]
    assert cheap.decision == "SHIP" and cheap.net_value > 0
    assert dear.decision == "ITERATE" and "unprofitable" in dear.reasons and dear.net_value < 0


def test_min_effect_changes_the_note_not_the_evidence():
    df = make(lifts=(0.10,))
    lo = analyze_experiment(df, "arm", "y", ["pre"], AnalysisConfig(min_effect=0.01))["arms"].iloc[0]
    hi = analyze_experiment(df, "arm", "y", ["pre"], AnalysisConfig(min_effect=0.30))["arms"].iloc[0]
    assert lo.lift == hi.lift and "worth-shipping bar" not in lo.reasons and "worth-shipping bar" in hi.reasons


def test_bad_inputs_give_clear_errors():
    df = make(n=100)
    with pytest.raises(AnalysisError, match="control arm 'zzz'"):
        analyze_experiment(df, "arm", "y", [], AnalysisConfig(control="zzz"))
    with pytest.raises(AnalysisError, match="not in the table"):
        analyze_experiment(df, "arm", "nope", [], AnalysisConfig())
    with pytest.raises(AnalysisError, match="numeric"):
        analyze_experiment(df.assign(y=df.y.astype(str)), "arm", "y", [], AnalysisConfig())
    with pytest.raises(AnalysisError, match="at least 30"):
        analyze_experiment(df.groupby("arm").head(10), "arm", "y", [], AnalysisConfig())
    with pytest.raises(AnalysisError, match="outcome cannot also"):
        analyze_experiment(df, "arm", "y", ["y"], AnalysisConfig())


def test_zero_control_mean_is_reported_not_crashed():
    df = make(n=200).assign(y=lambda d: np.where(d.arm == "control", 0.0, d.y))
    r = analyze_experiment(df, "arm", "y", [], AnalysisConfig())["arms"].iloc[0]
    assert r.decision == "INCONCLUSIVE (underpowered)" and "control mean is zero" in r.reasons


def test_missing_values_are_handled_and_reported():
    df = make(n=2000)
    df.loc[:50, "pre"] = np.nan
    df.loc[60:70, "y"] = np.nan
    r = analyze_experiment(df, "arm", "y", ["pre"], AnalysisConfig())
    assert any("dropped" in n for n in r["notes"]) and any("filled" in n for n in r["notes"])


def test_to_jsonable_has_no_nan_and_round_trips():
    import json
    r = analyze_experiment(make(lifts=(0.0,), sizes=[20_000, 12_000]), "arm", "y", [], AnalysisConfig())
    txt = json.dumps(to_jsonable(r))                       # would raise on NaN-free? json allows NaN, so check explicitly
    assert "NaN" not in txt and json.loads(txt)["arms"][0]["decision"] == "DO NOT TRUST"


def test_real_hillstrom_matches_the_published_arm_sizes_and_direction():
    from src.engine.datasets import HILLSTROM, hillstrom_available, load_hillstrom
    if not hillstrom_available():
        pytest.skip("sample_data/hillstrom.csv.gz not present")
    d = load_hillstrom()
    counts = d.segment.value_counts().to_dict()
    assert counts == {"Womens E-Mail": 21387, "Mens E-Mail": 21307, "No E-Mail": 21306}     # published arm sizes
    r = analyze_experiment(d, "segment", "conversion", [], AnalysisConfig(control=HILLSTROM.control))
    arms = r["arms"].set_index("arm")
    assert r["srm_ok"] and (arms.lift > 0.4).all() and (arms.p_adj < 0.001).all()
    assert arms.loc["Mens E-Mail", "lift"] > arms.loc["Womens E-Mail", "lift"]
