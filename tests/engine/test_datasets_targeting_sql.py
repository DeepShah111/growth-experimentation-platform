import numpy as np
import pandas as pd
import pytest

from src.engine.analyze import AnalysisConfig, analyze_experiment
from src.engine.datasets import (CRITEO, build_demo_warehouse, experiment_in_sql, guess_columns, load_criteo, load_hillstrom,
                                 load_simulated, read_upload, simulated_spec)
from src.engine.targeting import feature_matrix, policy_summary, profit_curve, qini_table, score_uplift, thin


@pytest.fixture(scope="module")
def wh():
    return build_demo_warehouse(n_users=8_000, seed=11, return_truth=True)


# ---------------------------------------------------------------------------------------------- datasets
def test_demo_warehouse_has_the_tables_the_sql_needs_and_hides_the_answer_key(wh):
    con, truth = wh
    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    assert {"users", "sessions", "events", "orders", "experiments", "assignments", "products"} <= tables
    assert not any("answer" in t or "truth" in t for t in tables) and len(truth) == 9
    assert con.execute("SELECT COUNT(*) FROM stg_events_clean").fetchone()[0] > 0


def test_simulated_frame_feeds_the_engine_and_the_srm_bug_is_caught(wh):
    con, _ = wh
    d = load_simulated(con, "exp_006")
    r = analyze_experiment(d, "variant", "orders_in", ["orders_pre"], AnalysisConfig(control="control"))
    assert not r["srm_ok"] and r["arms"].decision.iloc[0] == "DO NOT TRUST"
    ok = analyze_experiment(load_simulated(con, "exp_003"), "variant", "orders_in", ["orders_pre"], AnalysisConfig(control="control"))
    assert ok["srm_ok"]


def test_sql_experiment_analysis_agrees_with_the_python_engine_to_nine_digits(wh):
    con, _ = wh
    for exp in ("exp_001", "exp_003", "exp_004"):
        s = con.execute(experiment_in_sql(), {"exp_id": exp}).df().iloc[0]
        py = analyze_experiment(load_simulated(con, exp), "variant", "orders_in", ["orders_pre"], AnalysisConfig(control="control"))["arms"].iloc[0]
        assert s.cuped_lift == pytest.approx(py.adj_lift, rel=1e-9)
        assert s.cuped_half_width == pytest.approx(py.adj_half, rel=1e-9)
        assert s.naive_lift == pytest.approx(py.naive_lift, rel=1e-9)
        assert s.n_treat == py.n_treat and s.n_ctrl == py.n_ctrl


def test_simulated_spec_puts_the_primary_metric_first():
    assert list(simulated_spec("exp_007", "revenue_per_user").outcomes)[0] == "revenue_in"
    assert list(simulated_spec("exp_001", "orders").outcomes)[0] == "orders_in"


def test_upload_reader_enforces_limits_and_guesses_columns():
    csv = "group,converted,visits_pre\ncontrol,0,1\ntreatment,1,2\ncontrol,1,0\ntreatment,0,3\n"
    df = read_upload(csv)
    g = guess_columns(df)
    assert g["variant"] == "group" and g["control"] == "control" and g["outcome"] == "converted"
    with pytest.raises(ValueError, match="more than 2 rows"):
        read_upload(csv, max_rows=2)
    with pytest.raises(ValueError, match="at least two columns"):
        read_upload("a\n1\n2\n")


def test_criteo_loader_maps_treatment_to_arm(tmp_path, make_fake_criteo):
    p = tmp_path / "c.parquet"
    make_fake_criteo(5_000).to_parquet(p)
    d = load_criteo(p, max_rows=None)
    assert set(d.arm) == {"control", "treatment"} and CRITEO.control == "control"
    r = analyze_experiment(d, "arm", "visit", list(CRITEO.covariates), AnalysisConfig(control="control", planned_split={"control": 0.15, "treatment": 0.85}))
    assert r["srm_ok"]


def test_missing_hillstrom_file_gives_an_actionable_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="sample_data"):
        load_hillstrom(tmp_path / "nope.csv")


# --------------------------------------------------------------------------------------------- targeting
def _synthetic_arm_table(n=6_000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, n)                                           # x=1 responds (+1 order), x=0 does not
    noise = rng.random(n)
    arm = np.where(rng.random(n) < 0.5, "t", "c")
    y = rng.poisson(1.0 + (arm == "t") * x * 1.0).astype(float)
    return pd.DataFrame({"arm": arm, "x": x.astype(float), "z": noise, "y": y})


def test_feature_matrix_codes_categoricals_and_flags_them():
    d = pd.DataFrame({"a": [1.0, np.nan, 3.0], "city": ["x", "y", "x"]})
    X, mask = feature_matrix(d, ["a", "city"], ["city"])
    assert mask == [False, True] and X[1, 0] == 2.0 and list(X[:, 1]) == [0.0, 1.0, 0.0]      # NaN -> median, categories -> codes


def test_uplift_models_find_the_responders_and_the_table_says_so():
    d = _synthetic_arm_table()
    res = score_uplift(d, "arm", "t", "c", "y", ["x", "z"], kinds="ST", n_splits=3)
    q = qini_table(res, reps=40).set_index("model")
    assert q.loc["T-learner", "beats_random"] and q.loc["T-learner", "qini_per_1000_users"] > 50
    assert not q.loc["random", "beats_random"]


def test_profit_curve_arithmetic_and_thinning_keeps_peak_and_end():
    score = np.array([3.0, 2.0, 1.0, 0.0]); tau = np.array([2.0, 1.0, -1.0, -2.0])
    c = profit_curve(score, tau, value=10.0, cost=4.0)                     # gains 2,3,2,0 ; costs 4,8,12,16
    assert list(c.profit) == [20 - 4, 30 - 8, 20 - 12, 0 - 16]
    big = profit_curve(np.arange(5000)[::-1].astype(float), np.r_[np.ones(2500), -np.ones(2500)], 1.0, 0.0)
    t = thin(big, step=500)
    assert len(t) < 20 and t.profit.max() == big.profit.max() and t.fraction_treated.iloc[-1] == 1.0


def test_policy_summary_prefers_targeting_when_treatment_is_costly_and_effect_is_concentrated():
    d = _synthetic_arm_table(n=10_000, seed=2)
    res = score_uplift(d, "arm", "t", "c", "y", ["x", "z"], kinds="T", n_splits=3)
    s = policy_summary(res, "T-learner", value=10.0, cost=6.0)
    assert 0.35 < s["best_share"] < 0.65                                   # about the responding half
    assert s["holdout"]["profit_holdout"] > s["holdout"]["treat_everyone_holdout"]


def test_negative_outcomes_and_single_arm_inputs_are_rejected():
    d = _synthetic_arm_table(n=500)
    with pytest.raises(ValueError, match="non-negative"):
        score_uplift(d.assign(y=-d.y - 1), "arm", "t", "c", "y", ["x"])
    with pytest.raises(ValueError, match="both"):
        score_uplift(d[d.arm == "t"], "arm", "t", "c", "y", ["x"])
