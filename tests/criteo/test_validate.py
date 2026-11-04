"""Run the Criteo validation on a SYNTHETIC Criteo-shaped file (no download needed)."""

from src.criteo.validate import balance_table, diff_in_rates, lin_adjusted_effect, run


def test_balance_is_fine_when_randomised(make_fake_criteo):
    df = make_fake_criteo()
    b = balance_table(df[[f"f{i}" for i in range(12)]], df.treatment.to_numpy())
    assert b.smd.abs().max() < 0.05


def test_effect_estimators_agree_and_adjustment_does_not_widen_ci(make_fake_criteo):
    df = make_fake_criteo()
    t, y = df.treatment.to_numpy().astype(int), df.visit.to_numpy().astype(float)
    a = diff_in_rates(y, t)
    b = lin_adjusted_effect(y, t, df[[f"f{i}" for i in range(12)]].to_numpy())
    assert abs(a["abs_effect"] - b["abs_effect"]) < 0.01
    assert b["abs_ci_half"] <= a["abs_ci_half"] * 1.05


def test_pipeline_runs_and_finds_the_responding_group(tmp_path, make_fake_criteo):
    p = tmp_path / "fake.parquet"
    make_fake_criteo(60_000).to_parquet(p)
    res = run(str(p), reps=20, out_dir=tmp_path)
    rep = res["uplift"]["visit"].set_index("model")
    assert rep.loc["S-learner", "qini_extra_per_1000_users"] > rep.loc["random", "qini_extra_per_1000_users"]
    assert rep.loc["S-learner", "diff_vs_random_low"] > 0
