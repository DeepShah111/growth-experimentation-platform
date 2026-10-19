import numpy as np
import pandas as pd

from src.experiments.revenue import bootstrap_ratio, compare_estimators, decompose, tail_profile, winsorize


def _frame(n=60_000, lift=0.06, seed=0):
    """Zero-inflated, heavy-tailed revenue (lognormal baskets); treatment multiplies basket value by 1+lift."""
    rng = np.random.default_rng(seed)
    pre_level = rng.gamma(2.0, 0.5, n)
    variant = np.where(rng.random(n) < 0.5, "treatment", "control")
    buys = rng.random(n) < np.clip(0.25 * pre_level, 0, 0.95)
    basket = rng.lognormal(6.0, 1.0, n) * np.where(variant == "treatment", 1 + lift, 1.0)
    rev_in = np.where(buys, basket, 0.0)
    rev_pre = np.where(rng.random(n) < np.clip(0.25 * pre_level, 0, 0.95), rng.lognormal(6.0, 1.0, n), 0.0)
    return pd.DataFrame({"variant": variant, "revenue_in": rev_in, "revenue_pre": rev_pre})


def test_tail_profile_measures_concentration():
    y = np.array([0.0] * 90 + [1.0] * 9 + [100.0])
    p = tail_profile(y)
    assert p["share_zero"] == 0.9 and abs(p["top_1pct_share"] - 100 / 109) < 1e-12 and p["max"] == 100.0


def test_winsorize_caps_values_and_leaves_the_rest():
    y = np.array([1.0, 5.0, 50.0, 500.0])
    assert list(winsorize(y, 50.0)) == [1.0, 5.0, 50.0, 50.0]


def test_all_estimators_recover_a_known_lift_and_cuped_is_not_wider():
    df = _frame()
    est = compare_estimators(df, reps=200).set_index("estimator")
    for name, r in est.iterrows():
        if name.startswith("winsorized"):
            continue                                              # a capped mean is a different quantity
        assert r.ci_low < 0.06 < r.ci_high or abs(r.lift - 0.06) < 0.04, name
    naive = est.loc["naive mean ratio"]
    cuped = est.loc["CUPED (pre-period revenue)"]
    assert cuped.ci_width <= naive.ci_width * 1.02


def test_winsorizing_uses_one_pooled_cap_for_both_arms():
    df = _frame(seed=2)
    est = compare_estimators(df, cap_quantile=0.9, reps=50)
    cap = float(np.quantile(df.revenue_in, 0.9))
    assert f"{cap:,.0f}" in est.estimator.iloc[2]


def test_bootstrap_interval_brackets_the_point_estimate():
    df = _frame(seed=3)
    t, c = df.revenue_in[df.variant == "treatment"].to_numpy(), df.revenue_in[df.variant == "control"].to_numpy()
    lo, hi = bootstrap_ratio(t, c, reps=300)
    assert lo < t.mean() / c.mean() - 1 < hi and hi - lo > 0


def test_decompose_splits_purchase_rate_and_basket():
    df = pd.DataFrame({"variant": ["treatment"] * 4 + ["control"] * 4,
                       "revenue_in": [0, 0, 100, 100, 0, 0, 0, 100.0], "revenue_pre": [0.0] * 8})
    t = decompose(df)
    assert t.loc["treatment", "purchase_rate"] == 0.5 and t.loc["control", "purchase_rate"] == 0.25
    assert t.loc["relative lift", "purchase_rate"] == 1.0 and t.loc["relative lift", "revenue_per_user"] == 1.0
