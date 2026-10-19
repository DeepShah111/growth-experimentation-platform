import numpy as np
import pandas as pd

from src.experiments.validation import one_replicate, summarise, wilson


def _runs(n=400, sd=0.03, half_factor=1.96, truth=0.08, bias=0.0, seed=0):
    rng = np.random.default_rng(seed)
    est = truth + bias + rng.normal(0, sd, n)
    half = np.full(n, half_factor * sd)
    z = est / sd
    from scipy import stats
    p = 2 * (1 - stats.norm.cdf(np.abs(z)))
    return pd.DataFrame({"seed": range(n), "experiment": "exp_x", "estimator": "naive", "truth": truth,
                         "lift": est, "half": half, "p": p})


def test_wilson_interval_matches_known_values():
    lo, hi = wilson(95, 100)
    assert abs(lo - 0.8882) < 0.002 and abs(hi - 0.9784) < 0.002
    lo0, hi0 = wilson(0, 50)
    assert lo0 == 0.0 or lo0 < 1e-12
    assert 0.0 < hi0 < 0.1


def test_summarise_calibrated_case_has_coverage_near_95_and_se_ratio_near_1():
    s = summarise(_runs(n=2000)).iloc[0]
    assert 0.93 < s.coverage < 0.97 and 0.93 < s.se_ratio < 1.07 and abs(s.bias) < 3 * s.bias_se


def test_summarise_detects_intervals_that_are_too_narrow_and_biased_estimates():
    narrow = summarise(_runs(n=1000, half_factor=0.8)).iloc[0]            # half-width 0.8 sd instead of 1.96 sd
    assert narrow.coverage < 0.65 and narrow.se_ratio < 0.5
    biased = summarise(_runs(n=1000, bias=0.09)).iloc[0]                     # bias = 3 sd, so coverage should be about 0.15
    assert biased.bias > 5 * biased.bias_se and biased.coverage < 0.3


def test_summarise_reports_cuped_width_relative_to_naive():
    a, b = _runs(n=200, sd=0.03), _runs(n=200, sd=0.03, seed=1)
    b["estimator"] = "cuped"; b["half"] = b["half"] * 0.9
    s = summarise(pd.concat([a, b])).set_index("estimator")
    assert abs(s.loc["cuped", "cuped_ci_width_vs_naive"] - 0.9) < 1e-9


def test_one_replicate_returns_both_estimators_for_known_truth_experiments():
    r = pd.DataFrame(one_replicate(seed=1, n_users=6_000))
    truth = r[r.estimator == "naive"].set_index("experiment").truth.to_dict()
    assert truth["exp_001"] == 0.08 and truth["exp_003"] == 0.0 and truth["exp_007"] == 0.06
    assert set(r.estimator) == {"naive", "cuped"} and r.half.gt(0).all()
