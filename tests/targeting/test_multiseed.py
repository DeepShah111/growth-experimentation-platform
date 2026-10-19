import numpy as np
import pandas as pd

from src.targeting.multiseed import holdout_profit, paired_bootstrap, run_seeds, summarise


def _data(n=20_000, seed=0):
    rng = np.random.default_rng(seed)
    resp = rng.integers(0, 2, n)                       # responders (+1.0 order) vs non-responders (0)
    t = rng.integers(0, 2, n)
    y = rng.poisson(1.0 + t * resp * 1.0).astype(float)
    d = pd.DataFrame({"orders_pre": rng.poisson(1.0, n).astype(float), "revenue_pre": rng.gamma(2, 50, n),
                      "sessions_pre": rng.poisson(5, n).astype(float), "days_since_signup": rng.integers(1, 300, n).astype(float),
                      "is_new": resp.astype(float), "device": rng.choice(["a", "b"], n), "channel": rng.choice(["x", "y"], n),
                      "city": rng.choice(["p", "q"], n), "orders_in": y})
    return d, t, resp


def test_holdout_profit_is_less_optimistic_than_in_sample_for_a_useless_score():
    rng = np.random.default_rng(0)
    n = 40_000
    t = rng.integers(0, 2, n); y = rng.poisson(1.0 + 0.0 * t).astype(float)       # true effect is zero
    gaps = []
    for s in range(20):
        r = holdout_profit(rng.random(n), y, t, np.random.default_rng(s), value=50.0, cost=0.0)
        gaps.append(r["profit_in_sample"] - r["profit_holdout"])
    assert np.mean(gaps) > 0                                                       # selection optimism shows up


def test_good_score_earns_money_on_the_holdout_and_treating_everyone_does_not():
    d, t, resp = _data()
    y = d.orders_in.to_numpy()
    r = holdout_profit(resp + 0.01 * np.random.default_rng(1).random(len(d)), y, t, np.random.default_rng(2), value=50.0, cost=30.0)
    assert r["profit_holdout"] > 5_000 and r["treat_everyone_holdout"] < r["profit_holdout"]
    assert 0.35 < r["share_treated_on_B"] < 0.65                                  # about the responding half


def test_run_seeds_and_summary_shapes_and_best_learner_counts():
    d, t, resp = _data(12_000)
    res, first = run_seeds(d, t, [1, 2, 3])
    assert set(res.model) == {"S-learner", "T-learner", "X-learner", "random"} and len(res) == 12
    summ, pair = summarise(res)
    assert summ.best_learner_in_n_seeds.sum() == 3 and pair["n_seeds"] == 3
    assert summ.loc["S-learner", "qini_mean"] > summ.loc["random", "qini_mean"] + 20


def test_paired_bootstrap_separates_a_good_ranking_from_a_random_one():
    d, t, resp = _data(12_000, seed=4)
    from src.targeting.uplift import ipw_effect
    tau = ipw_effect(d.orders_in.to_numpy(), t)
    rng = np.random.default_rng(5)
    scores = pd.DataFrame({"S-learner": resp + 0.01 * rng.random(len(d)), "X-learner": rng.random(len(d)), "T-learner": rng.random(len(d))})
    pb = paired_bootstrap(scores, tau, reps=100).set_index("difference")
    assert pb.loc["S-learner minus X-learner", "interval_excludes_zero"] and pb.loc["S-learner minus X-learner", "mean"] > 20
    assert not pb.loc["X-learner minus T-learner", "interval_excludes_zero"]
