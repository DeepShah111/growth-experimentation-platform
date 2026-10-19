import numpy as np

from src.targeting.uplift import fit_predict, ipw_effect, qini_area


def _synthetic(n=20_000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, n)                       # x=1 users respond (+1 order), x=0 users do not
    t = rng.integers(0, 2, n)
    y = rng.poisson(1.0 + t * x * 1.0).astype(float)
    return x, t, y


def test_ipw_effect_is_unbiased_for_the_average_effect():
    x, t, y = _synthetic()
    assert abs(ipw_effect(y, t).mean() - 0.5) < 0.06      # true ATE = 0.5 * 1.0


def test_qini_rewards_a_good_ranking_and_ignores_a_random_one():
    x, t, y = _synthetic()
    tau = ipw_effect(y, t)
    good = qini_area(x + 0.01 * np.random.default_rng(1).random(len(x)), tau)
    rand = qini_area(np.random.default_rng(2).random(len(x)), tau)
    assert good > 50 and abs(rand) < 15


def test_learners_find_the_responding_group():
    x, t, y = _synthetic()
    X = x.reshape(-1, 1).astype(float)
    for kind in "STX":
        u = fit_predict(kind, X, y, t, X, [False])
        assert u[x == 1].mean() > u[x == 0].mean() + 0.5, kind
