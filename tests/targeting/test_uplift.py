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


def test_design_matrix_uses_only_the_eight_pre_experiment_features():
    """Leakage guard: outcomes, the treatment flag and anything from the answer key must never become model features."""
    import numpy as np
    import pandas as pd
    from src.targeting.uplift import CAT, NUM, design_matrix
    n = 50
    d = pd.DataFrame({c: np.arange(n, dtype=float) for c in NUM})
    for c in CAT:
        d[c] = np.where(np.arange(n) % 2 == 0, "a", "b")
    for leak in ("orders_in", "revenue_in", "variant", "true_relative_lift", "segment", "user_id"):
        d[leak] = np.arange(n, dtype=float)
    X, cat_mask = design_matrix(d)
    assert X.shape == (n, len(NUM) + len(CAT)) == (n, 8) and len(cat_mask) == 8
    assert set(NUM) | set(CAT) == {"orders_pre", "revenue_pre", "sessions_pre", "days_since_signup", "is_new", "device", "channel", "city"}
    assert not any(c.endswith("_in") for c in NUM + CAT)                 # nothing measured after assignment
