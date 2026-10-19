import numpy as np
import pandas as pd

from src.experiments.interaction import interaction_effect
from src.experiments.novelty import decay_test, weekly_effects


def _users(n, week_rates_t, week_rate_c, seed):
    """n users per arm; each user has a personal activity level, so weeks are correlated like real data."""
    rng = np.random.default_rng(seed)
    rows = []
    for variant, rates in (("control", [week_rate_c] * len(week_rates_t)), ("treatment", week_rates_t)):
        level = rng.gamma(2.0, 0.5, n)                                       # mean 1.0
        for u in range(n):
            for wk, r in enumerate(rates, start=1):
                rows.append((wk, f"{variant}{u}", variant, rng.poisson(r * level[u])))
    return pd.DataFrame(rows, columns=["exposure_week", "user_id", "variant", "n_orders"])


def test_decay_test_detects_a_fading_effect_and_not_a_constant_one():
    fading = _users(5000, [0.20, 0.17, 0.14, 0.12], 0.10, seed=1)       # treatment 2x -> 1.2x of control
    constant = _users(5000, [0.12, 0.12, 0.12, 0.12], 0.10, seed=2)
    f, c = decay_test(fading), decay_test(constant)
    assert f["p_value"] < 0.001 and f["effect_change_per_week"] < -0.015       # true change: -0.027 per week
    assert c["p_value"] > 0.01 and abs(c["effect_change_per_week"]) < 0.01


def test_sparse_weeks_are_dropped_instead_of_producing_nonsense():
    df = _users(3000, [0.2, 0.15, 0.12], 0.1, seed=5)
    stray = pd.DataFrame([(4, "treatmentX", "treatment", 0)], columns=df.columns)       # one lone user in week 4
    both = pd.concat([df, stray], ignore_index=True)
    assert 4 not in weekly_effects(both).exposure_week.tolist()
    assert decay_test(both)["weeks_used"] == [1, 2, 3]


def test_weekly_effects_recovers_the_lift_in_each_week():
    df = _users(10000, [0.20, 0.15], 0.10, seed=3)
    w = weekly_effects(df).set_index("exposure_week")
    assert abs(w.loc[1, "rel_lift"] - 1.0) < 0.15 and abs(w.loc[2, "rel_lift"] - 0.5) < 0.12
    assert (w.ci_half > 0).all()


def test_interaction_is_zero_when_effects_add_and_nonzero_when_they_do_not():
    def cells(means, n=40_000, sd=1.0):
        rows = []
        for (a, b), m in means.items():
            var = sd**2
            rows.append({"variant_a": a, "variant_b": b, "users": n, "orders": m * n, "orders_sq": (var + m * m) * n})
        return pd.DataFrame(rows)
    additive = {("control", "control"): 1.0, ("control", "treatment"): 1.1, ("treatment", "control"): 1.2, ("treatment", "treatment"): 1.3}
    sub = {("control", "control"): 1.0, ("control", "treatment"): 1.1, ("treatment", "control"): 1.2, ("treatment", "treatment"): 1.1}
    assert interaction_effect(cells(additive))["p_value"] > 0.99
    r = interaction_effect(cells(sub))
    assert abs(r["interaction"] - (-0.2)) < 1e-9 and r["p_value"] < 0.001
