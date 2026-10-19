import numpy as np

from src.experiments.stats_utils import (analytic_power, benjamini_hochberg, bonferroni, holm,
                             mde_rel, obrien_fleming_bounds, required_n_per_arm)


def test_required_n_matches_textbook():
    # d = 0.2 SD, alpha .05, power .8  ->  n per arm ~ 392
    n = required_n_per_arm(rel_lift=0.2, mean=1.0, sd=1.0)
    assert 390 < n < 395


def test_power_at_required_n_is_80_percent():
    n = required_n_per_arm(0.05, mean=2.0, sd=3.0)
    assert abs(analytic_power(0.05, n, 2.0, 3.0) - 0.80) < 0.01


def test_mde_and_required_n_are_inverses():
    n = 25_000
    m = mde_rel(n, mean=0.8, sd=1.7)
    assert abs(required_n_per_arm(m, 0.8, 1.7) - n) < 1


def test_cuped_rho_reduces_required_n():
    base = required_n_per_arm(0.05, 1.0, 2.0)
    assert required_n_per_arm(0.05, 1.0, 2.0, rho=0.6) < 0.7 * base


def test_multiple_testing_known_example():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    assert bonferroni(p).sum() == 1          # 0.05/10 = 0.005
    assert holm(p).sum() == 1
    assert benjamini_hochberg(p).sum() == 2  # standard BH illustration


def test_obf_bounds_are_decreasing_and_control_error():
    b = obrien_fleming_bounds(4, alpha=0.05, n_sims=100_000, seed=1)
    assert b[0] > b[1] > b[2] > b[3] > 1.9
    rng = np.random.default_rng(99)
    z = np.abs(np.cumsum(rng.standard_normal((100_000, 4)), axis=1)) / np.sqrt(np.arange(1, 5))
    assert abs((z > b).any(axis=1).mean() - 0.05) < 0.006
