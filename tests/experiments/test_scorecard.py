import dataclasses

import duckdb
import pytest

from src.config import SimConfig
from src.experiments import scorecard as sc
from src.simulation.simulate import simulate


def row(**kw):
    base = dict(srm_p=0.5, n_treat=1000, n_ctrl=1000, lift=0.10, ci_low=0.05, ci_high=0.15, p_adj=0.001, significant=True,
                guardrail_breach=False, guardrail_lift=0.0, interaction_p=None, overlap_with="", trend_p=None,
                trend_slope=None, latest_week_lift=None, latest_week_ci_half=None, underpowered=False, mde=0.02,
                net_value=float('nan'), net_ci_low=float('nan'), net_ci_high=float('nan'))
    base.update(kw)
    return base


def test_sample_ratio_failure_blocks_everything_even_a_big_lift():
    d, why = sc.decide(row(srm_p=1e-9, lift=0.5))
    assert d == "DO NOT TRUST" and "sample ratio mismatch" in why[0]


def test_srm_threshold_is_strict_and_borderline_values_only_add_a_note():
    assert sc.decide(row(srm_p=0.001))[0] == "SHIP"                 # exactly at the threshold is not a failure
    d, notes = sc.decide(row(srm_p=0.06))
    assert d == "SHIP" and any("borderline" in n for n in notes)


def test_significant_negative_is_not_shipped():
    assert sc.decide(row(lift=-0.05, ci_low=-0.08, ci_high=-0.02))[0] == "DO NOT SHIP (harmful)"


def test_clean_significant_win_ships_and_low_ci_end_is_noted():
    d, notes = sc.decide(row(ci_low=0.01))
    assert d == "SHIP" and any("worth-shipping bar" in n for n in notes)


@pytest.mark.parametrize("kw,needle", [
    (dict(guardrail_breach=True, guardrail_lift=-0.04), "guardrail"),
    (dict(interaction_p=0.01, overlap_with="exp_x"), "interacts"),
    (dict(trend_p=0.01, trend_slope=-0.002), "fading"),
])
def test_significant_win_with_a_blocker_is_iterate(kw, needle):
    d, why = sc.decide(row(**kw))
    assert d == "ITERATE" and needle in " ".join(why)


def test_a_significant_trend_that_is_not_negative_does_not_block():
    assert sc.decide(row(trend_p=0.01, trend_slope=+0.002))[0] == "SHIP"
    assert sc.decide(row(trend_p=0.40, trend_slope=-0.002))[0] == "SHIP"


def test_overlap_without_detected_interaction_is_only_a_note():
    d, notes = sc.decide(row(overlap_with="exp_x", interaction_p=0.4))
    assert d == "SHIP" and any("exp_x" in n for n in notes)


def test_non_significant_result_is_inconclusive_when_underpowered_else_no_effect():
    assert sc.decide(row(significant=False, lift=0.01, underpowered=True, mde=0.05))[0] == "INCONCLUSIVE (underpowered)"
    assert sc.decide(row(significant=False, lift=0.001, underpowered=False, mde=0.02))[0] == "NO EFFECT FOUND"


def test_scorecard_builds_without_the_answer_key_and_catches_the_srm_bug():
    cfg = dataclasses.replace(SimConfig(), n_users=10_000, chunk_users=5_000, seed=7)
    tables = simulate(cfg)
    con = duckdb.connect(":memory:")
    for name in ("users", "sessions", "orders", "experiments", "assignments"):        # NOTE: no answer_key tables at all
        con.register(f"_{name}", tables[name]); con.execute(f"CREATE TABLE {name} AS SELECT * FROM _{name}")
    t = sc.build(con, min_users_week=300).set_index("experiment")
    assert t.loc["exp_006", "decision"] == "DO NOT TRUST"
    assert set(t.decision) <= {"DO NOT TRUST", "DO NOT SHIP (harmful)", "SHIP", "ITERATE", "INCONCLUSIVE (underpowered)", "NO EFFECT FOUND"}
    assert t.loc[t.index != "exp_006", "srm_p"].min() > 0.001 and t.p_adj.drop("exp_006").between(0, 1).all()
    assert len(t) == 9 and t.reasons.str.len().min() > 10
    import numpy as np
    from src.experiments.stats_utils import holm_adjust
    trusted = t.drop(index="exp_006")                                               # the Holm family = trusted experiments only
    assert np.allclose(trusted.p_adj.to_numpy(), holm_adjust(trusted.p_value.to_numpy()))
    assert np.isnan(t.loc["exp_006", "p_adj"])


def test_holm_adjusted_p_values_match_a_hand_calculation_and_agree_with_holm():
    import numpy as np
    from src.experiments.stats_utils import holm, holm_adjust
    p = np.array([0.01, 0.04, 0.03])
    # sorted 0.01, 0.03, 0.04 -> x3, x2, x1 = 0.03, 0.06, 0.04 -> made monotone: 0.03, 0.06, 0.06
    assert np.allclose(holm_adjust(p), [0.03, 0.06, 0.06])
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rng.random(8) ** 3
        assert ((holm_adjust(q) <= 0.05) == holm(q, 0.05)).all()


def test_net_value_matches_a_hand_calculation():
    import numpy as np
    t = np.array([1.0, 2.0, 3.0, 2.0]); c = np.array([1.0, 1.0, 1.0, 1.0])
    net, lo, hi = sc.net_value_per_user(t, c, value_per_order=50.0, cost_per_treated=3.0)
    assert abs(net - (50 * 1.0 - 3.0)) < 1e-12                       # mean diff = 2.0 - 1.0 = 1.0
    se = np.sqrt(t.var(ddof=1) / 4 + 0.0)
    assert abs(lo - (50 * (1.0 - 1.96 * se) - 3.0)) < 1e-12 and abs(hi - (50 * (1.0 + 1.96 * se) - 3.0)) < 1e-12


def test_a_significant_win_that_loses_money_is_iterate_not_ship():
    d, why = sc.decide(row(net_value=-1.29, net_ci_low=-1.9, net_ci_high=-0.7))
    assert d == "ITERATE" and "unprofitable" in " ".join(why) and "target a subset" in " ".join(why)


def test_profitable_or_uncertain_economics_does_not_block_and_is_noted():
    d, notes = sc.decide(row(net_value=2.0, net_ci_low=0.5, net_ci_high=3.5))
    assert d == "SHIP" and any("profit at the assumed economics" in n for n in notes)
    d2, _ = sc.decide(row(net_value=-0.4, net_ci_low=-1.5, net_ci_high=0.7))              # interval includes zero: not proven unprofitable
    assert d2 == "SHIP"
