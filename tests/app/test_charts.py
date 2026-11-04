"""Chart builders: valid specs, and the maths behind the interactive profit sliders."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("altair")
import sys                                                   # noqa: E402
from pathlib import Path                                      # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app"))
import charts as C                                            # noqa: E402


def test_profit_curve_reproduces_the_saved_curve_at_the_original_economics():
    n = 100_000
    f = np.linspace(0.01, 1.0, 100)
    gain = 3_000 * np.sqrt(f)                                  # any concave cumulative gain in orders
    saved = pd.DataFrame({"fraction_treated": f, "profit_inr": 50.0 * gain - 3.0 * f * n})
    same = C.profit_curve(saved, value=50.0, cost=3.0, n_users=n)
    assert np.allclose(same.profit, saved.profit_inr)


def test_profit_curve_matches_a_hand_calculation_for_new_economics():
    n = 1_000
    saved = pd.DataFrame({"fraction_treated": [0.5, 1.0], "profit_inr": [50.0 * 100 - 3.0 * 500, 50.0 * 120 - 3.0 * 1000]})   # 100 and 120 extra orders
    new = C.profit_curve(saved, value=80.0, cost=1.0, n_users=n)
    assert np.allclose(new.profit, [80 * 100 - 1 * 500, 80 * 120 - 1 * 1000])
    zero = C.profit_curve(saved, value=0.0, cost=0.0, n_users=n)
    assert np.allclose(zero.profit, 0.0)


def test_partial_last_day_is_hidden_only_when_it_is_really_partial():
    d = pd.DataFrame({"revenue": [100.0] * 10 + [3.0]})
    assert len(C.drop_partial_last_day(d, "revenue")) == 10
    d2 = pd.DataFrame({"revenue": [100.0] * 10 + [90.0]})
    assert len(C.drop_partial_last_day(d2, "revenue")) == 11


def test_profit_chart_marks_treat_everyone_only_when_the_curve_reaches_100_percent():
    short = pd.DataFrame({"fraction_treated": [0.1, 0.5, 0.998], "profit": [10.0, 20.0, -5.0]})
    full = pd.DataFrame({"fraction_treated": [0.1, 0.5, 1.0], "profit": [10.0, 20.0, -5.0]})
    assert len(C.profit_chart(short).to_dict()["layer"]) == 3
    assert len(C.profit_chart(full).to_dict()["layer"]) == 4


def test_forest_chart_only_lists_decisions_that_occur_and_flags_untrusted_rows():
    sc = pd.DataFrame({"experiment": ["e1", "e2"], "name": ["a", "b"], "decision": ["SHIP", "DO NOT TRUST"], "lift": [0.1, 0.2], "ci_low": [0.05, 0.1], "ci_high": [0.15, 0.3]})
    spec = C.forest_chart(sc).to_dict()
    colour = [l for l in spec["layer"] if "color" in l.get("encoding", {}) and "scale" in l["encoding"]["color"]][0]["encoding"]["color"]["scale"]
    assert colour["domain"] == ["SHIP", "DO NOT TRUST"]
    assert any("do not trust" in str(l) for l in spec["layer"])


def test_uplift_models_chart_is_sorted_by_score_and_hides_the_reference_row():
    d = pd.DataFrame({"model": ["T-learner", "S-learner", "random", "ref: rank by true relative lift (not a ceiling)"],
                      "qini_orders_per_1000_users": [15.0, 25.0, -0.2, 18.0], "ci_low": [8.0, 19.0, -4.0, 13.0], "ci_high": [21.0, 31.0, 4.0, 23.0]})
    spec = C.uplift_models_chart(d).to_dict()
    order = [l for l in spec["layer"] if l.get("encoding", {}).get("y", {}).get("sort")][0]["encoding"]["y"]["sort"]
    assert order == ["S-learner", "T-learner", "random"]


def test_every_builder_returns_a_valid_altair_spec_on_minimal_data():
    f = pd.DataFrame({"experiment": ["e1"], "name": ["a"], "decision": ["SHIP"], "lift": [0.1], "ci_low": [0.05], "ci_high": [0.15], "n_treat": [10], "n_ctrl": [10], "srm_p": [0.5]})
    for ch in (C.forest_chart(f), C.srm_chart(f)):
        assert ch.to_dict()
    wk = pd.DataFrame({"exposure_week": [1, 2, 3], "rel_lift": [0.1, 0.08, 0.05], "ci_half": [0.04, 0.04, 0.05]})
    assert C.weekly_chart(wk).to_dict()
    fl = pd.DataFrame({"exp": ["exp_001"], "name": ["a"], "naive_lift": [0.1], "naive_CI±": [0.03], "cuped_lift": [0.11], "cuped_CI±": [0.03], "true_nominal": [0.08]})
    assert C.truth_chart(fl).to_dict()
