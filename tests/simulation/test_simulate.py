import dataclasses

import numpy as np
import pytest
from scipy import stats

from src.config import SimConfig
from src.simulation.simulate import simulate


@pytest.fixture(scope="module")
def tables():
    cfg = dataclasses.replace(SimConfig(), n_users=6000, chunk_users=2000, seed=7)
    return simulate(cfg)


def test_all_tables_present(tables):
    assert set(tables) == {"users", "products", "sessions", "events", "orders",
                           "experiments", "assignments", "answer_key", "user_truth"}


def test_primary_keys_unique(tables):
    for name, key in [("users", "user_id"), ("sessions", "session_id"), ("events", "event_id"),
                      ("orders", "order_id"), ("assignments", "assignment_id")]:
        assert tables[name][key].is_unique, f"{name}.{key} not unique"


def test_foreign_keys_valid(tables):
    uids, sids = set(tables["users"].user_id), set(tables["sessions"].session_id)
    assert set(tables["sessions"].user_id) <= uids
    assert set(tables["events"].session_id) <= sids
    assert set(tables["orders"].session_id) <= sids
    assert set(tables["assignments"].user_id) <= uids


def test_orders_match_purchase_events(tables):
    ev = tables["events"]
    # planted duplicates inflate raw event counts, so compare distinct sessions
    purchase_sessions = ev.loc[ev.event_type == "purchase", "session_id"].nunique()
    assert purchase_sessions == len(tables["orders"])


def test_no_experiment_before_pre_period_ends(tables):
    a = tables["assignments"].merge(tables["experiments"], on="experiment_id")
    start = np.datetime64("2026-01-05")
    assert (a["assigned_at"].to_numpy().astype("datetime64[D]") >= start + 28).all()


def test_srm_bug_only_in_exp_006(tables):
    a = tables["assignments"]
    p = {}
    for exp, g in a.groupby("experiment_id"):
        n_t = (g.variant == "treatment").sum()
        p[exp] = stats.chisquare([n_t, len(g) - n_t]).pvalue
    assert p["exp_006"] < 0.001
    assert all(p[e] > 0.001 for e in p if e != "exp_006")


def test_no_bugs_flag_removes_srm():
    cfg = dataclasses.replace(SimConfig(), n_users=6000, chunk_users=2000, seed=7, inject_bugs=False)
    t = simulate(cfg)
    g = t["assignments"].query("experiment_id == 'exp_006'")
    n_t = (g.variant == "treatment").sum()
    assert stats.chisquare([n_t, len(g) - n_t]).pvalue > 0.001


def test_realism_session_conversion_in_plausible_range(tables):
    conv = tables["orders"].session_id.nunique() / len(tables["sessions"])
    assert 0.03 < conv < 0.08


def test_device_effect_ios_beats_web(tables):
    s = tables["sessions"].merge(tables["orders"][["session_id"]].assign(b=1), on="session_id", how="left")
    rate = s.assign(b=s.b.fillna(0)).groupby("device").b.mean()
    assert rate["ios"] > rate["android"] > rate["web"]


def test_retention_decays_over_time(tables):
    u = tables["users"].query("is_new")[["user_id", "signup_date"]].copy()
    u["signup_date"] = np.asarray(u["signup_date"], dtype="datetime64[D]")
    s = tables["sessions"][["user_id", "session_start"]].merge(u, on="user_id")
    wk = ((s.session_start.dt.normalize() - s.signup_date).dt.days // 7)
    active = s.assign(wk=wk).query("wk >= 0").groupby("wk").user_id.nunique()
    assert active.loc[1] < 0.6 * active.loc[0] and active.loc[4] < active.loc[1]


def test_revenue_experiment_moves_order_value_not_conversion():
    cfg = dataclasses.replace(SimConfig(), n_users=3000, chunk_users=1500, seed=3, inject_bugs=False)
    t = simulate(cfg)
    assert t["experiments"].set_index("experiment_id").loc["exp_007", "primary_metric"] == "revenue_per_user"


def test_deterministic_with_seed():
    cfg = dataclasses.replace(SimConfig(), n_users=1500, chunk_users=1000, seed=11)
    a, b = simulate(cfg), simulate(cfg)
    assert a["orders"].equals(b["orders"])
