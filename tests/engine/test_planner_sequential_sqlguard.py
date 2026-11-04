import numpy as np
import pytest

from src.engine import planner, sequential
from src.engine.sqlguard import validate_readonly


# ------------------------------------------------------------------------------------------ planner
def test_binary_sample_size_matches_the_textbook_formula():
    # p0 = 10% -> 12% (20% relative), alpha .05, power .8: n per arm ~ 3,841 (two-proportion z, unpooled variance)
    n = planner.n_per_arm_binary(0.10, 0.20)
    assert 3_800 < n < 3_900


def test_plan_total_days_and_weeks_round_up_and_bonferroni_for_many_arms():
    two = planner.plan("rate", 0.10, 0.20, daily_users=1000, n_arms=2)
    four = planner.plan("rate", 0.10, 0.20, daily_users=1000, n_arms=4)
    assert two["total_users"] == 2 * two["n_per_arm"] and two["days"] == int(np.ceil(two["total_users"] / 1000))
    assert two["weeks"] == int(np.ceil(two["days"] / 7))
    assert four["alpha_used"] == pytest.approx(0.05 / 3) and four["n_per_arm"] > two["n_per_arm"]


def test_cuped_correlation_cuts_the_sample_by_one_minus_rho_squared():
    base = planner.plan("mean", 10.0, 0.05, sd=20.0)["n_per_arm"]
    adj = planner.plan("mean", 10.0, 0.05, sd=20.0, rho=0.6)["n_per_arm"]
    assert adj / base == pytest.approx(1 - 0.36, abs=0.01)


def test_mde_and_required_n_are_inverses_for_rates():
    n = planner.plan("rate", 0.05, 0.10)["n_per_arm"]
    assert planner.mde_for_n("rate", 0.05, n) == pytest.approx(0.10, abs=0.005)


def test_power_curve_is_monotone_and_hits_80_percent_at_the_planned_lift():
    p = planner.plan("rate", 0.05, 0.10)
    c = planner.power_curve("rate", 0.05, p["n_per_arm"], np.array([0.0, 0.05, 0.10, 0.2]))
    assert c.power.is_monotonic_increasing and abs(c.power.iloc[2] - 0.8) < 0.02 and abs(c.power.iloc[0] - 0.05) < 0.01


def test_bad_planner_inputs_raise():
    with pytest.raises(ValueError):
        planner.plan("rate", 0.95, 0.20)                       # 114% conversion is impossible
    with pytest.raises(ValueError):
        planner.plan("mean", 5.0, 0.1)                         # needs an SD


# ---------------------------------------------------------------------------------------- sequential
def test_peeking_inflates_naive_errors_and_the_corrections_hold():
    r = sequential.peeking_study(10, n_sims=40_000, seed=3).set_index("strategy")
    naive = r.false_positive_rate[r.index.str.contains("naive")].iloc[0]
    assert 0.17 < naive < 0.22                                  # known value for 10 looks is about 0.19
    assert abs(r.false_positive_rate.iloc[0] - 0.05) < 0.007    # one look: correct
    assert r.false_positive_rate[r.index.str.contains("Bonferroni")].iloc[0] < 0.035
    assert abs(r.false_positive_rate[r.index.str.contains("Fleming")].iloc[0] - 0.05) < 0.01


def test_false_positives_grow_with_the_number_of_looks():
    f = sequential.fpr_by_looks(20, n_sims=15_000)
    assert f.false_positive_rate.is_monotonic_increasing and f.false_positive_rate.iloc[0] == pytest.approx(0.05, abs=0.01)


# ------------------------------------------------------------------------------------------ sqlguard
@pytest.mark.parametrize("sql", [
    "SELECT * FROM users", "select 1;", "WITH a AS (SELECT 1) SELECT * FROM a", "EXPLAIN SELECT 1", "DESCRIBE users",
    "SELECT 'drop table x' AS note", "-- a comment\nSELECT 1", "SELECT 1 /* delete me */",
])
def test_read_only_queries_pass(sql):
    assert validate_readonly(sql)


@pytest.mark.parametrize("sql", [
    "DROP TABLE users", "DELETE FROM users", "INSERT INTO users VALUES (1)", "SELECT 1; DROP TABLE users", "UPDATE users SET city='x'",
    "COPY users TO '/tmp/x.csv'", "ATTACH 'x.db'", "INSTALL httpfs", "SELECT * FROM read_csv('/etc/passwd')",
    "SELECT * FROM read_parquet('x.parquet')", "PRAGMA database_list", "SET memory_limit='100GB'", "CREATE TABLE t AS SELECT 1",
    "SELECT getenv('HOME')", "", "   ", "x" * 6000,
])
def test_dangerous_or_empty_queries_are_rejected(sql):
    with pytest.raises(ValueError):
        validate_readonly(sql)


def test_the_locked_connection_cannot_touch_files_even_if_the_text_check_were_bypassed(tmp_path):
    import duckdb
    from src.engine.sqlguard import lock_down
    (tmp_path / "x.csv").write_text("a\n1\n")
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE t AS SELECT 1 AS a")
    lock_down(con)
    with pytest.raises(Exception):
        con.execute(f"SELECT * FROM read_csv('{tmp_path / 'x.csv'}')").fetchall()
    with pytest.raises(Exception):
        con.execute("SET enable_external_access=true")
    assert con.execute("SELECT * FROM t").fetchall() == [(1,)]
