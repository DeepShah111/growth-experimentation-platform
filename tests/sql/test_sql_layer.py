"""Every SQL file is run on a tiny hand-built database where the right answer was worked out by hand."""
import duckdb
import numpy as np
import pandas as pd
import pytest

from src.config import SQL_DIR
from src.warehouse.build_warehouse import apply_staging
from src.warehouse.run_sql import run_query


def ts(s):
    return pd.Timestamp(s)


def make_con(**tables):
    """In-memory DuckDB with the given pandas frames as tables, plus the staging views."""
    con = duckdb.connect(":memory:")
    base = {
        "events": pd.DataFrame({"event_id": pd.Series(dtype="int64"), "session_id": pd.Series(dtype="int64"),
                                "user_id": pd.Series(dtype="int64"), "event_type": pd.Series(dtype="str"),
                                "event_ts": pd.Series(dtype="datetime64[ns]"), "product_id": pd.Series(dtype="Int64"),
                                "ingested_at": pd.Series(dtype="datetime64[ns]")}),
    }
    base.update(tables)
    for name, df in base.items():
        con.register(f"_{name}", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM _{name}")
    if "events" in base:
        apply_staging(con)
    return con


def ev(rows):
    """rows: (event_id, session_id, user_id, type, event_ts, product_id, ingested_at or None)"""
    df = pd.DataFrame(rows, columns=["event_id", "session_id", "user_id", "event_type", "event_ts",
                                     "product_id", "ingested_at"])
    df["event_ts"] = pd.to_datetime(df.event_ts)
    df["ingested_at"] = pd.to_datetime(df.ingested_at.fillna(df.event_ts.astype(str)))
    df["product_id"] = df.product_id.astype("Int64")
    return df


# ------------------------------------------------------------------------------------ staging
def test_staging_removes_duplicates_including_null_product_and_flags_late_events():
    events = ev([
        (1, 10, 1, "product_view", "2026-01-05 10:00:00", 7, None),
        (2, 10, 1, "product_view", "2026-01-05 10:00:00", 7, None),          # exact duplicate of 1
        (3, 10, 1, "purchase",     "2026-01-05 10:05:00", None, None),
        (4, 10, 1, "purchase",     "2026-01-05 10:05:00", None, None),       # duplicate with NULL product
        (5, 10, 1, "product_view", "2026-01-05 10:01:00", 8, "2026-01-07 10:01:00"),   # 2 days late
        (6, 10, 1, "product_view", "2026-01-05 10:02:00", 9, "2026-01-06 10:02:00"),   # exactly 1 day: not late
    ])
    con = make_con(events=events)
    out = con.execute("SELECT event_id, is_late FROM stg_events_clean ORDER BY event_id").fetchall()
    assert out == [(1, False), (3, False), (5, True), (6, False)]


# ------------------------------------------------------------------------------------ funnel
def test_funnel_by_device_exact_rates():
    sessions = pd.DataFrame({"session_id": [1, 2, 3], "user_id": [1, 2, 3], "device": ["android", "android", "ios"],
                             "session_start": pd.to_datetime(["2026-01-05"] * 3), "duration_sec": [60] * 3})
    events = ev([
        (1, 1, 1, "product_view", "2026-01-05 10:00:00", 1, None),
        (2, 1, 1, "product_view", "2026-01-05 10:00:00", 1, None),        # duplicate, must not matter
        (3, 1, 1, "add_to_cart", "2026-01-05 10:01:00", 1, None),
        (4, 1, 1, "checkout_start", "2026-01-05 10:02:00", None, None),
        (5, 1, 1, "purchase", "2026-01-05 10:03:00", None, None),
        (6, 2, 2, "product_view", "2026-01-05 11:00:00", 2, None),
        (7, 3, 3, "product_view", "2026-01-05 12:00:00", 3, None),
        (8, 3, 3, "add_to_cart", "2026-01-05 12:01:00", 3, None),
    ])
    con = make_con(events=events, sessions=sessions)
    r = run_query(con, "funnel_by_device").set_index("device")
    a = r.loc["android"]
    assert (a.sessions, a.cart_rate, a.cart_to_checkout, a.checkout_to_purchase, a.session_conversion) == (2, 0.5, 1.0, 1.0, 0.5)
    i = r.loc["ios"]
    assert (i.sessions, i.cart_rate, i.cart_to_checkout, i.session_conversion) == (1, 1.0, 0.0, 0.0)
    assert np.isnan(i.checkout_to_purchase)           # nobody reached checkout: undefined, not 0
    assert list(r.index) == ["android", "ios"]        # ordered by sessions, descending


# ------------------------------------------------------------------------------------ sessionization
def test_sessionization_rebuilds_sessions_from_gaps():
    # user 1: 10:00, 10:30 (gap exactly 30 min -> same session), 11:01 (gap 31 min -> new), 11:05 (same) => 2
    # user 2: one event => 1.   Logged sessions: 3 for user 1, 1 for user 2 => ratio 3/4.
    events = ev([
        (1, 1, 1, "product_view", "2026-01-05 10:00:00", 1, None),
        (2, 1, 1, "product_view", "2026-01-05 10:30:00", 2, None),
        (3, 2, 1, "product_view", "2026-01-05 11:01:00", 3, None),
        (4, 2, 1, "product_view", "2026-01-05 11:05:00", 4, None),
        (5, 4, 2, "product_view", "2026-01-05 09:00:00", 5, None),
    ])
    users = pd.DataFrame({"user_id": [1, 2], "device": ["android", "android"]})
    sessions = pd.DataFrame({"session_id": [1, 2, 3, 4], "user_id": [1, 1, 1, 2], "device": ["android"] * 4,
                             "session_start": pd.to_datetime(["2026-01-05"] * 4), "duration_sec": [1] * 4})
    con = make_con(events=events, sessions=sessions, users=users)
    r = run_query(con, "sessionization_check").iloc[0]
    assert (r.logged_sessions, r.rebuilt_sessions, r.rebuilt_to_logged_ratio) == (4, 3, 0.75)


# ------------------------------------------------------------------------------------ window functions
def test_daily_revenue_running_total_moving_average_and_lags():
    revs = [150, 300, 150, 100, 100, 100, 100, 300]           # day 1 is made of two orders (100 + 50)
    rows, oid = [], 0
    for d, r in enumerate(revs):
        parts = [100.0, 50.0] if d == 0 else [float(r)]
        for p in parts:
            oid += 1
            rows.append((oid, 1, 1, pd.Timestamp("2026-01-05") + pd.Timedelta(days=d, hours=10), 1, p, 0.0, p))
    orders = pd.DataFrame(rows, columns=["order_id", "session_id", "user_id", "order_ts", "n_items",
                                         "gross_value", "discount_pct", "order_value"])
    r = run_query(make_con(orders=orders), "daily_revenue_trends")
    assert list(r.orders)[:2] == [2, 1]
    assert r.cumulative_revenue.iloc[2] == 600.0
    assert np.isnan(r.dod_change.iloc[0]) and r.dod_change.iloc[1] == 1.0 and r.dod_change.iloc[2] == -0.5
    assert r.revenue_7d_avg.iloc[6] == round(1000 / 7, 2) and r.revenue_7d_avg.iloc[7] == round(1150 / 7, 2)
    assert np.isnan(r.wow_change.iloc[6]) and r.wow_change.iloc[7] == 1.0     # day 8 (300) vs day 1 (150)


def test_top_products_per_category_ranks_ties_and_cuts_at_three():
    products = pd.DataFrame({"product_id": [1, 2, 3, 4, 5, 6, 7],
                             "category": ["A", "A", "A", "A", "A", "C", "C"], "price_inr": [10.0] * 7})
    carts = {1: 5, 2: 4, 3: 3, 4: 2, 5: 0, 6: 3, 7: 3}
    rows, eid = [], 0
    for p, n in carts.items():
        for k in range(n):
            eid += 1
            rows.append((eid, 1, 1, "add_to_cart", f"2026-01-05 10:{k:02d}:{p:02d}", p, None))
    rows.append((eid + 1, 1, 1, "add_to_cart", "2026-01-05 10:00:01", 1, None))        # exact duplicate of a product-1 cart event
    rows[-1] = (eid + 1, 1, 1, "add_to_cart", rows[0][4], 1, None)
    con = make_con(events=ev(rows), products=products)
    r = run_query(con, "top_products_per_category")
    a = r[r.category == "A"]
    assert list(a.product_id) == [1, 2, 3] and list(a.rank_in_category) == [1, 2, 3]   # rank 4 (product 4) cut
    assert a.cart_adds.iloc[0] == 5                                                      # duplicate not counted
    c = r[r.category == "C"]
    assert list(c.product_id) == [6, 7] and list(c.rank_in_category) == [1, 1]          # tie shares rank


def test_rfm_scores_segments_and_ignores_orders_after_as_of():
    spec = {1: (5, "2026-02-28", 500), 2: (4, "2026-02-20", 400), 3: (3, "2026-02-10", 300),
            4: (2, "2026-01-30", 200), 5: (1, "2026-01-01", 100)}
    rows, oid = [], 0
    for u, (n, last, tot) in spec.items():
        for k in range(n):
            oid += 1
            day = pd.Timestamp(last) - pd.Timedelta(days=k)
            rows.append((oid, 1, u, day + pd.Timedelta(hours=9), 1, tot / n, 0.0, tot / n))
    rows.append((oid + 1, 1, 5, pd.Timestamp("2026-03-05 09:00"), 1, 999.0, 0.0, 999.0))   # after as_of: ignored
    orders = pd.DataFrame(rows, columns=["order_id", "session_id", "user_id", "order_ts", "n_items",
                                         "gross_value", "discount_pct", "order_value"])
    r = run_query(make_con(orders=orders), "rfm_segments", {"as_of": "2026-03-01"}).set_index("segment")
    assert r.loc["champions", "users"] == 2 and r.loc["loyal", "users"] == 1 and r.loc["lost", "users"] == 2
    assert r.loc["champions", "share_of_revenue"] == 0.6 and r.loc["loyal", "share_of_revenue"] == 0.2
    assert list(r.index)[0] == "champions"


def test_activity_streaks_gaps_and_islands():
    def days(u, ds):
        return [(u, pd.Timestamp(d) + pd.Timedelta(hours=h)) for d, h in ds]
    rows = (days(1, [("2026-01-01", 9), ("2026-01-02", 9), ("2026-01-03", 9), ("2026-01-05", 9), ("2026-01-06", 9)])
            + days(2, [("2026-01-01", 9), ("2026-01-03", 9), ("2026-01-05", 9)])
            + days(3, [(f"2026-01-{d}", h) for d in range(10, 15) for h in (8, 20)])        # two sessions per day
            + days(4, [("2026-02-01", 9), ("2026-02-02", 9), ("2026-02-03", 9)]))
    sessions = pd.DataFrame(rows, columns=["user_id", "session_start"])
    sessions["session_id"] = range(1, len(sessions) + 1)
    r = run_query(make_con(sessions=sessions), "activity_streaks")
    assert dict(zip(r.longest_streak, r.users)) == {1: 1, 3: 2, 5: 1}      # user 2 -> 1, users 1 and 4 -> 3, user 3 -> 5


# ------------------------------------------------------------------------------------ data quality
def test_data_quality_counts_duplicates_and_late_events_per_day():
    events = ev([
        (1, 1, 1, "product_view", "2026-01-05 10:00:00", 1, None),
        (2, 1, 1, "product_view", "2026-01-05 10:00:00", 1, None),                       # duplicate
        (3, 1, 1, "product_view", "2026-01-05 10:01:00", 2, "2026-01-07 10:01:00"),      # late
        (4, 1, 1, "product_view", "2026-01-05 10:02:00", 3, None),
        (5, 1, 1, "product_view", "2026-01-05 10:03:00", 4, None),
        (6, 2, 1, "product_view", "2026-01-06 10:00:00", 1, None),
        (7, 2, 1, "product_view", "2026-01-06 10:01:00", 2, None),
    ])
    r = run_query(make_con(events=events), "data_quality_events")
    d1, d2 = r.iloc[0], r.iloc[1]
    assert (d1.raw_events, d1.distinct_events, d1.duplicates_removed, d1.duplicate_rate) == (5, 4, 1, 0.2)
    assert (d1.late_events, d1.late_rate) == (1, 0.25)
    assert (d2.duplicates_removed, d2.duplicate_rate, d2.late_events) == (0, 0.0, 0)


# ------------------------------------------------------------------------------------ experiments
def _exp_tables():
    experiments = pd.DataFrame({"experiment_id": ["exp_a", "exp_b", "exp_c"], "name": ["a", "b", "c"],
                                "start_date": pd.to_datetime(["2026-02-02", "2026-02-09", "2026-03-10"]).date,
                                "end_date": pd.to_datetime(["2026-02-15", "2026-02-22", "2026-03-20"]).date,
                                "primary_metric": ["orders_per_user"] * 3,
                                "exclusive_group": ["g", "g", "g"]})
    return experiments


def test_srm_counts_and_overlap_detector():
    a = pd.DataFrame({"assignment_id": range(1, 8),
                      "experiment_id": ["exp_a", "exp_a", "exp_a", "exp_b", "exp_b", "exp_c", "exp_c"],
                      "user_id": [1, 2, 3, 1, 2, 1, 2],
                      "variant": ["control", "treatment", "treatment", "control", "treatment", "control", "treatment"],
                      "assigned_at": pd.to_datetime(["2026-02-02"] * 7)})
    con = make_con(assignments=a, experiments=_exp_tables())
    srm = run_query(con, "srm_counts").set_index("experiment_id")
    assert (srm.loc["exp_a", "control_users"], srm.loc["exp_a", "treatment_users"]) == (1, 2)
    ov = run_query(con, "overlapping_experiments")
    # exp_a and exp_b overlap in time and share users 1, 2. exp_c is in the same group but at a different time.
    assert len(ov) == 1 and (ov.exp_a[0], ov.exp_b[0], ov.shared_users[0]) == ("exp_a", "exp_b", 2)


def test_weekly_retention_cohort_math():
    users = pd.DataFrame({"user_id": [1, 2, 3], "signup_date": pd.to_datetime(["2026-01-06", "2026-01-07", "2026-01-14"]).date,
                          "is_new": [True, True, False], "city": ["x"] * 3, "device": ["ios"] * 3, "channel": ["paid"] * 3})
    sess = pd.DataFrame({"session_id": [1, 2, 3, 4], "user_id": [1, 1, 2, 3],
                         "session_start": pd.to_datetime(["2026-01-06 10:00", "2026-01-13 10:00", "2026-01-07 10:00", "2026-01-14 10:00"]),  # user 3: not new, active in own signup week
                         "duration_sec": [1] * 4, "device": ["ios"] * 4})
    r = run_query(make_con(users=users, sessions=sess), "weekly_retention")
    assert list(r.cohort_size.unique()) == [2] and r.cohort_week.nunique() == 1   # the not-new user forms no cohort
    assert list(r.weeks_since_signup) == [0, 1] and list(r.active_users) == [2, 1] and list(r.retention) == [1.0, 0.5]


def test_user_weekly_mart_uses_complete_weeks_from_each_users_own_assignment():
    a = pd.DataFrame({"assignment_id": [1, 2, 3], "experiment_id": ["exp_a"] * 3, "user_id": [1, 2, 3],
                      "variant": ["control", "treatment", "control"],
                      "assigned_at": pd.to_datetime(["2026-02-02", "2026-02-05", "2026-02-02"])})
    def o(i, u, t): return (i, 1, u, pd.Timestamp(t), 1, 10.0, 0.0, 10.0)
    orders = pd.DataFrame([o(1, 1, "2026-02-03 10:00"), o(2, 1, "2026-02-10 10:00"), o(3, 1, "2026-02-11 10:00"),
                           o(4, 2, "2026-02-06 10:00"), o(5, 2, "2026-02-13 10:00"),
                           o(6, 3, "2026-02-01 10:00")],                            # before assignment: ignored
                          columns=["order_id", "session_id", "user_id", "order_ts", "n_items", "gross_value",
                                   "discount_pct", "order_value"])
    con = make_con(assignments=a, experiments=_exp_tables(), orders=orders)
    r = con.execute((SQL_DIR / "marts" / "experiment_user_weekly.sql").read_text(), {"exp_id": "exp_a"}).df()
    got = [(int(w), int(u), v, int(n)) for w, u, v, n in r[["exposure_week", "user_id", "variant", "n_orders"]].itertuples(index=False)]
    # exp_a ends 2026-02-15 (inclusive). User 2 (assigned Feb 5) has only ONE complete week (Feb 5-12); week 2 would spill past the end.
    assert got == [(1, 1, "control", 1), (1, 2, "treatment", 1), (1, 3, "control", 0),
                   (2, 1, "control", 2), (2, 3, "control", 0)]


def test_experiment_interaction_cells_use_the_overlap_window_only():
    a = pd.DataFrame({"assignment_id": [1, 2, 3, 4, 5], "experiment_id": ["exp_a", "exp_b", "exp_a", "exp_b", "exp_a"],
                      "user_id": [1, 1, 2, 2, 3], "variant": ["control", "control", "treatment", "treatment", "treatment"],
                      "assigned_at": pd.to_datetime(["2026-02-02", "2026-02-09", "2026-02-02", "2026-02-09", "2026-02-02"])})
    def o(i, u, t): return (i, 1, u, pd.Timestamp(t), 1, 10.0, 0.0, 10.0)
    orders = pd.DataFrame([o(1, 1, "2026-02-08 10:00"),      # before both are active: ignored
                           o(2, 1, "2026-02-10 10:00"),      # inside the overlap: counted
                           o(3, 2, "2026-02-12 10:00"),      # counted
                           o(4, 2, "2026-02-18 10:00")],     # exp_a is over by then: ignored
                          columns=["order_id", "session_id", "user_id", "order_ts", "n_items", "gross_value",
                                   "discount_pct", "order_value"])
    con = make_con(assignments=a, experiments=_exp_tables(), orders=orders)
    r = run_query(con, "experiment_interaction", {"exp_a": "exp_a", "exp_b": "exp_b"})
    got = [(x.variant_a, x.variant_b, int(x.users), int(x.orders), int(x.orders_sq)) for x in r.itertuples()]
    assert got == [("control", "control", 1, 1, 1), ("treatment", "treatment", 1, 1, 1)]    # user 3 is only in exp_a


def test_run_query_rejects_missing_parameters():
    con = make_con(assignments=pd.DataFrame({"experiment_id": ["x"], "user_id": [1], "variant": ["control"],
                                             "assigned_at": pd.to_datetime(["2026-02-02"])}),
                   experiments=_exp_tables(),
                   orders=pd.DataFrame({"user_id": [1], "order_ts": pd.to_datetime(["2026-02-03"]), "order_value": [1.0],
                                        "order_id": [1]}))
    with pytest.raises(ValueError):
        run_query(con, "experiment_interaction", {"exp_a": "exp_a"})


def test_experiment_user_metrics_rows_come_out_sorted_by_user_id():
    """Regression test for the reproducibility bug: no ORDER BY meant unstable row order between runs."""
    ids = [9, 3, 7, 1, 5, 2, 8, 4, 6]
    users = pd.DataFrame({"user_id": ids, "signup_date": pd.to_datetime(["2025-01-01"] * 9).date, "city": "x",
                          "device": "ios", "channel": "paid", "is_new": False})
    a = pd.DataFrame({"assignment_id": range(1, 10), "experiment_id": "exp_a", "user_id": ids,
                      "variant": ["control", "treatment"] * 4 + ["control"],
                      "assigned_at": pd.to_datetime(["2026-02-02"] * 9)})
    orders = pd.DataFrame({"order_id": [1], "session_id": [1], "user_id": [3], "order_ts": [ts("2026-02-03 10:00")],
                           "n_items": [1], "gross_value": [10.0], "discount_pct": [0.0], "order_value": [10.0]})
    sessions = pd.DataFrame({"session_id": [1], "user_id": [3], "session_start": [ts("2026-01-20 10:00")],
                             "duration_sec": [5], "device": ["ios"]})
    con = make_con(users=users, assignments=a, orders=orders, sessions=sessions, experiments=_exp_tables())
    df = con.execute((SQL_DIR / "marts" / "experiment_user_metrics.sql").read_text(), {"exp_id": "exp_a"}).df()
    assert list(df.user_id) == sorted(ids)
    assert df.loc[df.user_id == 3, "orders_in"].iloc[0] == 1 and df.loc[df.user_id == 3, "sessions_pre"].iloc[0] == 1
