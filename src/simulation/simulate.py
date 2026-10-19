"""Simulate a grocery-delivery app: users, sessions, events, orders and A/B experiments.

Because WE write the rules, we know the true effect of every experiment. That lets
the rest of the project test whether the analysis finds the right answer.

Run:  python -m src.simulation.simulate            (writes Parquet files to data/warehouse/)
"""
from __future__ import annotations

import argparse
import dataclasses

import numpy as np
import pandas as pd

from src.config import EXPERIMENTS, WAREHOUSE_DIR, Experiment, SimConfig

CITIES = ["Mumbai", "Delhi", "Bengaluru", "Pune", "Hyderabad"]
CITY_P = [0.28, 0.24, 0.22, 0.14, 0.12]
DEVICES = ["android", "ios", "web"]
DEVICE_P = [0.58, 0.30, 0.12]
CHANNELS = ["organic", "paid", "referral"]
CHANNEL_P = [0.50, 0.35, 0.15]
CATEGORIES = ["fruits_veg", "dairy", "bakery", "snacks", "beverages", "staples", "household", "personal_care"]

# Mon..Sun multipliers: weekends are busier.
DOW_SESSIONS = np.array([0.90, 0.90, 0.92, 0.95, 1.05, 1.25, 1.20])
DOW_CONV = np.array([0.97, 0.97, 0.98, 1.00, 1.02, 1.08, 1.06])
# Hour-of-day profile: morning and evening peaks.
HOUR_P = np.array([1, 1, 1, 1, 1, 2, 3, 5, 6, 5, 4, 4, 5, 5, 4, 4, 5, 7, 9, 10, 9, 6, 3, 2], dtype=float)
HOUR_P /= HOUR_P.sum()

# Funnel base rates per session, calibrated so overall session conversion is ~5%
# (published app benchmarks quote roughly 4.5-6% for e-commerce apps; sources disagree).
P_CART, P_CHECKOUT, P_PAY = 0.22, 0.55, 0.42

# Observable segment effects on the purchase step.
DEVICE_CONV = {"android": 1.00, "ios": 1.15, "web": 0.60}
CHANNEL_CONV = {"organic": 1.00, "paid": 0.85, "referral": 1.15}
CITY_SPEND = {"Mumbai": 1.15, "Delhi": 1.10, "Bengaluru": 1.10, "Pune": 1.00, "Hyderabad": 0.95}

# Engagement segments (latent): 0 = casual, 1 = medium, 2 = loyal.
SEG_MEAN_LIFE_DAYS = np.array([3.0, 20.0, 150.0])     # mean days a user stays active (churn)
SEG_SESSION_FACTOR = np.array([0.6, 1.0, 1.5])        # loyal users open the app more
SEG_WEIGHTS = {"organic": [0.60, 0.30, 0.10], "paid": [0.68, 0.25, 0.07], "referral": [0.50, 0.35, 0.15]}


# --------------------------------------------------------------------------- tables
def make_products(rng: np.random.Generator, cfg: SimConfig) -> pd.DataFrame:
    price = np.round(np.exp(rng.normal(np.log(120), 0.6, cfg.n_products)), 0).clip(10, 1500)
    return pd.DataFrame({
        "product_id": np.arange(1, cfg.n_products + 1),
        "category": rng.choice(CATEGORIES, cfg.n_products),
        "price_inr": price,
    })


def make_users(rng: np.random.Generator, cfg: SimConfig) -> tuple[pd.DataFrame, dict]:
    """Observable user table + hidden latent traits (never exported)."""
    n = cfg.n_users
    existing = rng.random(n) < 0.70
    signup_day = np.where(existing, rng.integers(-120, 0, n), rng.integers(0, cfg.n_days, n))
    device = rng.choice(DEVICES, n, p=DEVICE_P)
    channel = rng.choice(CHANNELS, n, p=CHANNEL_P)
    city = rng.choice(CITIES, n, p=CITY_P)

    # Engagement segment. Existing users are SURVIVORS (they are still active at day 0), so
    # their segment odds are tilted toward loyal users: weight_s * P(alive after `age` days).
    base_w = np.array([SEG_WEIGHTS[c] for c in channel])
    age = np.where(existing, -signup_day, 0)
    w = base_w * np.exp(-age[:, None] / SEG_MEAN_LIFE_DAYS[None, :])
    w = w / w.sum(axis=1, keepdims=True)
    seg = (rng.random(n)[:, None] > np.cumsum(w, axis=1)).sum(axis=1).clip(0, 2)

    # Churn: exponential lifetime (memoryless, so existing users' remaining life uses the same mean).
    life = np.ceil(rng.exponential(SEG_MEAN_LIFE_DAYS[seg])).astype(int)
    churn_day = np.where(existing, life - 1, signup_day + life - 1)

    z = rng.normal(size=n)
    ch_sess = np.where(channel == "referral", 1.10, np.where(channel == "paid", 0.95, 1.0))
    lam = np.clip(np.exp(np.log(0.22) + 0.6 * z) * SEG_SESSION_FACTOR[seg] * ch_sess, 0.01, 3.0)
    dev_conv = np.vectorize(DEVICE_CONV.get)(device)
    chn_conv = np.vectorize(CHANNEL_CONV.get)(channel)
    f_pay = np.exp(0.25 * (0.5 * z + np.sqrt(0.75) * rng.normal(size=n))) * dev_conv * chn_conv
    city_f = np.vectorize(CITY_SPEND.get)(city)
    spend = np.exp(0.40 * (0.5 * z + np.sqrt(0.75) * rng.normal(size=n))) * city_f

    start = pd.Timestamp(cfg.start_date)
    users = pd.DataFrame({
        "user_id": np.arange(1, n + 1),
        "signup_date": (start + pd.to_timedelta(signup_day, unit="D")).date,
        "city": city, "device": device, "channel": channel,
        "is_new": signup_day >= 0,
    })
    latent = {"signup_day": signup_day, "churn_day": churn_day, "seg": seg, "lam": lam,
              "f_pay": f_pay, "spend": spend, "device": device}
    return users, latent


def make_assignments(rng: np.random.Generator, cfg: SimConfig, signup_day: np.ndarray) -> dict:
    """Randomise users into control (0) / treatment (1) for every experiment.

    Everyone who has signed up by the experiment's end day is enrolled, from
    max(signup day, experiment start day). Experiments are randomised independently.
    """
    arms = {}
    for e in EXPERIMENTS:
        eligible = signup_day <= e.end_day
        arm = np.where(eligible, (rng.random(cfg.n_users) < 0.5).astype(int), -1)
        enroll_day = np.maximum(signup_day, e.start_day)
        arms[e.experiment_id] = {"arm": arm, "enroll_day": enroll_day}
    return arms


# ----------------------------------------------------------------------- effect model
def lift_for_sessions(e: Experiment, arm, enroll_day, is_new, seg, prof, u, day) -> np.ndarray:
    """Relative lift on purchase probability for each session (0 if not treated / outside window)."""
    in_win = (arm[u] == 1) & (day >= enroll_day[u]) & (day <= e.end_day)
    if e.kind == "constant":
        lift = np.full(len(u), e.lift)
    elif e.kind == "novelty":
        since = np.maximum(day - enroll_day[u], 0)
        lift = e.long_run_lift + (e.lift - e.long_run_lift) * 0.5 ** (since / e.half_life_days)
    elif e.kind == "new_users_only":
        lift = np.where(is_new[u], e.lift, 0.0)
    elif e.kind == "by_engagement":
        lift = np.asarray(e.seg_lifts)[seg[u]]
    elif e.kind == "by_profile":
        ch_idx, dev_idx = prof
        lift = np.asarray(e.channel_lifts)[ch_idx[u]] + np.asarray(e.device_lifts)[dev_idx[u]]
    else:  # none
        lift = np.zeros(len(u))
    return np.where(in_win, lift, 0.0)


# ----------------------------------------------------------------------- main engine
def simulate(cfg: SimConfig = SimConfig()) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed)
    products = make_products(rng, cfg)
    users, latent = make_users(rng, cfg)
    arms = make_assignments(rng, cfg, latent["signup_day"])

    prices = products["price_inr"].to_numpy()
    pop = rng.dirichlet(np.full(cfg.n_products, 0.4))        # a few popular products
    start = np.datetime64(pd.Timestamp(cfg.start_date), "s")
    days = np.arange(cfg.n_days)
    dow = days % 7

    is_new = users["is_new"].to_numpy()
    prof = (users["channel"].map({c: i for i, c in enumerate(CHANNELS)}).to_numpy(),
            users["device"].map({d: i for i, d in enumerate(DEVICES)}).to_numpy())
    sessions_parts, events_parts, orders_parts = [], [], []
    sid0 = 0

    for lo in range(0, cfg.n_users, cfg.chunk_users):
        hi = min(lo + cfg.chunk_users, cfg.n_users)
        sl = slice(lo, hi)
        m = hi - lo
        sd = latent["signup_day"][sl]

        # ---- how many sessions each user has each day
        rate = latent["lam"][sl][:, None] * DOW_SESSIONS[dow][None, :]
        cd = latent["churn_day"][sl]
        rate = rate * ((days[None, :] >= sd[:, None]) & (days[None, :] <= cd[:, None]))
        counts = rng.poisson(rate)
        newbies = np.where(sd >= 0)[0]                         # new users always open the app on signup day
        counts[newbies, sd[newbies]] = np.maximum(counts[newbies, sd[newbies]], 1)

        rows, cols = np.nonzero(counts)
        reps = counts[rows, cols]
        u_loc = np.repeat(rows, reps)
        day = np.repeat(cols, reps)
        ns = len(u_loc)
        u_glob = u_loc + lo                                    # 0-based global user index

        # ---- session timestamps
        hour = rng.choice(24, ns, p=HOUR_P)
        secs = day.astype("int64") * 86400 + hour * 3600 + rng.integers(0, 3600, ns)
        s_start = start + secs.astype("timedelta64[s]")
        session_id = sid0 + 1 + np.arange(ns)
        sid0 += ns

        # ---- conversion probability with experiment effects
        p_pay = np.clip(P_PAY * latent["f_pay"][u_glob], 0.05, 0.95)
        mult = np.ones(ns)
        for e in EXPERIMENTS:
            if e.target != "conversion":
                continue
            a = arms[e.experiment_id]
            mult *= 1.0 + lift_for_sessions(e, a["arm"], a["enroll_day"], is_new, latent["seg"], prof, u_glob, day)
        p_pay = np.clip(p_pay * mult, 0.0, 0.98)
        p_cart = np.clip(P_CART * DOW_CONV[dow[day]], 0, 0.95)

        # ---- funnel draws
        k_views = 1 + rng.poisson(2.0, ns)
        cart = rng.random(ns) < p_cart
        chk = cart & (rng.random(ns) < P_CHECKOUT)
        buy = chk & (rng.random(ns) < p_pay)

        sessions_parts.append(pd.DataFrame({
            "session_id": session_id,
            "user_id": u_glob + 1,
            "session_start": s_start,
            "duration_sec": np.clip(np.exp(rng.normal(5.0, 0.7, ns)), 10, 3600).astype(int),
            "device": latent["device"][u_glob],
        }))

        # ---- events
        total_views = int(k_views.sum())
        v_idx = np.repeat(np.arange(ns), k_views)
        within = np.arange(total_views) - np.repeat(np.cumsum(k_views) - k_views, k_views)
        v_off = (within + 1) * 20 + rng.integers(0, 15, total_views)
        v_prod = rng.choice(cfg.n_products, total_views, p=pop) + 1

        def step(mask, offset_extra, etype):
            idx = np.where(mask)[0]
            return idx, k_views[idx] * 20 + offset_extra + rng.integers(0, 10, len(idx)), etype

        c_idx, c_off, _ = step(cart, 30, "add_to_cart")
        k_idx, k_off, _ = step(chk, 60, "checkout_start")
        b_idx, b_off, _ = step(buy, 120, "purchase")
        c_prod = rng.choice(cfg.n_products, len(c_idx), p=pop) + 1

        idx_all = np.concatenate([v_idx, c_idx, k_idx, b_idx])
        off_all = np.concatenate([v_off, c_off, k_off, b_off])
        type_all = np.concatenate([
            np.full(len(v_idx), "product_view"), np.full(len(c_idx), "add_to_cart"),
            np.full(len(k_idx), "checkout_start"), np.full(len(b_idx), "purchase")])
        prod_all = np.concatenate([v_prod, c_prod, np.zeros(len(k_idx), int), np.zeros(len(b_idx), int)])
        events_parts.append(pd.DataFrame({
            "session_id": session_id[idx_all],
            "user_id": u_glob[idx_all] + 1,
            "event_type": type_all,
            "event_ts": s_start[idx_all] + off_all.astype("timedelta64[s]"),
            "product_id": pd.Series(prod_all, dtype="Int64").mask(prod_all == 0),
        }))

        # ---- orders (one per purchase session)
        no = len(b_idx)
        n_items = np.clip(1 + rng.poisson(1.2, no), 1, 8)
        items = rng.choice(cfg.n_products, (no, 8), p=pop)
        mask = np.arange(8)[None, :] < n_items[:, None]
        gross = (prices[items] * mask).sum(axis=1) * latent["spend"][u_glob[b_idx]]
        for e in EXPERIMENTS:
            if e.target != "order_value":
                continue
            a = arms[e.experiment_id]
            gross = gross * (1.0 + lift_for_sessions(e, a["arm"], a["enroll_day"], is_new,
                                                      latent["seg"], prof, u_glob[b_idx], day[b_idx]))
        disc = np.where(rng.random(no) < 0.12, rng.uniform(0.05, 0.20, no), 0.0)
        orders_parts.append(pd.DataFrame({
            "session_id": session_id[b_idx],
            "user_id": u_glob[b_idx] + 1,
            "order_ts": s_start[b_idx] + b_off.astype("timedelta64[s]"),
            "n_items": n_items,
            "gross_value": np.round(gross, 2),
            "discount_pct": np.round(disc, 3),
            "order_value": np.round(gross * (1 - disc), 2),
        }))

    sessions = pd.concat(sessions_parts, ignore_index=True)
    events = pd.concat(events_parts, ignore_index=True)
    orders = pd.concat(orders_parts, ignore_index=True)

    # ---- assignments table (before bugs)
    start_ts = pd.Timestamp(cfg.start_date)
    assign_rows, truth_rows, exp_rows = [], [], []
    session_day = ((sessions["session_start"] - start_ts).dt.days).to_numpy()
    for e in EXPERIMENTS:
        a = arms[e.experiment_id]
        enrolled = (a["arm"] >= 0) & (a["enroll_day"] <= e.end_day)
        uid = np.where(enrolled)[0]
        keep = np.ones(len(uid), dtype=bool)

        if cfg.inject_bugs and e.experiment_id == "exp_006":
            # Bug: bot-filtering removes low-activity TREATMENT users (never control).
            win = (session_day >= e.start_day) & (session_day <= e.end_day)
            n_sess = np.bincount(sessions["user_id"].to_numpy()[win] - 1, minlength=cfg.n_users)
            cutoff = np.quantile(n_sess[uid], 0.30)
            low = n_sess[uid] <= cutoff
            treated = a["arm"][uid] == 1
            keep = ~(treated & low & (rng.random(len(uid)) < 0.35))

        uid_k = uid[keep]
        assign_rows.append(pd.DataFrame({
            "experiment_id": e.experiment_id,
            "user_id": uid_k + 1,
            "variant": np.where(a["arm"][uid_k] == 1, "treatment", "control"),
            "assigned_at": start_ts + pd.to_timedelta(a["enroll_day"][uid_k], unit="D")
                           + pd.to_timedelta(rng.integers(0, 86400, len(uid_k)), unit="s"),
        }))
        exp_rows.append({
            "experiment_id": e.experiment_id, "name": e.name,
            "start_date": (start_ts + pd.Timedelta(days=e.start_day)).date(),
            "end_date": (start_ts + pd.Timedelta(days=e.end_day)).date(),
            "primary_metric": "revenue_per_user" if e.target == "order_value" else "orders_per_user",
            "exclusive_group": e.exclusive_group,
        })
        truth_rows.append({
            "experiment_id": e.experiment_id, "effect_kind": e.kind,
            "nominal_relative_lift": e.lift, "long_run_relative_lift": e.long_run_lift,
            "seg_lifts": ",".join(map(str, e.seg_lifts)),
            "planted_issue": e.planted_issue if cfg.inject_bugs else "", "notes": e.notes,
        })

    assignments = pd.concat(assign_rows, ignore_index=True)
    assignments.insert(0, "assignment_id", np.arange(1, len(assignments) + 1))

    # ---- planted data-quality bugs on events
    if cfg.inject_bugs:
        dup = events.sample(frac=0.006, random_state=cfg.seed)          # exact duplicate events
        events = pd.concat([events, dup], ignore_index=True)
    events = events.sort_values(["event_ts", "session_id"], kind="stable").reset_index(drop=True)
    events.insert(0, "event_id", np.arange(1, len(events) + 1))
    delay = rng.integers(1, 30, len(events)).astype("timedelta64[s]")
    if cfg.inject_bugs:                                                  # 1.5% arrive 1-3 days late
        late = rng.random(len(events)) < 0.015
        delay = np.where(late, rng.integers(86400, 3 * 86400, len(events)).astype("timedelta64[s]"), delay)
    events["ingested_at"] = events["event_ts"].to_numpy() + delay

    orders = orders.sort_values("order_ts").reset_index(drop=True)
    orders.insert(0, "order_id", np.arange(1, len(orders) + 1))

    return {
        "users": users, "products": products, "sessions": sessions, "events": events,
        "orders": orders, "experiments": pd.DataFrame(exp_rows), "assignments": assignments,
        "answer_key": pd.DataFrame(truth_rows),
        # Hidden per-user truth, for DIAGNOSTICS ONLY (never used as a model feature).
        "user_truth": pd.DataFrame({
            "user_id": users["user_id"].to_numpy(),
            "engagement_segment": np.array(["casual", "medium", "loyal"])[latent["seg"]],
        }),
    }


def write_parquet(tables: dict[str, pd.DataFrame], out_dir=WAREHOUSE_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_parquet(out_dir / f"{name}.parquet", index=False, compression="zstd")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate the simulated warehouse.")
    ap.add_argument("--n-users", type=int, default=SimConfig.n_users)
    ap.add_argument("--seed", type=int, default=SimConfig.seed)
    ap.add_argument("--no-bugs", action="store_true", help="skip planted data-quality bugs")
    args = ap.parse_args()
    cfg = dataclasses.replace(SimConfig(), n_users=args.n_users, seed=args.seed, inject_bugs=not args.no_bugs)
    tables = simulate(cfg)
    write_parquet(tables)
    for k, v in tables.items():
        print(f"{k:12s} {len(v):>10,d} rows")
    total = sum(p.stat().st_size for p in WAREHOUSE_DIR.glob("*.parquet")) / 1e6
    print(f"\nParquet size on disk: {total:.1f} MB")
