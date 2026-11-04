"""Three studies on the simulated warehouse.

  1. power     : how many users do we need? Is our own power maths right? (A/A + injected lift)
  2. peeking   : how badly does 'checking every day and stopping when p<0.05' inflate false wins?
  3. multiple  : what happens when we run many experiments and how do corrections change verdicts?

Run:  python -m src.experiments.studies all          (needs data/experimentation.duckdb)
Outputs: artifacts/*.csv and artifacts/*.png
"""
from __future__ import annotations

import argparse

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.config import DB_PATH, ROOT, SQL_DIR
from src.experiments.stats_utils import (analytic_power, benjamini_hochberg, bonferroni, cuped_adjust, holm,
                          mde_rel, obrien_fleming_bounds, required_n_per_arm, welch_t)

ART = ROOT / "artifacts"
RNG_SEED = 2026


def _con():
    return duckdb.connect(str(DB_PATH), read_only=True)


def _user_metrics(con, exp_id: str) -> pd.DataFrame:
    return con.execute((SQL_DIR / "marts" / "experiment_user_metrics.sql").read_text(), {"exp_id": exp_id}).df()


# ============================================================================ 1. POWER
def study_power(sims: int = 400) -> pd.DataFrame:
    ART.mkdir(exist_ok=True)
    con = _con()
    rng = np.random.default_rng(RNG_SEED)

    # ---- (a) how many users did exp_002 (true +1.5%) need?
    d2 = _user_metrics(con, "exp_002")
    ctrl = d2[d2.variant == "control"]
    mean, sd = ctrl.orders_in.mean(), ctrl.orders_in.std(ddof=1)
    rho = np.corrcoef(d2.orders_in, d2.orders_pre)[0, 1]
    n_have = int(min((d2.variant == "control").sum(), (d2.variant == "treatment").sum()))
    rows = [{
        "question": "exp_002 true lift +1.5%",
        "users_per_arm_available": n_have,
        "needed_no_cuped": round(required_n_per_arm(0.015, mean, sd)),
        "needed_with_cuped": round(required_n_per_arm(0.015, mean, sd, rho=rho)),
        "mde_available_no_cuped": round(mde_rel(n_have, mean, sd), 4),
        "mde_available_with_cuped": round(mde_rel(n_have, mean, sd, rho=rho), 4),
        "corr_outcome_pre": round(rho, 3),
    }]
    pd.DataFrame(rows).to_csv(ART / "power_exp002.csv", index=False)

    # ---- (b) validate our power maths by simulation on a real null experiment (exp_003)
    d3 = _user_metrics(con, "exp_003")
    y_all, x_all = d3.orders_in.to_numpy(float), d3.orders_pre.to_numpy(float)
    m3, s3 = y_all.mean(), y_all.std(ddof=1)
    rho3 = np.corrcoef(y_all, x_all)[0, 1]

    def one_run(n, lift, use_cuped):
        # bootstrap WITH replacement: sampling 2n of ~95k users without replacement would
        # understate variance (finite-population effect) and give too few false positives.
        idx = rng.choice(len(y_all), 2 * n, replace=True)
        y, x = y_all[idx].copy(), x_all[idx]
        t_mask = np.zeros(2 * n, bool)
        t_mask[rng.choice(2 * n, n, replace=False)] = True
        if lift > 0:                                   # inject a known lift: each order duplicates w.p. lift
            y[t_mask] = y[t_mask] + rng.binomial(y[t_mask].astype(int), lift)
        if use_cuped:
            y = cuped_adjust(y, x)
        return welch_t(y[t_mask], y[~t_mask])[1] < 0.05

    out = []
    for n in (10_000, 25_000, 45_000):
        for lift in (0.0, 0.015, 0.03, 0.05, 0.08):
            for cup in (False, True):
                emp = np.mean([one_run(n, lift, cup) for _ in range(sims)])
                ana = analytic_power(lift, n, m3, s3, rho=rho3 if cup else 0.0)
                out.append({"n_per_arm": n, "true_lift": lift, "cuped": cup,
                            "empirical_power": round(emp, 3), "analytic_power": round(ana, 3)})
    res = pd.DataFrame(out)
    res.to_csv(ART / "power_validation.csv", index=False)

    # ---- chart
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for cup, ls in ((False, "-"), (True, "--")):
        sub = res[(res.n_per_arm == 45_000) & (res.cuped == cup)]
        ax.plot(sub.true_lift * 100, sub.empirical_power, ls, marker="o",
                label=f"empirical, {'CUPED' if cup else 'plain'}")
        ax.plot(sub.true_lift * 100, sub.analytic_power, ":", color="grey")
    ax.axhline(0.8, color="red", lw=0.8)
    ax.set_xlabel("true relative lift (%)"); ax.set_ylabel("power")
    ax.set_title("Power at 45,000 users/arm (grey dotted = analytic formula)")
    ax.legend(); fig.tight_layout(); fig.savefig(ART / "power_curve.png", dpi=130); plt.close(fig)
    return res


# =========================================================================== 2. PEEKING
def _daily_cumulative_orders(con, exp_id: str) -> np.ndarray:
    """(users x days) cumulative order counts inside the experiment window, from assignment onward."""
    df = con.execute("""
        SELECT a.user_id,
               DATE_DIFF('day', e.start_date, CAST(o.order_ts AS DATE)) AS day_idx,
               COUNT(*) AS n
        FROM assignments a
        JOIN experiments e ON e.experiment_id = a.experiment_id
        JOIN orders o ON o.user_id = a.user_id
         AND o.order_ts >= a.assigned_at AND o.order_ts < e.end_date + INTERVAL 1 DAY
        WHERE a.experiment_id = $e
        GROUP BY 1, 2""", {"e": exp_id}).df()
    users = con.execute("SELECT user_id FROM assignments WHERE experiment_id = $e ORDER BY user_id",
                        {"e": exp_id}).df().user_id.to_numpy()
    n_days = int(con.execute("SELECT DATE_DIFF('day', start_date, end_date) + 1 FROM experiments WHERE experiment_id=$e",
                             {"e": exp_id}).fetchone()[0])
    mat = np.zeros((len(users), n_days))
    row = np.searchsorted(users, df.user_id.to_numpy())
    mat[row, df.day_idx.to_numpy()] = df.n.to_numpy()
    return np.cumsum(mat, axis=1)


def _permutation_t_paths(cum: np.ndarray, reps: int, rng, chunk: int = 100) -> np.ndarray:
    """A/A tests: randomly relabel users, return |t| at every day (reps x days)."""
    n, days = cum.shape
    sq = cum**2
    tot, tot_sq = cum.sum(0), sq.sum(0)
    out = np.empty((reps, days))
    for s in range(0, reps, chunk):
        r = min(chunk, reps - s)
        W = (rng.random((r, n)) < 0.5).astype(np.float64)
        nt = W.sum(1)[:, None]; nc = n - nt
        st, sqt = W @ cum, W @ sq
        sc, sqc = tot - st, tot_sq - sqt
        mt, mc = st / nt, sc / nc
        vt = (sqt - nt * mt**2) / (nt - 1); vc = (sqc - nc * mc**2) / (nc - 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            t = (mt - mc) / np.sqrt(vt / nt + vc / nc)
        out[s:s + r] = np.abs(np.nan_to_num(t))
    return out


def study_peeking(reps: int = 1000) -> pd.DataFrame:
    ART.mkdir(exist_ok=True)
    con = _con()
    rng = np.random.default_rng(RNG_SEED)
    cum = _daily_cumulative_orders(con, "exp_003")           # true null experiment, 28 days
    days = cum.shape[1]
    t_abs = _permutation_t_paths(cum, reps, rng)
    z05 = stats.norm.ppf(0.975)

    weekly = [6, 13, 20, 27]                                  # 0-based day indices of weekly looks
    obf = obrien_fleming_bounds(len(weekly))
    strategies = {
        "1 look at the end (correct)": (t_abs[:, -1] > z05),
        "peek weekly (4 looks), naive 5%": (t_abs[:, weekly] > z05).any(axis=1),
        "peek every day (28 looks), naive 5%": (t_abs > z05).any(axis=1),
        "peek every day, Bonferroni": (t_abs > stats.norm.ppf(1 - 0.05 / (2 * days))).any(axis=1),
        "peek weekly, O'Brien-Fleming bounds": (t_abs[:, weekly] > obf).any(axis=1),
    }
    res = pd.DataFrame({"strategy": list(strategies),
                        "false_positive_rate": [round(float(v.mean()), 3) for v in strategies.values()]})
    res.to_csv(ART / "peeking_false_positive_rate.csv", index=False)

    # false-positive rate vs number of equally spaced looks (naive)
    ks, rates = [1, 2, 4, 7, 14, 28], []
    for k in ks:
        look_idx = np.unique(np.round(np.linspace(days / k, days, k)).astype(int) - 1)
        rates.append(float((t_abs[:, look_idx] > z05).any(axis=1).mean()))
    pd.DataFrame({"n_looks": ks, "false_positive_rate": rates}).to_csv(ART / "peeking_vs_looks.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.plot(ks, np.array(rates) * 100, marker="o")
    ax.axhline(5, color="red", lw=0.8, ls="--")
    ax.set_xlabel("number of times you look at the result"); ax.set_ylabel("A/A tests declared 'significant' (%)")
    ax.set_title("Peeking inflates false wins (true effect = 0, nominal 5%)")
    fig.tight_layout(); fig.savefig(ART / "peeking_inflation.png", dpi=130); plt.close(fig)
    return res


# ================================================================== 3. MULTIPLE TESTING
def study_multiple(n_null_tests: int = 5000, n_groups: int = 2000) -> dict:
    ART.mkdir(exist_ok=True)
    con = _con()
    rng = np.random.default_rng(RNG_SEED)
    meta = con.execute("SELECT experiment_id, name, primary_metric FROM experiments ORDER BY 1").df()

    rows = []
    for _, r in meta.iterrows():
        d = _user_metrics(con, r.experiment_id)
        metric = "revenue" if r.primary_metric == "revenue_per_user" else "orders"
        y, x = d[f"{metric}_in"].to_numpy(float), d[f"{metric}_pre"].to_numpy(float)
        is_t = (d.variant == "treatment").to_numpy()
        ya = cuped_adjust(y, x)
        lift = ya[is_t].mean() / ya[~is_t].mean() - 1
        p = welch_t(ya[is_t], ya[~is_t])[1]
        srm_p = stats.chisquare([is_t.sum(), (~is_t).sum()]).pvalue
        rows.append({"experiment": r.experiment_id, "name": r["name"], "cuped_lift": lift, "p_value": p, "SRM_p": srm_p})
    t = pd.DataFrame(rows)
    t["srm_ok"] = t.SRM_p >= 0.001
    ok = t[t.srm_ok].copy()                                    # never test an experiment with broken randomisation
    p = ok.p_value.to_numpy()
    ok["raw_p<0.05"] = p < 0.05
    ok["bonferroni"] = bonferroni(p)
    ok["holm"] = holm(p)
    ok["benjamini_hochberg"] = benjamini_hochberg(p)
    t = t.merge(ok[["experiment", "raw_p<0.05", "bonferroni", "holm", "benjamini_hochberg"]], on="experiment", how="left")
    t.to_csv(ART / "multiple_testing_verdicts.csv", index=False)

    # family-wise error: run m null tests, how often is at least one 'significant'?
    d3 = _user_metrics(con, "exp_003")
    y3 = d3.orders_in.to_numpy(float)
    p_null = np.empty(n_null_tests)
    for i in range(len(p_null)):
        idx = rng.permutation(len(y3)) < len(y3) // 2
        p_null[i] = welch_t(y3[idx], y3[~idx])[1]
    fam = []
    for m in (1, 5, 10, 20):
        groups = rng.choice(p_null, size=(n_groups, m))
        fam.append({"n_null_experiments": m,
                    "P(>=1 false win), no correction": round(float((groups < 0.05).any(1).mean()), 3),
                    "theory 1-0.95^m": round(1 - 0.95**m, 3),
                    "with Holm": round(float(np.mean([holm(g).any() for g in groups[: n_groups // 2]])), 3)})
    fam = pd.DataFrame(fam)
    fam.to_csv(ART / "family_wise_error.csv", index=False)
    return {"verdicts": t, "fwer": fam}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("study", choices=["power", "peeking", "multiple", "all"])
    ap.add_argument("--sims", type=int, default=400, help="simulations per power cell")
    ap.add_argument("--reps", type=int, default=1000, help="A/A permutations")
    a = ap.parse_args()
    pd.set_option("display.width", 220, "display.max_columns", 30)
    if a.study in ("power", "all"):
        print("\n=== POWER ===")
        pd.set_option("display.max_rows", 100)
        r = study_power(a.sims)
        print(pd.read_csv(ART / "power_exp002.csv").T.to_string(header=False))
        print(); print(r[r.n_per_arm == 45_000].to_string(index=False))
    if a.study in ("peeking", "all"):
        print("\n=== PEEKING (true effect = 0) ===")
        print(study_peeking(a.reps).to_string(index=False))
        print(pd.read_csv(ART / "peeking_vs_looks.csv").to_string(index=False))
    if a.study in ("multiple", "all"):
        print("\n=== MULTIPLE TESTING ===")
        o = study_multiple()
        print(o["verdicts"].to_string(index=False)); print(); print(o["fwer"].to_string(index=False))
