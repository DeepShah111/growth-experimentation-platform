"""Multi-seed uplift evaluation with an honest (hold-out) profit cut-off.

Two problems with a single uplift run:
  1. The result depends on the random cross-fitting folds and model seeds, so one run is one draw. We repeat with
     several seeds and report the spread, plus how often each learner is best.
  2. The profit view picks the best "treat the top X%" cut-off on the same data it then scores. Here the cut-off is
     chosen on one random half of the users and scored on the OTHER half; the in-sample number is shown next to it
     so the optimism is visible.

The seed spread measures algorithmic noise only (folds, model seeds, split). Sampling noise, from having only 100k
users, is what the bootstrap intervals in src.targeting.uplift show. Both matter.

Run:  python -m src.targeting.multiseed --exp exp_008 --seeds 10
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config import ROOT
from src.targeting.uplift import COST_PER_TREATED, VALUE_PER_ORDER, cross_fit, ipw_effect, load, qini_area

LEARNERS = ["S-learner", "T-learner", "X-learner"]


def holdout_profit(score: np.ndarray, y: np.ndarray, t: np.ndarray, rng: np.random.Generator,
                   value: float = VALUE_PER_ORDER, cost: float = COST_PER_TREATED) -> dict:
    """Pick the best cut-off on half A, score it on half B. All profits are per 1,000 users."""
    n = len(score)
    perm = rng.permutation(n)
    a, b = perm[: n // 2], perm[n // 2:]
    tau_a, tau_b = ipw_effect(y[a], t[a]), ipw_effect(y[b], t[b])

    order_a = np.argsort(-score[a], kind="stable")
    k = np.arange(1, len(a) + 1)
    profit_a = value * np.cumsum(tau_a[order_a]) - cost * k
    k_star = int(np.argmax(profit_a)) + 1
    threshold = score[a][order_a][k_star - 1]                       # treat users scoring at or above this

    treated_b = score[b] >= threshold
    profit_b = value * tau_b[treated_b].sum() - cost * treated_b.sum()
    per1000 = lambda x, m: float(x / m * 1000)                      # noqa: E731
    return {"share_treated_chosen_on_A": k_star / len(a),
            "share_treated_on_B": float(treated_b.mean()),
            "profit_in_sample": per1000(profit_a[k_star - 1], len(a)),
            "profit_holdout": per1000(profit_b, len(b)),
            "treat_everyone_holdout": per1000(value * tau_b.sum() - cost * len(b), len(b))}


def run_seeds(d: pd.DataFrame, t: np.ndarray, seeds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (one row per seed x model, the scores of the FIRST seed for the paired bootstrap)."""
    y = d.orders_in.to_numpy(float)
    tau = ipw_effect(y, t)
    rows, first = [], None
    for sd in seeds:
        scores = cross_fit(d, t, seed=sd)
        first = scores if first is None else first
        for m in [*LEARNERS, "random"]:
            s = scores[m].to_numpy()
            rows.append({"seed": sd, "model": m, "qini": qini_area(s, tau),
                         **holdout_profit(s, y, t, np.random.default_rng(10_000 + sd))})
    return pd.DataFrame(rows), first


def paired_bootstrap(scores: pd.DataFrame, tau: np.ndarray, reps: int = 300, seed: int = 7) -> pd.DataFrame:
    """Sampling uncertainty of the DIFFERENCE between learners: the same resampled users score both models."""
    rng = np.random.default_rng(seed)
    n = len(tau)
    pairs = [("S-learner", "X-learner"), ("S-learner", "T-learner"), ("X-learner", "T-learner")]
    diffs = {p: [] for p in pairs}
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        q = {m: qini_area(scores[m].to_numpy()[idx], tau[idx]) for m in LEARNERS}
        for a, b in pairs:
            diffs[(a, b)].append(q[a] - q[b])
    rows = []
    for (a, b), v in diffs.items():
        lo, hi = np.percentile(v, [2.5, 97.5])
        rows.append({"difference": f"{a} minus {b}", "mean": float(np.mean(v)), "ci_low": float(lo), "ci_high": float(hi),
                     "interval_excludes_zero": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows)


def summarise(res: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    best = res[res.model.isin(LEARNERS)].loc[lambda x: x.groupby("seed").qini.idxmax()].model.value_counts()
    g = res.groupby("model")
    summ = pd.DataFrame({
        "qini_mean": g.qini.mean(), "qini_sd": g.qini.std(), "qini_min": g.qini.min(), "qini_max": g.qini.max(),
        "best_learner_in_n_seeds": [int(best.get(m, 0)) for m in g.qini.mean().index],
        "profit_in_sample_per_1000": g.profit_in_sample.mean(),
        "profit_holdout_per_1000": g.profit_holdout.mean(), "profit_holdout_sd": g.profit_holdout.std(),
        "treat_everyone_per_1000": g.treat_everyone_holdout.mean(),
        "share_treated": g.share_treated_on_B.mean(),
    })
    wide = res.pivot(index="seed", columns="model", values="qini")
    pair = {"S_minus_X_mean": float((wide["S-learner"] - wide["X-learner"]).mean()),
            "S_minus_X_sd": float((wide["S-learner"] - wide["X-learner"]).std()),
            "S_beats_X_in_n_seeds": int((wide["S-learner"] > wide["X-learner"]).sum()), "n_seeds": len(wide)}
    return summ.sort_values("qini_mean", ascending=False), pair


def main(exp_id: str = "exp_008", n_seeds: int = 10) -> None:
    d, t = load(exp_id)
    seeds = list(range(1, n_seeds + 1))
    print(f"{exp_id}: {len(d):,} users, {n_seeds} seeds (folds, model seeds and the profit split all change per seed)")
    res, first = run_seeds(d, t, seeds)
    summ, pair = summarise(res)
    art = ROOT / "artifacts"; art.mkdir(exist_ok=True)
    res.to_csv(art / f"uplift_multiseed_runs_{exp_id}.csv", index=False)
    summ.to_csv(art / f"uplift_multiseed_summary_{exp_id}.csv")
    pd.set_option("display.width", 220, "display.max_columns", 20, "display.float_format", lambda v: f"{v:,.2f}")
    print("\nQini = extra orders per 1,000 users vs random targeting. Profit in INR per 1,000 users (ASSUMED INR "
          f"{VALUE_PER_ORDER:.0f}/order, INR {COST_PER_TREATED:.0f}/treated user).")
    print(summ.to_string())
    print(f"\nS minus X: {pair['S_minus_X_mean']:+.2f} +/- {pair['S_minus_X_sd']:.2f} (sd across seeds); "
          f"S beat X in {pair['S_beats_X_in_n_seeds']} of {pair['n_seeds']} seeds.")
    pb = paired_bootstrap(first, ipw_effect(d.orders_in.to_numpy(float), t))
    pb.to_csv(art / f"uplift_paired_bootstrap_{exp_id}.csv", index=False)
    print(f"\nSampling uncertainty of the differences (paired bootstrap, 300 resamples, scores of seed {seeds[0]}):")
    print(pb.to_string(index=False))
    print("In-sample profit picks the cut-off and scores on the same users; hold-out picks it on one half and scores the other.")
    print("Seed spread is algorithmic noise only; sampling noise (100k users) is in the bootstrap CIs of src.targeting.uplift.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="exp_008", choices=["exp_008", "exp_009"])
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args(); main(a.exp, a.seeds)
