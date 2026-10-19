"""Does the platform's statistics deserve trust?  Repeat the whole experiment in many fresh simulated worlds.

For the experiments whose true effect is constant and known (exp_001, 002, 003, 007) we re-simulate the app with a new
random seed, re-run the same analysis (naive and CUPED) and ask, over many replications:
  bias        is the average estimate equal to the built-in truth?
  SE ratio    is the reported standard error equal to the real spread of the estimates?  (1.0 is honest; below 1 means
              the intervals are too narrow)
  coverage    how often does the 95% interval contain the truth?  (should be about 95%)
  rejection   how often is p < 0.05?  (power when the effect is real; false-positive rate for the null experiment)
  CUPED gain  how much narrower is the CUPED interval than the naive one?

"Truth" is the lift written into the simulator for constant-effect experiments (relative lift on purchase probability,
or on basket value for exp_007). If a bias shows up, the cause could be the estimator OR the assumption that the
nominal lift equals the realised lift in orders per user; the study flags it, it does not decide which.

Run:  python -m src.experiments.validation --seeds 60 --n-users 20000
"""
from __future__ import annotations

import argparse
import dataclasses

import duckdb
import numpy as np
import pandas as pd

from src.config import EXPERIMENTS, ROOT, SQL_DIR, SimConfig
from src.experiments.first_look import analyse
from src.simulation.simulate import simulate

CONSTANT = [e for e in EXPERIMENTS if e.kind in ("constant", "none")]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion (honest even with few replications)."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(centre - half), float(centre + half)


def one_replicate(seed: int, n_users: int) -> list[dict]:
    cfg = dataclasses.replace(SimConfig(), n_users=n_users, chunk_users=min(n_users, 10_000), seed=seed)
    tables = simulate(cfg)
    con = duckdb.connect(":memory:")
    for name in ("users", "sessions", "orders", "experiments", "assignments"):
        con.register(f"_{name}", tables[name])
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM _{name}")
    mart = (SQL_DIR / "marts" / "experiment_user_metrics.sql").read_text()
    rows = []
    for e in CONSTANT:
        df = con.execute(mart, {"exp_id": e.experiment_id}).df()
        metric = "revenue" if e.target == "order_value" else "orders"
        r = analyse(df, f"{metric}_in", f"{metric}_pre")
        for est in ("naive", "cuped"):
            rows.append({"seed": seed, "experiment": e.experiment_id, "estimator": est, "truth": e.lift if e.kind == "constant" else 0.0,
                         "lift": r[f"{est}_lift"], "half": r[f"{est}_CI±"], "p": r[f"{est}_p"]})
    return rows


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (exp, est), g in runs.groupby(["experiment", "estimator"]):
        n = len(g)
        truth = float(g.truth.iloc[0])
        covered = int(((g.lift - g.half <= truth) & (truth <= g.lift + g.half)).sum())
        lo, hi = wilson(covered, n)
        sd = float(g.lift.std(ddof=1))
        out.append({"experiment": exp, "estimator": est, "reps": n, "truth": truth, "mean_estimate": float(g.lift.mean()),
                    "bias": float(g.lift.mean() - truth), "bias_se": sd / np.sqrt(n), "sd_of_estimates": sd,
                    "mean_reported_se": float((g.half / 1.96).mean()), "se_ratio": float((g.half / 1.96).mean() / sd),
                    "coverage": covered / n, "coverage_low": lo, "coverage_high": hi,
                    "reject_rate": float((g.p < 0.05).mean()), "mean_ci_width": float((2 * g.half).mean())})
    s = pd.DataFrame(out)
    w = s.pivot(index="experiment", columns="estimator", values="mean_ci_width")
    s["cuped_ci_width_vs_naive"] = s.experiment.map(w["cuped"] / w["naive"]) if {"cuped", "naive"} <= set(w.columns) else np.nan
    return s


def main(n_seeds: int = 60, n_users: int = 20_000) -> None:
    runs = []
    for i in range(n_seeds):
        runs += one_replicate(1000 + i, n_users)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{n_seeds} worlds done", flush=True)
    runs = pd.DataFrame(runs)
    summ = summarise(runs)
    art = ROOT / "artifacts"; art.mkdir(exist_ok=True)
    runs.to_csv(art / "validation_runs.csv", index=False)
    summ.to_csv(art / "validation_summary.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.float_format", lambda v: f"{v:.4f}")
    print(f"\n{n_seeds} fresh simulated worlds x {n_users:,} users. Truth = lift built into the simulator.\n")
    print(summ[["experiment", "estimator", "truth", "mean_estimate", "bias", "bias_se", "sd_of_estimates", "se_ratio",
                "coverage", "coverage_low", "coverage_high", "reject_rate", "cuped_ci_width_vs_naive"]].to_string(index=False))
    print("\nHow to read it: bias within about 2 bias_se of zero = unbiased; se_ratio near 1 = honest standard errors;")
    print("coverage near 0.95 with a Wilson range that contains 0.95 = well-calibrated intervals; reject_rate is power for real")
    print("effects and the false-positive rate for exp_003 (truth 0). With few worlds the coverage range is wide: trust se_ratio more.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=60)
    ap.add_argument("--n-users", type=int, default=20_000)
    a = ap.parse_args(); main(a.seeds, a.n_users)
