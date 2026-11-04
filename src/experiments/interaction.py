"""Interaction check for two experiments that ran on the same users at the same time.

Question: is the effect of A the same whether or not the user also got B?
Input: the 2x2 table from sql/analysis/experiment_interaction.sql (cells with users, orders, orders_sq).
Interaction = (effect of A when B is on) - (effect of A when B is off).  Zero means the effects simply add.

Run:  python -m src.experiments.interaction --a exp_001 --b exp_004
"""
from __future__ import annotations

import argparse

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from src.config import DB_PATH, ROOT, SQL_DIR


def interaction_effect(cells: pd.DataFrame) -> dict:
    c = cells.set_index(["variant_a", "variant_b"])
    mean = (c["orders"] / c["users"]).to_dict()
    var = ((c["orders_sq"] / c["users"] - (c["orders"] / c["users"]) ** 2) * c["users"] / (c["users"] - 1)).to_dict()
    n = c["users"].to_dict()
    key = lambda a, b: (a, b)  # noqa: E731
    tt, tc, ct, cc = (key("treatment", "treatment"), key("treatment", "control"),
                      key("control", "treatment"), key("control", "control"))
    eff_a_given_b = mean[tt] - mean[ct]
    eff_a_given_not_b = mean[tc] - mean[cc]
    inter = eff_a_given_b - eff_a_given_not_b
    se = np.sqrt(sum(var[k] / n[k] for k in (tt, tc, ct, cc)))
    z = inter / se
    return {"effect_A_when_B_on": eff_a_given_b, "effect_A_when_B_off": eff_a_given_not_b,
            "interaction": inter, "ci_half": 1.96 * se, "p_value": float(2 * stats.norm.sf(abs(z))),
            "users": int(sum(n.values()))}


def main(a: str, b: str) -> None:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    cells = con.execute((SQL_DIR / "analysis" / "experiment_interaction.sql").read_text(),
                        {"exp_a": a, "exp_b": b}).df()
    pd.set_option("display.width", 200)
    print(f"Users in both {a} and {b}, by variant combination:")
    print(cells.drop(columns="orders_sq").to_string(index=False))
    r = interaction_effect(cells)
    (ROOT / "artifacts").mkdir(exist_ok=True)
    pd.DataFrame([r]).to_csv(ROOT / "artifacts" / f"interaction_{a}_{b}.csv", index=False)
    print(f"\nEffect of {a} (orders/user) when {b} is ON : {r['effect_A_when_B_on']:+.4f}")
    print(f"Effect of {a} (orders/user) when {b} is OFF: {r['effect_A_when_B_off']:+.4f}")
    print(f"Interaction = {r['interaction']:+.4f} +/- {r['ci_half']:.4f}, p = {r['p_value']:.3f}")
    print("Not significant does NOT prove the effects add; it means this sample cannot tell.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="exp_001"); ap.add_argument("--b", default="exp_004")
    a = ap.parse_args(); main(a.a, a.b)
