"""Run every analysis query in sql/analysis/ and save each result as a CSV in artifacts/sql/.

Run:  python -m src.warehouse.run_sql            (needs the DuckDB warehouse)
      python -m src.warehouse.run_sql --only rfm_segments
"""
from __future__ import annotations

import argparse
import re

import duckdb
import pandas as pd

from src.config import DB_PATH, ROOT, SQL_DIR

OUT = ROOT / "artifacts" / "sql"

# Parameters for queries that take them ($name placeholders). Queries not listed run with none.
# $as_of is filled in from the data (day after the last order) when left as None.
PARAMS: dict[str, dict] = {
    "experiment_interaction": {"exp_a": "exp_001", "exp_b": "exp_004"},
    "rfm_segments": {"as_of": None},
}


def _needed(sql: str) -> set[str]:
    return set(re.findall(r"\$([a-z_]+)", sql))


def run_query(con: duckdb.DuckDBPyConnection, name: str, params: dict | None = None) -> pd.DataFrame:
    sql = (SQL_DIR / "analysis" / f"{name}.sql").read_text()
    need = _needed(sql)
    p = dict(PARAMS.get(name, {})) if params is None else dict(params)
    if "as_of" in need and p.get("as_of") is None:
        p["as_of"] = con.execute("SELECT CAST(MAX(order_ts) AS DATE) FROM orders").fetchone()[0]
    p = {k: v for k, v in p.items() if k in need}
    missing = need - set(p)
    if missing:
        raise ValueError(f"{name}: missing parameters {sorted(missing)}")
    return con.execute(sql, p).df() if p else con.execute(sql).df()


def main(only: str | None = None) -> dict[str, pd.DataFrame]:
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH), read_only=True)
    names = sorted(f.stem for f in (SQL_DIR / "analysis").glob("*.sql"))
    if only:
        names = [n for n in names if n == only]
    results = {}
    pd.set_option("display.width", 200, "display.max_columns", 20)
    for n in names:
        df = run_query(con, n)
        df.to_csv(OUT / f"{n}.csv", index=False)
        results[n] = df
        print(f"\n=== {n}  ({len(df):,} rows) ===")
        print(df.head(8).to_string(index=False))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    main(ap.parse_args().only)
