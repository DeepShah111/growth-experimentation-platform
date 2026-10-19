"""Load the Parquet files into a DuckDB database and create the staging views.

Run:  python -m src.warehouse.build_warehouse
"""
from __future__ import annotations

import duckdb

from src.config import DB_PATH, SQL_DIR, WAREHOUSE_DIR


def apply_staging(con: duckdb.DuckDBPyConnection) -> None:
    """Create the staging views (clean layer on top of the raw tables). Safe to call repeatedly."""
    for f in sorted((SQL_DIR / "staging").glob("*.sql")):
        con.execute(f.read_text())


def build(db_path=DB_PATH, parquet_dir=WAREHOUSE_DIR) -> duckdb.DuckDBPyConnection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    ddl = (SQL_DIR / "schema" / "00_schema.sql").read_text().replace("{dir}", parquet_dir.as_posix())
    con.execute(ddl)
    apply_staging(con)
    return con


if __name__ == "__main__":
    con = build()
    for (t,) in con.execute("SHOW TABLES").fetchall():
        n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"{t:12s} {n:>10,d}")
    n = con.execute("SELECT COUNT(*) FROM stg_events_clean").fetchone()[0]
    print(f"{'stg_events_clean (view)':24s} {n:>10,d}")
