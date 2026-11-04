"""Dataset catalogue for the platform: real randomised data first, a simulator with a known answer for validation.

Every loader returns a plain DataFrame plus a `DatasetSpec` that tells the analysis page which column is the arm, which
columns are outcomes and which are safe pre-experiment covariates. A user CSV is handled the same way.
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ROOT, SQL_DIR

SAMPLE_DIR = ROOT / "sample_data"
HILLSTROM_URL = ("http://www.minethatdata.com/"
                 "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv")


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    variant: str
    control: str
    outcomes: dict                      # column -> plain-English meaning
    covariates: tuple                   # pre-experiment columns, safe for adjustment
    real: bool
    description: str
    caveats: str = ""
    guardrails: tuple = ()
    features: tuple = ()                # columns usable as targeting features
    categorical: tuple = ()
    extra: dict = field(default_factory=dict)


# ------------------------------------------------------------------------------------ Hillstrom
HILLSTROM = DatasetSpec(
    key="hillstrom", label="Hillstrom e-mail test (real, randomised)", variant="segment", control="No E-Mail",
    outcomes={"conversion": "bought in the 2 weeks after the e-mail (0/1)", "visit": "visited the website (0/1)",
              "spend": "dollars spent in the 2 weeks"},
    covariates=("recency", "history", "mens", "womens", "newbie"), real=True,
    description="64,000 past customers randomly split into men's e-mail, women's e-mail or no e-mail (about 21,300 each). "
                "Published by Kevin Hillstrom (MineThatData) for the public e-mail analytics challenge.",
    caveats="Dates from 2008 and one retailer. Spend is zero for about 99% of users, so it is heavy-tailed and noisy.",
    guardrails=("visit",), features=("recency", "history", "mens", "womens", "newbie", "zip_code", "channel"),
    categorical=("zip_code", "channel"))


def _hillstrom_candidates():
    env = os.environ.get("HILLSTROM_PATH")
    return [Path(env)] if env else [SAMPLE_DIR / "hillstrom.csv.gz", ROOT / "data" / "hillstrom.csv",
                                    ROOT / "data" / "hillstrom.csv.gz"]


def hillstrom_available() -> bool:
    return any(p.exists() for p in _hillstrom_candidates())


def load_hillstrom(path: str | Path | None = None, allow_download: bool = False) -> pd.DataFrame:
    """Load the Hillstrom file from a local path (committed sample, data/, or HILLSTROM_PATH).

    allow_download=True tries the original MineThatData URL when no local copy exists (needs internet).
    """
    paths = [Path(path)] if path else _hillstrom_candidates()
    for p in paths:
        if p.exists():
            return pd.read_csv(p)
    if allow_download:
        return pd.read_csv(HILLSTROM_URL)
    raise FileNotFoundError("Hillstrom data not found. Put the CSV at sample_data/hillstrom.csv.gz or data/hillstrom.csv "
                            "(python -m src.engine.make_samples can fetch it), or set HILLSTROM_PATH.")


# -------------------------------------------------------------------------------------- Criteo
CRITEO_FEATS = tuple(f"f{i}" for i in range(12))
CRITEO = DatasetSpec(
    key="criteo", label="Criteo ads uplift test (real, randomised)", variant="arm", control="control",
    outcomes={"visit": "visited the advertiser's site (0/1)", "conversion": "bought (0/1)"},
    covariates=CRITEO_FEATS, real=True,
    description="Randomised advertising test: about 85% of users were targeted, 15% held out. Criteo Uplift v2.1 "
                "(13.98M rows in full); this page uses a random sample.",
    caveats="Features are anonymised. Analysis is by assignment (treatment), not by whether an ad was actually shown.",
    features=CRITEO_FEATS)


def _criteo_candidates():
    env = os.environ.get("CRITEO_PATH")
    return [Path(env)] if env else [SAMPLE_DIR / "criteo_100k.parquet", ROOT / "data" / "criteo_1m.parquet"]


def criteo_available() -> bool:
    return any(p.exists() for p in _criteo_candidates())


def load_criteo(path: str | Path | None = None, max_rows: int | None = 300_000, seed: int = 2026) -> pd.DataFrame:
    paths = [Path(path)] if path else _criteo_candidates()
    for p in paths:
        if p.exists():
            df = pd.read_parquet(p)
            if max_rows and len(df) > max_rows:
                df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)
            df["arm"] = np.where(df["treatment"] == 1, "treatment", "control")
            return df
    raise FileNotFoundError("Criteo sample not found. Run: python -m src.criteo.prepare, then python -m src.engine.make_samples")


# ------------------------------------------------------------------------------------ simulator
def simulated_spec(exp_id: str, primary: str) -> DatasetSpec:
    money = primary == "revenue_per_user"
    return DatasetSpec(
        key=f"sim:{exp_id}", label=f"Simulated app, {exp_id} (known truth)", variant="variant", control="control",
        outcomes=({"revenue_in": "revenue per user in the window", "orders_in": "orders per user in the window"} if money
                  else {"orders_in": "orders per user in the window", "revenue_in": "revenue per user in the window"}),
        covariates=("revenue_pre",) if money else ("orders_pre",), real=False,
        description="A simulated grocery app with planted effects and bugs, used as a test bench because the true answer is known.",
        caveats="Simulated: good for checking that the platform finds the truth, not as evidence about the real world.",
        guardrails=("orders_in",) if money else ("revenue_in",),
        features=("orders_pre", "revenue_pre", "sessions_pre", "days_since_signup", "is_new", "device", "channel", "city"),
        categorical=("device", "channel", "city"))


def build_demo_warehouse(n_users: int = 12_000, seed: int = 42, return_truth: bool = False):
    """Small in-memory DuckDB warehouse from the simulator (all observable tables + the staging view).

    The hidden answer key is NOT loaded into the database; with return_truth=True it is returned separately as a DataFrame
    so the app can reveal it on request without the SQL box being able to see it.
    """
    import dataclasses

    import duckdb

    from src.config import SimConfig
    from src.simulation.simulate import simulate
    tables = simulate(dataclasses.replace(SimConfig(), n_users=n_users, chunk_users=min(n_users, 6_000), seed=seed))
    con = duckdb.connect(":memory:")
    for name in ("users", "products", "sessions", "events", "orders", "experiments", "assignments"):
        con.register(f"_{name}", tables[name])
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM _{name}")
        con.unregister(f"_{name}")
    for f in sorted((SQL_DIR / "staging").glob("*.sql")):
        con.execute(f.read_text())
    return (con, tables["answer_key"]) if return_truth else con


def experiment_in_sql(exp_sql: bool = True) -> str:
    """The complete 'analyse an experiment in pure SQL' statement: the per-user mart wrapped as the `metrics` CTE
    followed by sql/analysis/experiment_in_sql.sql. Parameter: $exp_id."""
    mart = (SQL_DIR / "marts" / "experiment_user_metrics.sql").read_text().rstrip()
    mart = mart[: mart.rindex(";")]
    body = (SQL_DIR / "analysis" / "experiment_in_sql.sql").read_text()
    return f"WITH metrics AS (\n{mart}\n),\n{body}"


def load_simulated(con, exp_id: str) -> pd.DataFrame:
    return con.execute((SQL_DIR / "marts" / "experiment_user_metrics.sql").read_text(), {"exp_id": exp_id}).df()


# ------------------------------------------------------------------------------------- uploads
MAX_UPLOAD_ROWS = 1_000_000


def read_upload(data: bytes | str, max_rows: int = MAX_UPLOAD_ROWS) -> pd.DataFrame:
    """Parse an uploaded CSV with a hard row cap. Raises ValueError with a message meant for the user."""
    buf = io.BytesIO(data if isinstance(data, bytes) else data.encode())
    try:
        df = pd.read_csv(buf, nrows=max_rows + 1)
    except Exception as exc:                                    # noqa: BLE001 - surface any parse problem as one clean message
        raise ValueError(f"could not read the file as CSV: {exc}") from exc
    if len(df) > max_rows:
        raise ValueError(f"file has more than {max_rows:,} rows; upload a sample")
    if df.shape[1] < 2:
        raise ValueError("the CSV needs at least two columns (an arm label and an outcome)")
    return df


def guess_columns(df: pd.DataFrame) -> dict:
    """Best guess of (variant, control, outcome, covariates) for an uploaded table, to pre-fill the mapping form."""
    names = {c.lower(): c for c in df.columns}
    variant = next((names[n] for n in ("variant", "arm", "group", "treatment", "segment", "bucket") if n in names), None)
    if variant is None:
        cand = [c for c in df.columns if 2 <= df[c].nunique() <= 6 and not pd.api.types.is_float_dtype(df[c])]
        variant = cand[0] if cand else df.columns[0]
    arms = sorted(df[variant].astype(str).unique())
    control = next((a for a in arms if a.lower() in ("control", "ctrl", "holdout", "0", "false", "no e-mail", "baseline", "a")), arms[0])
    nums = [c for c in df.columns if c != variant and pd.api.types.is_numeric_dtype(df[c])]
    outcome = next((names[n] for n in ("conversion", "converted", "revenue", "spend", "orders", "purchase", "outcome", "y") if n in names
                    and names[n] in nums), nums[0] if nums else None)
    return {"variant": variant, "control": control, "outcome": outcome, "numeric": nums}
