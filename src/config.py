"""Central configuration: simulation size, experiment definitions, paths."""
from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
WAREHOUSE_DIR = DATA_DIR / "warehouse"
# On Colab the project lives on Google Drive, but a DuckDB database file is faster and safer on the
# runtime's local disk. It is cheap to rebuild from the Parquet files (python -m src.warehouse.build_warehouse).
ON_COLAB = os.path.exists("/content")
DB_PATH = Path(os.environ.get("GEP_DB", "/content/gep_work/experimentation.duckdb" if ON_COLAB
                              else str(DATA_DIR / "experimentation.duckdb")))
SQL_DIR = ROOT / "sql"


@dataclass(frozen=True)
class SimConfig:
    """Size knobs. Lower n_users if your laptop struggles."""

    n_users: int = 100_000
    n_days: int = 84               # 12 weeks; day 0 is a Monday
    pre_period_days: int = 28      # days 0-27: nobody is in an experiment yet
    start_date: dt.date = dt.date(2026, 1, 5)
    seed: int = 42
    n_products: int = 300
    chunk_users: int = 20_000      # memory control: users simulated per batch
    inject_bugs: bool = True       # planted data-quality problems (see ANSWER KEY)


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    name: str
    start_day: int
    end_day: int
    kind: str                       # constant | novelty | new_users_only | by_engagement | by_profile | none
    lift: float                     # relative lift on purchase probability
    long_run_lift: float = 0.0      # only used by 'novelty'
    half_life_days: float = 6.0     # only used by 'novelty'
    exclusive_group: str | None = None
    planted_issue: str = ""
    notes: str = ""
    target: str = "conversion"      # conversion -> acts on purchase probability; order_value -> acts on basket value
    seg_lifts: tuple = ()           # only for kind='by_engagement': lift per engagement segment (casual, medium, loyal)
    channel_lifts: tuple = ()       # only for kind='by_profile': added lift per channel (organic, paid, referral)
    device_lifts: tuple = ()        # only for kind='by_profile': added lift per device (android, ios, web)


# Day numbers are relative to start_date. Days 0-27 are the pre-period.
EXPERIMENTS: tuple[Experiment, ...] = (
    Experiment("exp_001", "checkout_button_v2", 28, 55, "constant", 0.08,
               exclusive_group="homepage_surface",
               planted_issue="overlaps exp_004 although both claim the same surface",
               notes="Clear positive effect: +8% relative on checkout->purchase."),
    Experiment("exp_002", "delivery_fee_banner", 28, 69, "constant", 0.015,
               notes="Tiny effect. Tests whether the sample size had enough power."),
    Experiment("exp_003", "recs_carousel", 42, 69, "none", 0.0,
               notes="True null. Should come out 'not significant' ~95% of the time."),
    Experiment("exp_004", "homepage_redesign", 28, 55, "novelty", 0.12,
               long_run_lift=0.01, half_life_days=6.0, exclusive_group="homepage_surface",
               planted_issue="overlaps exp_001",
               notes="Novelty effect: +12% at first, decays to about +1%."),
    Experiment("exp_005", "onboarding_flow", 28, 83, "new_users_only", 0.18,
               notes="Heterogeneous: +18% for users who joined during the window, 0 for others."),
    Experiment("exp_006", "search_ranking_v3", 42, 69, "none", 0.0,
               planted_issue="SRM: low-activity treatment users silently dropped from assignments",
               notes="True effect is ZERO, but the bug makes a naive comparison look positive."),
    Experiment("exp_007", "free_delivery_threshold", 28, 69, "constant", 0.06, target="order_value",
               notes="Revenue metric: +6% basket value, conversion unchanged. Heavy-tailed outcome."),
    Experiment("exp_008", "winback_offer_profile", 28, 83, "by_profile", 0.0,
               channel_lifts=(0.04, 0.30, -0.10), device_lifts=(0.0, -0.05, 0.15),
               notes="Uplift test bed (learnable): lift = channel effect + device effect, e.g. paid+web ~ +45%, "
                     "referral+ios ~ -15%. Depends only on OBSERVABLE user attributes."),
    Experiment("exp_009", "personalised_offer_latent", 28, 83, "by_engagement", 0.0,
               seg_lifts=(0.30, 0.10, -0.03),
               notes="Heterogeneity tied to a HIDDEN engagement segment. Deliberately hard: shows when uplift "
                     "models cannot recover an effect."),
)


# ------------------------------------------------------------------ business assumptions (NOT measured facts)
VALUE_PER_ORDER = 50.0     # contribution margin per order in INR (~12% of a ~INR 400 basket)
COST_PER_TREATED = 3.0     # INR cost of showing the win-back offer to one user
