# v2 upgrade bundle: how to apply

Copy the folders in this bundle over your repository (same relative paths; allow overwrite), then:

```
pip install -r requirements.txt -r app/requirements.txt -r api/requirements.txt
python -m pytest                    # 167 tests
streamlit run app/streamlit_app.py  # interactive app
uvicorn api.main:app --reload       # API docs at http://127.0.0.1:8000/docs
```

## New files
- `src/engine/`  analyze.py, planner.py, sequential.py, targeting.py, uplift_core.py, decision.py, datasets.py, sqlguard.py, make_samples.py
- `api/`  main.py, requirements.txt, __init__.py      + `Dockerfile`, `.dockerignore`
- `app/live_pages.py`  five interactive pages
- `sql/analysis/experiment_in_sql.sql`  a whole experiment analysis in SQL
- `sample_data/hillstrom.csv.gz`  the public Hillstrom challenge file (see attribution note in docs/DEPLOYMENT.md)
- `tests/engine/*`, `tests/api/test_api.py`, `tests/app/test_live_pages.py`
- `pyproject.toml` (lint config), `docs/DEPLOYMENT.md` (replaces the old one)

## Changed files (review the diff)
- `src/experiments/`: stats_utils (shared lift_delta, regression_adjust, net_value_per_user), first_look / revenue / novelty (use the shared
  helpers instead of private copies), scorecard (decision rule moved to src/engine/decision.py; no more global MIN_EFFECT; min_effect is an
  argument), studies (unused `reps` parameter replaced by explicit sizes), interaction (survival function for tiny p-values)
- `src/targeting/uplift.py`, `multiseed.py`, `src/criteo/validate.py`, `diagnostics.py`: import the pure uplift maths from src/engine/uplift_core.py
  (the Criteo code no longer depends on the DuckDB module); orphan comment and two unused variables removed
- `sql/analysis/rfm_segments.sql`: rank-based scores so tied users get identical scores (NTILE split ties by user_id)
- `sql/analysis/weekly_retention.sql`: only weeks that have fully happened, zero counted explicitly (right-censoring)
- `tests/sql/test_sql_layer.py`: retention test data extended + two new tests (ties, censoring)
- `app/streamlit_app.py`, `app/charts.py`, `app/requirements.txt`: new pages routed in, new chart builders, new dependencies
- `README.md`, `.gitignore`, `requirements.txt`, `.github/workflows/ci.yml`

## Before you push
1. If you edited README.md since you uploaded it (git showed `M`), merge my changes by hand instead of overwriting.
2. My test files assume your layout is tests/<area>/test_*.py with a shared tests/conftest.py. If yours differs, move them.
3. Run `docker build -t gep-api .` once on your machine: I could not run Docker in my environment.
4. Re-run the artifact-producing scripts only if you want fresh `artifacts/`; nothing in v2 changes their numbers EXCEPT
   `python -m src.warehouse.run_sql` (rfm_segments and weekly_retention outputs change because of the two SQL fixes).
