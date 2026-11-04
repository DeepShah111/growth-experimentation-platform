# Project guide (read this first)

## What this project is, in plain words
A pretend grocery-delivery app generates data. Because the data is simulated, we know the TRUE effect of every
experiment. The platform then checks whether our analysis methods find the right answer, and whether the data and the
experiments can be trusted at all. A real dataset (Criteo ads) confirms the methods behave on real randomised data.
It is a prototype on 100,000 simulated users. It is not a production system.

## The story in five steps (how data flows through the folders)
1. `src/simulation` creates the app data (users, sessions, events, orders, experiments) with planted problems.
2. `src/warehouse` loads it into DuckDB and `sql/` turns raw events into clean tables and reports.
3. `src/experiments` checks each experiment: is it trustworthy, how big must it be, did it fade, do two experiments interact.
4. `src/targeting` builds uplift models: which users should get an offer, and is it worth it.
5. `src/criteo` repeats the key methods on real data. `notebooks/` and `app/` show the results; `tests/` guards everything.

## Folder map
| path | what is inside |
|---|---|
| `src/config.py` | sizes, the 9 experiment definitions, file paths |
| `src/simulation/simulate.py` | builds the fake app and the answer key |
| `src/warehouse/build_warehouse.py` | loads Parquet into DuckDB, creates the staging view |
| `src/warehouse/run_sql.py` | runs every query in `sql/analysis/` and saves CSVs |
| `src/experiments/stats_utils.py` | power formulas, CUPED, O'Brien-Fleming bounds, Holm / BH / Bonferroni |
| `src/experiments/first_look.py` | sample-ratio check, naive vs CUPED effect, compared with the truth |
| `src/experiments/studies.py` | power, peeking and multiple-testing simulations |
| `src/experiments/novelty.py` | week-by-week lift and a trend test (does the effect fade?) |
| `src/experiments/interaction.py` | do two overlapping experiments change each other's effect? |
| `src/experiments/revenue.py` | heavy-tailed revenue: tail profile, 4 estimators, purchase-rate vs basket split |
| `src/experiments/scorecard.py` | all trust gates (incl. economics) in one run, ending in ship / iterate / hold per experiment |
| `src/experiments/validation.py` | repeats the analysis in 100 fresh simulated worlds: bias, interval coverage, honest standard errors |
| `src/targeting/uplift.py` | S / T / X learners, Qini curves, profit view |
| `src/targeting/multiseed.py` | repeats uplift over seeds, paired bootstrap, hold-out profit cut-off |
| `src/criteo/` | download a sample, balance checks, doubly-robust estimates, model comparison |
| `sql/staging/` | one trustworthy row per event (duplicates removed, late events flagged) |
| `sql/marts/` | per-user tables used by the statistics code |
| `sql/analysis/` | 11 reports: funnel, retention, sessionization, window functions, top-N, RFM, streaks, data quality, SRM, overlap, interaction |
| `tests/` | 89 tests (the app tests need streamlit and altair installed); each SQL query is checked against a hand-computed answer |
| `docs/memos/` | two filled decision memos: ship (exp_001) and targeting (exp_008) |
| `notebooks/` | two readable result notebooks |
| `app/` | Streamlit guided tour with 11 sections and interactive charts (`streamlit_app.py` = pages and wording, `charts.py` = chart builders); not deployed yet |
| `assets/`, `docs/*.dot` | the system diagram, the data-and-model-flow diagram and their sources |
| `docs/DEPLOYMENT.md` | how to deploy the app and check the live link |
| `.github/workflows/ci.yml` | runs the tests on every push to GitHub (not tried yet: nothing is pushed) |

## Where we are
Done: simulator, SQL layer with tests, A/B studies, novelty and interaction checks, uplift with a fixed row order,
multi-seed uplift, heavy-tail revenue analysis, the decision scorecard, Criteo validation.
Next: API, deployment, MLflow, GitHub push, decision memo, then the resume. All numbers in the README come from executed runs.

## How to talk about it
Thirty-second version: "I built a simulated grocery app where I know the true effect of nine experiments, then tested
whether standard analysis methods recover it. They catch a planted logging bug that fakes a +19% lift, show that peeking
daily inflates false positives from 5% to 25%, and I repeated the key methods on real randomised Criteo data."

Safe to say (each comes from an executed output): daily peeking 25.0% false positives vs 5.8% for one look; the planted
sample-ratio bug in exp_006 makes a zero-effect experiment look like +19%; exp_002 would need about 350k users per arm;
all three uplift learners beat random targeting on exp_008 (S beats T in a paired bootstrap; S vs X is marginal); the decision scorecard catches the broken exp_006 and ships exp_001 and exp_007, but misses the fading exp_004 and the segment-only exp_005.

Do NOT say: "production-grade", "at scale", "S-learner is best", "advanced SQL" beyond the tested queries, "deployed",
or anything about real business impact. The data is simulated and the profit numbers use assumed costs.

## Commands
`python -m src.simulation.simulate --n-users 100000`, `python -m src.warehouse.build_warehouse`,
`python -m src.warehouse.run_sql`, `python -m src.experiments.first_look`, `python -m src.experiments.novelty`,
`python -m src.experiments.interaction`, `python -m src.experiments.studies all`,
`python -m src.targeting.uplift --exp exp_008`, `python -m pytest`.
