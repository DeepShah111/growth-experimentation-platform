# Deploying the platform: Streamlit app + REST API

Two things go live. The **Streamlit app** (interactive pages) on Streamlit Community Cloud, and the **REST API** (Docker) on a free
container host such as a Hugging Face Docker Space. Both run the same code in `src/engine/`.

## 0. Before anything
- Push the repository to GitHub (public for the free tiers). Commit `artifacts/`, `docs/`, `sql/`, `assets/` and **`sample_data/`**
  (about 0.5 MB: `hillstrom.csv.gz`; the Criteo sample is optional, see below).
- Run the tests: `python -m pytest` (167 tests; they need no internet and no big files).
- Optional Criteo page: `python -m src.criteo.prepare` then `python -m src.engine.make_samples` creates `sample_data/criteo_100k.parquet`
  (about 2 MB). Commit it. Without it the app simply hides the Criteo option.
- Hillstrom data: the committed copy is the public challenge file published by Kevin Hillstrom (MineThatData, 2008). Keep the attribution
  in the README. If you prefer not to redistribute it, delete `sample_data/hillstrom.csv.gz` and run `python -m src.engine.make_samples`
  locally; the deployed app will then hide the Hillstrom option.

## 1. Streamlit app (Community Cloud, free)
1. share.streamlit.io, sign in with GitHub, **Create app**.
2. Repository `<you>/growth-experimentation-platform`, branch `main`, **Main file path `app/streamlit_app.py`**.
3. Dependencies come from `app/requirements.txt` (it already lists numpy, scipy, scikit-learn, duckdb, pyarrow, altair).
4. Deploy. The first start builds a 12,000-user demo warehouse in a few seconds (cached afterwards).

### Check the live link
- [ ] Opens in a private window without signing in.
- [ ] **Analyze an experiment** shows Hillstrom with two decision cards and an interval chart. Switch Outcome to `spend`, raise "Cost of
      treating one user" to 500: the decision turns from SHIP to ITERATE.
- [ ] Dataset = Simulated app, Experiment = `exp_006`: the card says DO NOT TRUST.
- [ ] **Plan an experiment**: halving the lift roughly quadruples users per arm.
- [ ] **Peeking simulator**: 10 looks gives about 19% false wins, 1 look about 5%.
- [ ] **Live targeting** loads in a few seconds and shows a plain-language verdict.
- [ ] **SQL workbench**: every project query runs; the pure-SQL tab shows "SQL and Python agree"; `DROP TABLE users` is refused.
- [ ] The older pages (scorecard, memos, validation) still render.

## 2. REST API (Docker)
Local:
```
docker build -t gep-api .
docker run -p 8000:8000 gep-api
curl http://127.0.0.1:8000/health
# interactive docs: http://127.0.0.1:8000/docs
```
Hugging Face Docker Space: create a Space with SDK = Docker, push this repository (or the files the Dockerfile copies), and set the
Space's app port to 8000. Any host that runs a container works the same way; the image reads the port from `$PORT`.

CI (`.github/workflows/ci.yml`) runs the tests on Python 3.11 and 3.12, lints, builds the image and smoke-tests `/health` and `/plan`.

## 3. Link everything
- README: replace both `YOUR-APP-NAME` links with the real app and API URLs, commit, push.
- GitHub About: add the app URL as Website; topics `ab-testing`, `causal-inference`, `uplift-modeling`, `duckdb`, `sql`, `fastapi`.
- Resume: put the app URL next to the project name.

## Things to know
- Free Streamlit apps sleep after inactivity; open your link a few minutes before sharing it. Free container hosts also cold-start.
- The SQL box is read-only, runs with file and network access switched off, a 1 GB memory limit and a 1,000-row cap. It queries simulated data only.
- The API caps JSON at 200,000 records and uploads at 30 MB / 1,000,000 rows. It stores nothing.
- Uploaded data is analysed in memory and discarded. Do not upload personal data to a public demo.
