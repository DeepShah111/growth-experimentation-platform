# Deploying the results viewer (Streamlit Community Cloud, free)

The app (`app/streamlit_app.py`) only reads the small files in `artifacts/` and `docs/memos/`, so it needs no database and no data.

## Before you start
- The repository must be on GitHub, and it must be **public** for the free tier.
- `artifacts/` and `docs/` must be committed (they are: only `data/` is git-ignored).
- Dependencies are read from `app/requirements.txt`: Streamlit Community Cloud looks in the app's own folder first and then in the repo root.

## Steps
1. Go to share.streamlit.io and sign in with GitHub. Allow access to your repository if asked.
2. Click **Create app** (or **New app**) and choose the option to deploy from an existing GitHub repo.
3. Repository: `<your-username>/growth-experimentation-platform`. Branch: `main`. **Main file path: `app/streamlit_app.py`**.
4. If a custom app URL (subdomain) field is offered, choose a short readable name, for example `growth-experimentation-platform`.
5. Click **Deploy**. The first build takes a few minutes; open **Manage app** to watch the logs.

## Check the live link (do all of these)
- [ ] The URL opens in a private/incognito window without signing in.
- [ ] The sidebar shows 11 sections and the **Overview** page loads with four headline numbers and a coloured chart.
- [ ] **Decision scorecard**: four count boxes (3 SHIP, 1 ITERATE, 4 INCONCLUSIVE, 1 DO NOT TRUST), the interval chart, and the "Why did it decide that?" dropdown changes the reasons.
- [ ] **SQL layer**: every query in the dropdown shows SQL code on the left and a chart on the right (the last one, the per-user table, shows code and a note).
- [ ] **Who gets the offer?**, tab "Does it pay?": moving the two sliders changes the profit curve and the numbers above it.
- [ ] **Decision memos** shows both memos; **Real data (Criteo)** and **Platform validation** show charts, not "not found" messages.
- [ ] No red error box on any page, and no page says a file "was not found in artifacts/".

## Then add the link everywhere
- In `README.md`, replace `https://YOUR-APP-NAME.streamlit.app` (it appears twice on the first lines) with your real URL, commit and push.
- On GitHub, click the gear next to **About**, paste the URL into **Website**, and add topics such as `ab-testing`, `causal-inference`, `uplift-modeling`, `duckdb`, `sql`.
- Put the URL on your resume next to the project name.

## Things to know
- Free apps go to sleep after a period without visitors (community answers say about 12 hours). A visitor sees "this app has gone to sleep"; clicking the wake-up button restarts it in about a minute. **Open your own link a few minutes before sending it to a recruiter.**
- If a page shows "not found" for a file, the file is missing from the repo's `artifacts/` folder: check it on GitHub.
- If the build fails, open the logs in **Manage app**; the usual causes are a wrong main-file path or a missing package in `app/requirements.txt`.
