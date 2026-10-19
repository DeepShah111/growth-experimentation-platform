import numpy as np
import pandas as pd

from src.criteo.diagnostics import aipw, propensity_auc, run


def confounded(n=40_000, seed=0, imbalance=True):
    """Visit depends strongly on f0. If imbalance=True, treatment also depends on f0 (a broken randomisation)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 12)).astype("float32")
    pt = 1 / (1 + np.exp(-(1.6 + (0.5 * X[:, 0] if imbalance else 0.0))))
    t = (rng.random(n) < pt).astype("int8")
    base = 1 / (1 + np.exp(-(-3.0 + 0.9 * X[:, 0])))
    visit = (rng.random(n) < np.clip(base + 0.01 * t, 0, 1)).astype("int8")       # true effect = +1 point
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(12)])
    df["treatment"], df["exposure"], df["visit"] = t, (t * (rng.random(n) < 0.04)).astype("int8"), visit
    df["conversion"] = (visit * (rng.random(n) < 0.06)).astype("int8")
    return df


def test_propensity_auc_detects_broken_randomisation_but_not_clean_one():
    bad, good = confounded(imbalance=True), confounded(imbalance=False)
    F = [f"f{i}" for i in range(12)]
    assert propensity_auc(bad[F].to_numpy(), bad.treatment.to_numpy().astype(int)) > 0.56
    assert propensity_auc(good[F].to_numpy(), good.treatment.to_numpy().astype(int)) < 0.52


def test_aipw_beats_naive_when_randomisation_is_confounded():
    df = confounded(n=60_000, imbalance=True)
    F = [f"f{i}" for i in range(12)]
    X, t, y = df[F].to_numpy(), df.treatment.to_numpy().astype(int), df.visit.to_numpy().astype(float)
    naive = y[t == 1].mean() - y[t == 0].mean()
    dr = aipw(X, y, t)["effect"]
    assert abs(dr - 0.01) < abs(naive - 0.01)          # closer to the planted +1 point


def test_run_end_to_end(tmp_path):
    p = tmp_path / "c.parquet"; confounded(30_000).to_parquet(p)
    res = run(str(p), out_dir=tmp_path)
    assert {"balance", "propensity_auc", "effects", "deciles"} <= set(res)


def test_robust_comparison_finds_heterogeneity(tmp_path, make_fake_criteo):
    from src.criteo.diagnostics import robust_comparison
    p = tmp_path / "h.parquet"; make_fake_criteo(60_000).to_parquet(p)
    out = robust_comparison(str(p), reps=20, out_dir=tmp_path, outcomes=("visit",)).set_index("model")
    assert out.loc["S-learner", "qini_extra_per_1000_users"] > out.loc["random", "qini_extra_per_1000_users"]
