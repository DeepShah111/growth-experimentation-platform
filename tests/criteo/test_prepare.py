import gzip

import numpy as np
import pandas as pd

import src.criteo.prepare as cp


def test_sampling_keeps_rate_and_columns(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    n = 60_000
    df = pd.DataFrame(rng.normal(size=(n, 12)).astype("float32"), columns=cp.FEATS)
    df["treatment"] = (rng.random(n) < 0.85).astype("int8")
    df["conversion"] = (rng.random(n) < 0.003).astype("int8")
    df["visit"] = (rng.random(n) < 0.05).astype("int8")
    df["exposure"] = (df.treatment * (rng.random(n) < 0.04)).astype("int8")
    src = tmp_path / "fake.csv.gz"
    with gzip.open(src, "wt") as f:
        df.to_csv(f, index=False)
    monkeypatch.setitem(cp.PUBLISHED, "rows", n)          # so p = n_target / n works on the small file
    s, tot = cp.sample_stream(str(src), 20_000, seed=1, chunksize=25_000)
    assert tot["rows"] == n and abs(len(s) - 20_000) < 600
    assert abs(s.treatment.mean() - 0.85) < 0.01
    assert set(cp.FEATS + ["treatment", "conversion", "visit", "exposure"]) <= set(s.columns)
