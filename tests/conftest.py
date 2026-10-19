"""Shared fixtures. Fixtures (not `from tests.x import ...`) because some environments, e.g. Colab,
have another installed package called `tests` that would shadow ours."""
import numpy as np
import pandas as pd
import pytest


def fake_criteo(n=40_000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 12)).astype("float32")
    t = (rng.random(n) < 0.85).astype("int8")
    p0 = 1 / (1 + np.exp(-(-3.0 + 0.5 * X[:, 1])))                 # baseline visit prob, ~5%
    p1 = np.clip(p0 * (1 + np.where(X[:, 0] > 0.5, 0.6, 0.0)), 0, 1)  # only users with f0>0.5 respond
    visit = (rng.random(n) < np.where(t == 1, p1, p0)).astype("int8")
    conv = (visit * (rng.random(n) < 0.06)).astype("int8")
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(12)])
    df["treatment"], df["exposure"], df["visit"], df["conversion"] = t, t, visit, conv
    return df


@pytest.fixture
def make_fake_criteo():
    """Factory for a small Criteo-shaped DataFrame: make_fake_criteo(n_rows, seed)."""
    return fake_criteo
