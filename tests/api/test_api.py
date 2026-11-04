import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402

client = TestClient(app)


def records(n=4000, lift=0.15, seed=0, split=(0.5, 0.5)):
    rng = np.random.default_rng(seed)
    arm = np.where(rng.random(n) < split[0], "control", "treatment")
    pre = rng.poisson(2.0, n)
    y = rng.poisson(2.0 * np.where(arm == "treatment", 1 + lift, 1.0))
    return pd.DataFrame({"variant": arm, "orders": y, "orders_pre": pre}).to_dict("records")


def test_health_and_datasets():
    assert client.get("/health").json()["status"] == "ok"
    ds = client.get("/datasets").json()
    assert {d["key"] for d in ds} == {"hillstrom", "criteo"} and all("outcomes" in d for d in ds)


def test_analyze_returns_a_decision_per_arm_with_reasons():
    r = client.post("/analyze", json={"records": records(), "variant": "variant", "outcome": "orders", "covariates": ["orders_pre"]})
    assert r.status_code == 200
    body = r.json()
    arm = body["arms"][0]
    assert arm["arm"] == "treatment" and arm["decision"] == "SHIP" and arm["ci_low"] < arm["lift"] < arm["ci_high"]
    assert body["srm"]["ok"] and "reasons" in arm and arm["p_adj"] < 0.05


def test_a_broken_split_is_not_trusted_through_the_api_either():
    r = client.post("/analyze", json={"records": records(split=(0.62, 0.38)), "variant": "variant", "outcome": "orders"})
    assert r.json()["arms"][0]["decision"] == "DO NOT TRUST" and r.json()["srm"]["ok"] is False


def test_config_changes_the_decision():
    base = {"records": records(), "variant": "variant", "outcome": "orders", "covariates": ["orders_pre"]}
    cheap = client.post("/analyze", json={**base, "config": {"value_per_unit": 10, "cost_per_treated": 1}}).json()["arms"][0]
    dear = client.post("/analyze", json={**base, "config": {"value_per_unit": 10, "cost_per_treated": 500}}).json()["arms"][0]
    assert cheap["decision"] == "SHIP" and dear["decision"] == "ITERATE"


def test_errors_are_422_with_a_readable_message():
    base = {"records": records(), "variant": "variant", "outcome": "orders"}
    r = client.post("/analyze", json={**base, "config": {"control": "nope"}})
    assert r.status_code == 422 and "control arm 'nope'" in r.json()["detail"]
    assert client.post("/analyze", json={**base, "outcome": "missing"}).status_code == 422
    assert client.post("/analyze", json={**base, "config": {"alpha": 5}}).status_code == 422


def test_csv_upload_endpoint():
    df = pd.DataFrame.from_records(records())
    files = {"file": ("x.csv", df.to_csv(index=False), "text/csv")}
    r = client.post("/analyze/csv", files=files, data={"variant": "variant", "outcome": "orders", "covariates": "orders_pre",
                                                         "config": '{"min_effect": 0.05}'})
    assert r.status_code == 200 and r.json()["arms"][0]["decision"] == "SHIP"
    bad = client.post("/analyze/csv", files={"file": ("x.csv", "a\n1\n", "text/csv")}, data={"variant": "a", "outcome": "a"})
    assert bad.status_code == 422


def test_plan_and_peeking_endpoints():
    p = client.post("/plan", json={"kind": "rate", "baseline": 0.1, "rel_lift": 0.2, "daily_users": 1000}).json()
    assert 3_800 < p["n_per_arm"] < 3_900 and p["days"] == int(np.ceil(p["total_users"] / 1000)) and len(p["power_curve"]) == 12
    assert client.post("/plan", json={"kind": "rate", "baseline": 0.99, "rel_lift": 0.5}).status_code == 422
    pk = client.post("/peeking", json={"n_looks": 10, "n_sims": 5000}).json()["results"]
    naive = [r for r in pk if "naive" in r["strategy"]][0]
    assert naive["false_positive_rate"] > 0.12
