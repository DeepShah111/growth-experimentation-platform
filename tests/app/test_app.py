"""Every page of the Streamlit app must render (a) with no artifacts at all and (b) with minimal artifacts."""
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py")


def _render_all(monkeypatch, art_dir):
    monkeypatch.setenv("GEP_ARTIFACTS", str(art_dir))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, at.exception
    pages = list(at.sidebar.radio[0].options)
    assert len(pages) >= 12
    for p in pages:
        at = at.sidebar.radio[0].set_value(p).run()
        assert not at.exception, (p, at.exception)
    return at, pages


def test_all_pages_render_without_any_artifacts(tmp_path, monkeypatch):
    _, pages = _render_all(monkeypatch, tmp_path)
    assert "Decision scorecard" in pages and "Platform validation" in pages


def _fake_artifacts(d: Path):
    sc = pd.DataFrame({"experiment": ["exp_001", "exp_006", "exp_008"], "name": ["a", "b", "c"],
                       "decision": ["SHIP", "DO NOT TRUST", "ITERATE"], "lift": [0.11, 0.13, 0.07],
                       "ci_low": [0.08, 0.10, 0.04], "ci_high": [0.14, 0.17, 0.10], "p_adj": [1e-12, None, 1e-7],
                       "mde": [0.04, 0.04, 0.036], "srm_p": [0.6, 1e-132, 0.2],
                       "reasons": ["lift +11% | overlaps exp_004", "sample ratio mismatch", "real lift | unprofitable"]})
    sc.to_csv(d / "scorecard.csv", index=False)
    pd.DataFrame({"exposure_week": [1, 2, 3], "rel_lift": [0.08, 0.09, 0.05], "ci_half": [0.05, 0.05, 0.05]}).to_csv(d / "novelty_weekly_exp_004.csv", index=False)
    pd.DataFrame([{"interaction": 0.01, "p_value": 0.2}]).to_csv(d / "interaction_exp_001_exp_004.csv", index=False)
    pd.DataFrame({"estimator": ["naive"], "lift": [0.05], "ci_low": [0.01], "ci_high": [0.09]}).to_csv(d / "revenue_estimators_exp_007.csv", index=False)
    va = pd.DataFrame({"experiment": ["exp_001"], "estimator": ["cuped"], "truth": [0.08], "mean_estimate": [0.083], "bias": [0.003],
                       "bias_se": [0.003], "se_ratio": [1.07], "coverage": [0.98], "coverage_low": [0.93], "coverage_high": [0.99],
                       "reject_rate": [0.74], "cuped_ci_width_vs_naive": [0.93]})
    va.to_csv(d / "validation_summary.csv", index=False)
    pd.DataFrame({"model": ["S-learner"], "qini_mean": [22.6]}).to_csv(d / "uplift_multiseed_summary_exp_008.csv", index=False)
    pd.DataFrame({"difference": ["S minus X"], "mean": [5.1]}).to_csv(d / "uplift_paired_bootstrap_exp_008.csv", index=False)


def test_pages_render_with_minimal_artifacts_and_scorecard_shows_decisions(tmp_path, monkeypatch):
    _fake_artifacts(tmp_path)
    at, _ = _render_all(monkeypatch, tmp_path)
    at = at.sidebar.radio[0].set_value("Decision scorecard").run()
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["SHIP"] == "1" and metrics["ITERATE"] == "1" and metrics["DO NOT TRUST"] == "1"
    assert metrics["INCONCLUSIVE (underpowered)"] == "0"                           # a decision that does not occur must show 0
    at.selectbox[0].set_value("exp_006").run()
    assert any("sample ratio mismatch" in m.value for m in at.markdown)
