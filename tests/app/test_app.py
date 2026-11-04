"""The Streamlit app: every page must render (a) with no artifacts at all, (b) with the repository's real artifacts,
and (c) the decision scorecard must show the right counts and reasons."""
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
APP = str(ROOT / "app" / "streamlit_app.py")


def _run(monkeypatch, art_dir=None):
    if art_dir is not None:
        monkeypatch.setenv("GEP_ARTIFACTS", str(art_dir))
    else:
        monkeypatch.delenv("GEP_ARTIFACTS", raising=False)
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception, at.exception
    return at


def _walk_pages(at):
    pages = list(at.sidebar.radio[0].options)
    assert len(pages) >= 10
    for p in pages:
        at = at.sidebar.radio[0].set_value(p).run()
        assert not at.exception, (p, at.exception)
    return at, pages


def test_all_pages_render_without_any_artifacts(tmp_path, monkeypatch):
    at, pages = _walk_pages(_run(monkeypatch, tmp_path))
    assert {"Overview", "Decision scorecard", "SQL layer", "Platform validation"} <= set(pages)


@pytest.mark.skipif(not (ROOT / "artifacts" / "scorecard.csv").exists(), reason="repo artifacts not present")
def test_all_pages_and_interactions_render_with_the_real_repo_artifacts(monkeypatch):
    at = _run(monkeypatch)
    at, _ = _walk_pages(at)
    at = at.sidebar.radio[0].set_value("SQL layer").run()
    for q in list(at.selectbox[0].options):                                    # every SQL query choice
        at = at.selectbox[0].set_value(q).run()
        assert not at.exception, (q, at.exception)
    at = at.sidebar.radio[0].set_value("Decision scorecard").run()
    for e in list(at.selectbox[0].options):                                    # every experiment's reasons
        at = at.selectbox[0].set_value(e).run()
        assert not at.exception, (e, at.exception)
    at = at.sidebar.radio[0].set_value("Overview").run()
    assert [m.label for m in at.metric][:2] == ["Experiments analysed", "Caught as untrustworthy"]
    assert len(at.metric) == 4                                                   # all four headline numbers present


def _fake_scorecard(d: Path):
    pd.DataFrame({"experiment": ["exp_001", "exp_006", "exp_008"], "name": ["a", "b", "c"], "primary_metric": ["orders"] * 3,
                  "decision": ["SHIP", "DO NOT TRUST", "ITERATE"], "lift": [0.11, 0.13, 0.07], "ci_low": [0.08, 0.10, 0.04],
                  "ci_high": [0.14, 0.17, 0.10], "p_adj": [1e-12, None, 1e-7], "mde": [0.04, 0.04, 0.036], "srm_p": [0.6, 1e-132, 0.2],
                  "n_treat": [45000, 40000, 50000], "n_ctrl": [45000, 47000, 50000], "net_value": [None, None, -1.29],
                  "reasons": ["lift +11% | overlaps exp_004", "sample ratio mismatch", "real lift | unprofitable"]}).to_csv(d / "scorecard.csv", index=False)


def test_scorecard_page_shows_counts_zero_counts_and_reasons(tmp_path, monkeypatch):
    _fake_scorecard(tmp_path)
    at = _run(monkeypatch, tmp_path)
    at, _ = _walk_pages(at)
    at = at.sidebar.radio[0].set_value("Decision scorecard").run()
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["SHIP"] == "1" and metrics["ITERATE"] == "1" and metrics["DO NOT TRUST"] == "1"
    assert metrics["INCONCLUSIVE (underpowered)"] == "0"                           # a decision that does not occur must show 0
    at.selectbox[0].set_value("exp_006").run()
    assert any("sample ratio mismatch" in m.value for m in at.markdown)
    at = at.sidebar.radio[0].set_value("Overview").run()
    assert any(m.label == "Offer sent to everyone" and "-1.29" in m.value for m in at.metric)
