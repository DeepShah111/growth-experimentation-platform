"""The interactive pages must do what the app promises: change a control, the answer changes."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
APP = str(ROOT / "app" / "streamlit_app.py")


def _open(page, monkeypatch, tmp_path):
    monkeypatch.setenv("GEP_ARTIFACTS", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=240).run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, at.exception
    return at


def _by_key(widgets, key):
    return next(w for w in widgets if w.key == key)


def test_plan_page_changes_with_the_sliders(monkeypatch, tmp_path):
    at = _open("Plan an experiment", monkeypatch, tmp_path)
    first = {m.label: m.value for m in at.metric}
    _by_key(at.slider, "pl_lift").set_value(5).run()                 # half the lift -> roughly 4x the users
    assert not at.exception
    second = {m.label: m.value for m in at.metric}
    n1, n2 = int(first["Users per arm"].replace(",", "")), int(second["Users per arm"].replace(",", ""))
    assert 3.5 < n2 / n1 < 4.5
    _by_key(at.slider, "pl_rho").set_value(0.6).run()                # a covariate cuts the sample
    n3 = int({m.label: m.value for m in at.metric}["Users per arm"].replace(",", ""))
    assert n3 < 0.7 * n2


def test_peeking_page_shows_more_false_wins_with_more_looks(monkeypatch, tmp_path):
    at = _open("Peeking simulator", monkeypatch, tmp_path)
    _by_key(at.slider, "pk_looks").set_value(1).run()
    one = " ".join(m.value for m in at.markdown)
    _by_key(at.slider, "pk_looks").set_value(20).run()
    twenty = " ".join(m.value for m in at.markdown)
    assert "With **1 looks**" in one and "With **20 looks**" in twenty and not at.exception


def test_analyze_page_decision_moves_with_the_economics_and_the_dataset(monkeypatch, tmp_path):
    at = _open("Analyze an experiment", monkeypatch, tmp_path)
    assert at.selectbox[0].value.startswith("Hillstrom")
    assert "Mens E-Mail" in " ".join(m.value for m in at.markdown)          # a decision card per treatment arm renders
    _by_key(at.selectbox, "an_out").set_value("spend").run()
    assert not at.exception
    spend = " ".join(m.value for m in at.markdown)
    assert ":green[**SHIP**]" in spend                                # e-mail pays at the default economics
    cost = next(w for w in at.number_input if w.key.startswith("an_cost_"))
    cost.set_value(500.0).run()                                       # ... and stops paying when each e-mail costs 500
    assert not at.exception
    dear = " ".join(m.value for m in at.markdown)
    assert ":orange[**ITERATE**]" in dear and ":green[**SHIP**]" not in dear


def test_analyze_page_catches_the_planted_bug_in_the_simulator(monkeypatch, tmp_path):
    at = _open("Analyze an experiment", monkeypatch, tmp_path)
    _by_key(at.selectbox, "an_src").set_value("Simulated app (known truth)").run()
    _by_key(at.selectbox, "an_exp").set_value("exp_006").run()
    assert not at.exception
    assert ":red[**DO NOT TRUST**]" in " ".join(m.value for m in at.markdown)


def test_targeting_page_says_when_no_model_beats_random(monkeypatch, tmp_path):
    at = _open("Live targeting", monkeypatch, tmp_path)
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert any("treat everyone" in l for l in labels)
    assert at.warning or at.success or at.info                          # one of the three honest verdicts is shown


def test_sql_workbench_runs_every_project_query_and_rejects_dangerous_sql(monkeypatch, tmp_path):
    at = _open("SQL workbench", monkeypatch, tmp_path)
    for q in list(_by_key(at.selectbox, "sq_q").options):
        _by_key(at.selectbox, "sq_q").set_value(q).run()
        assert not at.exception and not at.error, (q, [e.value for e in at.error])
    _by_key(at.text_area, "sq_sql").set_value("DROP TABLE users").run()
    _by_key(at.button, "sq_run").click().run()
    assert any("only read-only" in e.value for e in at.error)
    _by_key(at.text_area, "sq_sql").set_value("SELECT COUNT(*) AS n FROM users").run()
    _by_key(at.button, "sq_run").click().run()
    assert not any("only read-only" in e.value for e in at.error) and not at.exception
