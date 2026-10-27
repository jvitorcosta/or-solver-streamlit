"""Headless runs of the whole app with Streamlit's AppTest."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.element_tree import Button

from solvedor.config.settings import extract_examples_from_resources_directory

_APP = str(Path(__file__).parents[1] / "streamlit_app.py")
_TEXTBOOK = "max 3x + 2y\nsubject to:\nx + 2y <= 4\n2x + y <= 4\n"


@pytest.fixture
def app() -> AppTest:
    at = AppTest.from_file(_APP, default_timeout=60).run()
    assert not at.exception
    return at


def _button(at: AppTest, key: str) -> Button:
    return next(b for b in at.button if b.key == key)


def _solve(at: AppTest, text: str) -> AppTest:
    at.text_area(key="problem_text").set_value(text)
    _button(at, "solve").click().run()
    assert not at.exception
    return at


def _objective(at: AppTest) -> str | None:
    labels = {"Objective Value", "Valor Objetivo"}
    return next((m.value for m in at.metric if m.label in labels), None)


def test_widgets_have_labels(app: AppTest) -> None:
    assert app.text_area(key="problem_text").label
    assert app.selectbox(key="language").label


def test_solve_shows_results(app: AppTest) -> None:
    _solve(app, _TEXTBOOK)
    assert _objective(app) == "6.666667"
    assert len(app.get("plotly_chart")) == 1


def test_results_survive_a_language_switch(app: AppTest) -> None:
    _solve(app, _TEXTBOOK)
    app.selectbox(key="language").set_value("pt").run()
    assert not app.exception
    assert app.text_area(key="problem_text").value == _TEXTBOOK
    assert _objective(app) == "6.666667"
    assert any(m.label == "Valor Objetivo" for m in app.metric)


def test_results_survive_an_unrelated_rerun(app: AppTest) -> None:
    _solve(app, _TEXTBOOK)
    app.run()
    assert _objective(app) == "6.666667"


def test_parse_error_is_shown_and_kept(app: AppTest) -> None:
    _solve(app, "max x\nsubject to:\nx <=\n")
    assert len(app.error) == 1
    app.run()
    assert len(app.error) == 1


def test_example_load_and_clear(app: AppTest) -> None:
    app.button_group[0].set_value("production_planning").run()
    _button(app, "load_example").click().run()
    assert "product_a" in (app.text_area(key="problem_text").value or "")
    assert any("Production Planning" in c.value for c in app.caption)

    _button(app, "clear").click().run()
    assert not app.exception
    assert app.text_area(key="problem_text").value == ""
    assert not any("None" in c.value for c in app.caption)
    assert app.session_state["example_key"] is None
    assert _objective(app) is None


def test_pill_keeps_selection_across_language_switch(app: AppTest) -> None:
    app.button_group[0].set_value("transportation").run()
    app.selectbox(key="language").set_value("pt").run()
    assert not app.exception
    assert app.session_state["example_key"] == "transportation"
    assert _button(app, "load_example")  # the preview is still open


def test_empty_solve_does_not_crash(app: AppTest) -> None:
    _solve(app, "   ")
    assert _objective(app) is None


@pytest.mark.parametrize("language_code", ["en", "pt"])
def test_every_gallery_example_loads_and_solves(app: AppTest, language_code: str) -> None:
    app.selectbox(key="language").set_value(language_code).run()
    for key in extract_examples_from_resources_directory():
        app.button_group[0].set_value(key).run()
        _button(app, "load_example").click().run()
        _button(app, "solve").click().run()
        assert not app.exception, key
        assert _objective(app) is not None, key
        assert len(app.error) == 0, key
