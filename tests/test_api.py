import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from streamlit.testing.v1 import AppTest

from solvedor.solver.api import solve_text
from solvedor.solver.models import SolverStatus
from solvedor.ui.results import format_number

_EXAMPLES = Path(__file__).parents[1] / "src" / "solvedor" / "resources" / "examples.yaml"


def test_solve_text_returns_problem_and_solution() -> None:
    example = yaml.safe_load(_EXAMPLES.read_text(encoding="utf-8"))["production_planning"]
    result = solve_text(example["problem"])
    assert result.error is None
    assert result.problem is not None
    assert result.solution is not None
    assert result.solution.status is SolverStatus.OPTIMAL


def test_solve_text_reports_the_parse_error_line() -> None:
    result = solve_text("max x + y\nsubject to:\nx + y <= 4\nx +* y <= 2\n")
    assert (result.problem, result.solution) == (None, None)
    assert result.error is not None
    assert result.error.line == 4


def test_api_does_not_import_streamlit() -> None:
    code = "import sys, solvedor.solver.api; print('streamlit' in sys.modules)"
    output = subprocess.run(  # noqa: S603 - fixed command, our own interpreter
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    assert output.strip() == "False"


@pytest.mark.parametrize(
    ("value", "whole", "expected"),
    [
        (0.0, False, "0"),
        (-0.0, False, "0"),
        (-1e-9, False, "0"),
        (2.5, False, "2.5"),
        (4 / 3, False, "1.333333"),
        (2.9999999, True, "3"),
        (1800.0, False, "1800"),
    ],
)
def test_format_number(value: float, whole: bool, expected: str) -> None:
    assert format_number(value, whole=whole) == expected


def _render(text: str) -> AppTest:
    """Render solve_text's result with the real translations in a headless Streamlit run."""

    def app(problem_text: str) -> None:
        from solvedor.config import language
        from solvedor.solver.api import solve_text
        from solvedor.ui.results import render_results

        translations = language.load_language_translations(language_code="en")
        render_results(solve_text(problem_text), translations)

    at = AppTest.from_function(app, kwargs={"problem_text": text})
    at.run()
    assert not at.exception
    return at


def test_render_zero_objective_is_zero_not_na() -> None:
    at = _render("min x\nsubject to:\nx <= 4\n")
    assert [m.value for m in at.metric][3] == "0"


def test_render_parse_error_shows_the_input_as_plain_text() -> None:
    at = _render("max x\nsubject to:\n**bold** <= 4\n")
    assert len(at.error) == 1
    assert "**bold**" in at.text[0].value


def test_render_unbounded() -> None:
    at = _render("max x + y\nsubject to:\nx - y <= 1\n")
    assert "unbounded" in at.warning[0].value.lower()


def test_render_two_variable_problem_draws_the_plot() -> None:
    at = _render("max 3x + 2y\nsubject to:\nx + 2y <= 4\n2x + y <= 4\n")
    assert len(at.get("plotly_chart")) == 1


def _render_status(status: str, values: dict[str, float]) -> AppTest:
    """Render a two-variable problem with a given solver status (statuses the page can't force)."""

    def app(status: str, values: dict[str, float]) -> None:
        from solvedor.config import language
        from solvedor.solver.api import SolveResult
        from solvedor.solver.models import Solution, SolverStatus
        from solvedor.solver.parser import parse_lp_problem
        from solvedor.ui.results import render_results

        problem = parse_lp_problem("max x + y\nsubject to:\nx + y <= 4\nwhere:\ninteger x, y\n")
        objective = sum(values.values()) if values else None
        solution = Solution(
            status=SolverStatus(status),
            objective_value=objective,
            variable_values=values,
            solver_name="SCIP",
        )
        translations = language.load_language_translations(language_code="en")
        render_results(SolveResult(problem=problem, solution=solution), translations)

    at = AppTest.from_function(app, kwargs={"status": status, "values": values})
    at.run()
    assert not at.exception
    return at


def test_render_best_found_before_time_limit_shows_values_and_warning() -> None:
    at = _render_status(SolverStatus.FEASIBLE.value, {"x": 3.0, "y": 0.0})

    assert any("not proven optimal" in w.value for w in at.warning)
    assert any(m.value == "3" for m in at.metric)


def test_render_time_limit_without_solution_shows_the_hint() -> None:
    at = _render_status(SolverStatus.TIME_LIMIT.value, {})

    assert any("time limit" in w.value for w in at.warning)
    assert any("smaller model" in i.value for i in at.info)
    assert len(at.dataframe) == 0
