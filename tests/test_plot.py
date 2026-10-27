from pathlib import Path

import pytest
import yaml

from solvedor.config.language import load_language_translations
from solvedor.solver.api import solve_text
from solvedor.solver.backends import solve
from solvedor.solver.models import Solution, SolverStatus
from solvedor.solver.parser import parse_lp_problem
from solvedor.ui.plot import build_figure, feasible_polygon, plot_box

_EXAMPLES = Path(__file__).parents[1] / "src" / "solvedor" / "resources" / "examples.yaml"
_BOX = (-1.0, 10.0, -1.0, 10.0)


def _rounded(points: list[tuple[float, float]]) -> set[tuple[float, float]]:
    return {(round(x, 6), round(y, 6)) for x, y in points}


def _signed_area(points: list[tuple[float, float]]) -> float:
    pairs = zip(points, points[1:] + points[:1], strict=True)
    return sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in pairs) / 2


def test_textbook_region() -> None:
    problem = parse_lp_problem("max 3x + 2y\nsubject to:\nx + 2y <= 4\n2x + y <= 4\n")
    polygon = feasible_polygon(problem, _BOX)
    assert _rounded(polygon) == _rounded([(0, 0), (2, 0), (4 / 3, 4 / 3), (0, 2)])
    assert _signed_area(polygon) > 0  # counter-clockwise


def test_region_respects_bounds() -> None:
    problem = parse_lp_problem("max x + y\nsubject to:\nx + y <= 100\nwhere:\nx <= 3\ny <= 2\n")
    assert _rounded(feasible_polygon(problem, _BOX)) == {(0, 0), (3, 0), (3, 2), (0, 2)}


def test_unbounded_region_clipped() -> None:
    problem = parse_lp_problem("max x + y\nsubject to:\nx - y <= 1\n")
    polygon = feasible_polygon(problem, _BOX)
    assert len(polygon) >= 3
    assert all(-1 <= x <= 10 and -1 <= y <= 10 for x, y in polygon)
    assert (10.0, 10.0) in _rounded(polygon)  # the open side reaches the box corner


def test_single_point_region() -> None:
    problem = parse_lp_problem("min x\nsubject to:\nx + y = 2\nx - y = 0\n")
    assert _rounded(feasible_polygon(problem, _BOX)) == {(1, 1)}


def test_equality_gives_a_segment() -> None:
    problem = parse_lp_problem("min x\nsubject to:\nx + y = 2\n")
    assert _rounded(feasible_polygon(problem, _BOX)) == {(0, 2), (2, 0)}


def test_infeasible_has_no_figure() -> None:
    t = load_language_translations(language_code="en")
    result = solve_text("max x + y\nsubject to:\nx + y <= 1\nx + y >= 2\n")
    assert result.problem is not None
    assert build_figure(result.problem, result.solution, t) is None


def test_three_variables_have_no_figure() -> None:
    t = load_language_translations(language_code="en")
    result = solve_text("max x + y + z\nsubject to:\nx + y + z <= 1\n")
    assert result.problem is not None
    assert build_figure(result.problem, result.solution, t) is None


def test_box_contains_solution() -> None:
    example = yaml.safe_load(_EXAMPLES.read_text(encoding="utf-8"))["production_planning"]
    problem = parse_lp_problem(example["problem"])
    solution = solve(problem)
    a, b = solution.variable_values["product_a"], solution.variable_values["product_b"]
    assert (round(a, 6), round(b, 6)) == (20, 20)
    x_min, x_max, y_min, y_max = plot_box(problem, solution)
    assert x_min <= a <= x_max and y_min <= b <= y_max
    assert (x_min, y_min) == (0, 0)  # starts at the lower bounds


@pytest.mark.parametrize("language_code", ["en", "pt"])
def test_figure_uses_the_problem_and_translations(language_code: str) -> None:
    t = load_language_translations(language_code=language_code)
    result = solve_text("max 3x + 2y\nsubject to:\nx + 2y <= 4\n2x + y <= 4\n")
    assert result.problem is not None
    figure = build_figure(result.problem, result.solution, t)
    assert figure is not None
    names = [trace.name for trace in figure.data]
    assert names[0] == t.visualization.region_label
    assert "x + 2y ≤ 4" in names
    assert "2x + y ≤ 4" in names
    star = figure.data[-1]
    assert star.name == t.visualization.optimum_label
    assert star.x[0] == pytest.approx(4 / 3)
    assert figure.layout.annotations[0].text == t.visualization.direction_label


def test_solver_error_has_no_figure() -> None:
    t = load_language_translations(language_code="en")
    problem = parse_lp_problem("max x + y\nsubject to:\nx + y <= 4\n")
    solution = Solution(status=SolverStatus.ERROR, solver_message="boom")
    assert build_figure(problem, solution, t) is None


def test_unbounded_region_view_is_not_cramped() -> None:
    problem = parse_lp_problem(
        "max x + y\nsubject to:\nx + y <= 4\nx - y <= 2\nwhere:\nfree x, y\n"
    )
    x_min, x_max, y_min, y_max = plot_box(problem, solve(problem))
    assert x_max - x_min >= 3 and y_max - y_min >= 3


def _contrast(foreground: str, background: str) -> float:
    def luminance(color: str) -> float:
        channels = [int(color.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)]
        linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    light, dark = sorted((luminance(foreground), luminance(background)), reverse=True)
    return (light + 0.05) / (dark + 0.05)


@pytest.mark.parametrize("background", ["#ffffff", "#0e1117"])  # Streamlit light and dark themes
def test_direction_arrow_is_readable_on_light_and_dark_themes(background: str) -> None:
    t = load_language_translations(language_code="en")
    result = solve_text("max 3x + 2y\nsubject to:\nx + 2y <= 4\n2x + y <= 4\n")
    assert result.problem is not None

    figure = build_figure(result.problem, result.solution, t)

    assert figure is not None
    arrow = figure.layout.annotations[0]
    assert _contrast(arrow.arrowcolor, background) >= 3
    assert _contrast(arrow.font.color, background) >= 3
