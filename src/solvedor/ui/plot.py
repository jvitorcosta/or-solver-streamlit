"""Draw the feasible region of a two-variable problem from its own constraints.

The region is the view box clipped by every constraint and bound, one half-plane at a time
(Sutherland-Hodgman). An equality clips from both sides, which leaves a segment or a point.
"""

from __future__ import annotations

import itertools
import math

import plotly.graph_objects as go

from solvedor.config.language import TranslationSchema
from solvedor.solver.models import (
    ConstraintOperator,
    ObjectiveDirection,
    Problem,
    Solution,
    SolverStatus,
)

Point = tuple[float, float]
Box = tuple[float, float, float, float]  # x_min, x_max, y_min, y_max
HalfPlane = tuple[float, float, float]  # a, b, c meaning a*x + b*y <= c

_EPS = 1e-9
# Readable on Streamlit's light (#ffffff) and dark (#0e1117) backgrounds: contrast 4.5 and 4.2
_ARROW_COLOR = "#3a76d0"
_COLORS = ["#d62728", "#1f77b4", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2", "#17becf"]


def _axes(problem: Problem) -> tuple[str, str] | None:
    names = problem.extract_all_variable_names()
    return (names[0], names[1]) if len(names) == 2 else None


def _lines(problem: Problem) -> list[tuple[float, float, ConstraintOperator, float]]:
    """Every constraint and finite bound as (a, b, operator, c) for a*x + b*y (op) c."""
    axes = _axes(problem)
    if axes is None:
        return []
    lines = []
    for constraint in problem.constraints:
        coefficients = constraint.expression.variable_coefficients_map
        lines.append(
            (
                float(coefficients.get(axes[0], 0)),
                float(coefficients.get(axes[1], 0)),
                constraint.operator,
                float(constraint.rhs),
            )
        )
    for index, name in enumerate(axes):
        unit = (1.0, 0.0) if index == 0 else (0.0, 1.0)
        variable = problem.variables[name]
        if variable.lower_bound is not None:
            lines.append((*unit, ConstraintOperator.GREATER_EQUAL, float(variable.lower_bound)))
        if variable.upper_bound is not None:
            lines.append((*unit, ConstraintOperator.LESS_EQUAL, float(variable.upper_bound)))
    return lines


def _half_planes(problem: Problem) -> list[HalfPlane]:
    planes = []
    for a, b, operator, c in _lines(problem):
        if operator is not ConstraintOperator.GREATER_EQUAL:
            planes.append((a, b, c))
        if operator is not ConstraintOperator.LESS_EQUAL:
            planes.append((-a, -b, -c))
    return planes


def _clip(polygon: list[Point], plane: HalfPlane) -> list[Point]:
    a, b, c = plane
    scale = max(abs(a), abs(b), 1.0)

    def inside(p: Point) -> bool:
        return a * p[0] + b * p[1] <= c + _EPS * scale

    def crossing(p: Point, q: Point) -> Point:
        fp, fq = a * p[0] + b * p[1] - c, a * q[0] + b * q[1] - c
        t = fp / (fp - fq)
        return (p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))

    result: list[Point] = []
    for p, q in zip(polygon, polygon[1:] + polygon[:1], strict=True):
        if inside(q):
            if not inside(p):
                result.append(crossing(p, q))
            result.append(q)
        elif inside(p):
            result.append(crossing(p, q))
    return result


def _dedupe(points: list[Point]) -> list[Point]:
    unique: list[Point] = []
    for point in points:
        if not any(math.dist(point, other) < 1e-7 for other in unique):
            unique.append(point)
    return unique


def feasible_polygon(problem: Problem, box: Box) -> list[Point]:
    """Vertices of the feasible region inside the box, counter-clockwise; [] if it is empty."""
    x_min, x_max, y_min, y_max = box
    polygon = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
    for plane in _half_planes(problem):
        polygon = _clip(polygon, plane)
        if not polygon:
            return []
    return _dedupe(polygon)


def _vertices(problem: Problem) -> list[Point]:
    """Feasible intersections of every pair of constraint or bound lines."""
    planes = _half_planes(problem)
    lines = [(a, b, c) for a, b, _, c in _lines(problem)]
    points = []
    for (a1, b1, c1), (a2, b2, c2) in itertools.combinations(lines, 2):
        determinant = a1 * b2 - a2 * b1
        if abs(determinant) < _EPS:
            continue
        point = ((c1 * b2 - c2 * b1) / determinant, (a1 * c2 - a2 * c1) / determinant)
        if all(
            a * point[0] + b * point[1] <= c + 1e-7 * max(abs(a), abs(b), 1) for a, b, c in planes
        ):
            points.append(point)
    return points


def plot_box(problem: Problem, solution: Solution | None) -> Box:
    """A view box around the region's corners and the solution, with some margin."""
    axes = _axes(problem)
    points = _vertices(problem)
    if axes and solution is not None and solution.variable_values:
        points.append((solution.variable_values[axes[0]], solution.variable_values[axes[1]]))
    if not points:
        points = [(0.0, 0.0)]
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    # At least the distance from the origin, so a lone corner far out still gets context.
    largest = max(abs(v) for v in xs + ys)
    span = max(max(xs) - min(xs), max(ys) - min(ys), largest, 1.0)
    # An unbounded region gets extra room past its last corner so its open side shows.
    margin = span * (0.5 if solution is None or not solution.is_feasible else 0.15)
    # Each axis covers at least the full span, so a flat region is not drawn squashed.
    x_high, y_high = max(max(xs), min(xs) + span), max(max(ys), min(ys) + span)
    box = [min(xs) - margin, x_high + margin, min(ys) - margin, y_high + margin]
    if axes:
        # Start the view at a lower bound, so the axis is the region's edge as in a textbook.
        for index, name in enumerate(axes):
            low = problem.variables[name].lower_bound
            if low is not None:
                box[2 * index] = max(box[2 * index], float(low))
    return (box[0], box[1], box[2], box[3])


def _format(value: float) -> str:
    return f"{value:g}"


def _label(text: str) -> str:
    return text.replace("*", "").replace("<=", "≤").replace(">=", "≥")


def build_figure(
    problem: Problem, solution: Solution | None, t: TranslationSchema
) -> go.Figure | None:
    """The feasible region, constraint lines, objective direction and solution, or None."""
    axes = _axes(problem)
    no_region = (SolverStatus.INFEASIBLE, SolverStatus.ERROR)  # nothing trustworthy to draw
    if axes is None or (solution is not None and solution.status in no_region):
        return None
    box = plot_box(problem, solution)
    region = feasible_polygon(problem, box)
    if not region:
        return None
    x_min, x_max, y_min, y_max = box
    figure = go.Figure()

    closed = [*region, region[0]]
    figure.add_trace(
        go.Scatter(
            x=[p[0] for p in closed],
            y=[p[1] for p in closed],
            fill="toself",
            fillcolor="rgba(44, 160, 44, 0.18)",
            line={"color": "rgba(44, 160, 44, 0.9)", "width": 2},
            mode="lines+markers" if len(region) < 3 else "lines",
            name=t.visualization.region_label,
            hoverinfo="skip",
        )
    )

    constraint_lines = _lines(problem)[: len(problem.constraints)]
    for index, (constraint, (a, b, _, c)) in enumerate(
        zip(problem.constraints, constraint_lines, strict=True)
    ):
        segment = _segment_in_box(a, b, c, box)
        if segment is None:
            continue
        label = _label(str(constraint))
        figure.add_trace(
            go.Scatter(
                x=[segment[0][0], segment[1][0]],
                y=[segment[0][1], segment[1][1]],
                mode="lines",
                line={"color": _COLORS[index % len(_COLORS)], "width": 2, "dash": "dash"},
                name=label,
                hovertemplate=f"{label}<extra></extra>",
            )
        )

    coefficients = problem.objective.expression.variable_coefficients_map
    gx, gy = float(coefficients.get(axes[0], 0)), float(coefficients.get(axes[1], 0))
    if problem.objective.direction is ObjectiveDirection.MINIMIZE:
        gx, gy = -gx, -gy
    length = math.hypot(gx, gy)
    if length > 0:
        # The arrow points where the objective improves, sized to a fifth of the view.
        size = 0.2 * min(x_max - x_min, y_max - y_min) / length
        cx, cy = sum(p[0] for p in region) / len(region), sum(p[1] for p in region) / len(region)
        figure.add_annotation(
            x=cx + gx * size,
            y=cy + gy * size,
            ax=cx,
            ay=cy,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowwidth=2,
            arrowcolor=_ARROW_COLOR,
            text=t.visualization.direction_label,
            font={"color": _ARROW_COLOR},
        )

    if solution is not None and solution.is_feasible:
        sx, sy = solution.variable_values[axes[0]], solution.variable_values[axes[1]]
        figure.add_trace(
            go.Scatter(
                x=[sx],
                y=[sy],
                mode="markers",
                marker={
                    "symbol": "star",
                    "size": 18,
                    "color": "gold",
                    "line": {"color": "black", "width": 1},
                },
                name=t.visualization.optimum_label,
                hovertemplate=(
                    f"{axes[0]} = {_format(sx)}<br>{axes[1]} = {_format(sy)}<extra></extra>"
                ),
            )
        )

    figure.update_layout(
        xaxis={"title": axes[0], "range": [x_min, x_max], "zeroline": True},
        yaxis={"title": axes[1], "range": [y_min, y_max], "zeroline": True},
        height=520,
        margin={"l": 40, "r": 20, "t": 70, "b": 40},
        # Above the plot: below it, the legend covered the x-axis title.
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    return figure


def _segment_in_box(a: float, b: float, c: float, box: Box) -> tuple[Point, Point] | None:
    """Where the line a*x + b*y = c crosses the box, or None if it misses it."""
    x_min, x_max, y_min, y_max = box
    points: list[Point] = []
    if abs(b) > _EPS:
        points += [(x, (c - a * x) / b) for x in (x_min, x_max)]
    if abs(a) > _EPS:
        points += [((c - b * y) / a, y) for y in (y_min, y_max)]
    inside = _dedupe(
        [
            p
            for p in points
            if x_min - 1e-9 <= p[0] <= x_max + 1e-9 and y_min - 1e-9 <= p[1] <= y_max + 1e-9
        ]
    )
    return (inside[0], inside[1]) if len(inside) >= 2 else None
