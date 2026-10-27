import random
import time
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from solvedor.solver.backends import solve
from solvedor.solver.models import (
    Constraint,
    ConstraintOperator,
    LinearExpression,
    ObjectiveDirection,
    ObjectiveFunction,
    Problem,
    Solution,
    SolverStatus,
    Variable,
    VariableType,
)
from solvedor.solver.parser import parse_lp_problem

_EXAMPLES = Path(__file__).parents[1] / "src" / "solvedor" / "resources" / "examples.yaml"


def _solve(text: str) -> Solution:
    return solve(parse_lp_problem(text))


def test_textbook_lp_optimum() -> None:
    solution = _solve("max 3x + 2y\nsubject to:\nx + 2y <= 4\n2x + y <= 4\n")
    assert solution.status is SolverStatus.OPTIMAL
    assert solution.solver_name == "GLOP"
    assert solution.objective_value == pytest.approx(20 / 3, abs=1e-6)
    assert solution.variable_values["x"] == pytest.approx(4 / 3, abs=1e-6)
    assert solution.variable_values["y"] == pytest.approx(4 / 3, abs=1e-6)


def test_unbounded_lp() -> None:
    assert _solve("max x + y\nsubject to:\nx - y <= 1\n").status is SolverStatus.UNBOUNDED


def test_objective_only_var_unbounded() -> None:
    assert _solve("max x + y\nsubject to:\nx <= 5\n").status is SolverStatus.UNBOUNDED


def test_infeasible_lp() -> None:
    solution = _solve("max x\nsubject to:\nx <= 1\nx >= 2\n")
    assert solution.status is SolverStatus.INFEASIBLE
    assert solution.objective_value is None
    assert solution.variable_values == {}


def test_infeasible_mip() -> None:
    solution = _solve("max x\nsubject to:\nx <= 10\nwhere:\n1.2 <= x <= 1.8\ninteger x\n")
    assert solution.status is SolverStatus.INFEASIBLE
    assert solution.solver_name == "SCIP"


def test_free_variable_goes_negative() -> None:
    solution = _solve("min y\nsubject to:\ny >= -3\nwhere:\nfree y\n")
    assert solution.objective_value == pytest.approx(-3)


def test_equality_constraint() -> None:
    solution = _solve("min x + y\nsubject to:\nx + y = 3\nx >= 1\n")
    assert solution.status is SolverStatus.OPTIMAL
    assert solution.objective_value == pytest.approx(3)


def test_binary_knapsack() -> None:
    # Values 6, 10, 12 with weights 1, 2, 3 and capacity 5: take items b and c.
    solution = _solve("max 6a + 10b + 12c\nsubject to:\na + 2b + 3c <= 5\nwhere:\nbinary a, b, c\n")
    assert solution.status is SolverStatus.OPTIMAL
    assert solution.objective_value == pytest.approx(22)
    assert [round(solution.variable_values[v]) for v in "abc"] == [0, 1, 1]


def test_integer_values_are_whole() -> None:
    solution = _solve("max 3x + 2y\nsubject to:\nx + 2y <= 4\n2x + y <= 4\nwhere:\ninteger x, y\n")
    assert solution.objective_value == pytest.approx(6)
    assert all(abs(value - round(value)) < 1e-6 for value in solution.variable_values.values())


def test_zero_objective_value() -> None:
    solution = _solve("min x\nsubject to:\nx <= 4\n")
    assert solution.status is SolverStatus.OPTIMAL
    assert solution.objective_value == 0.0


def _hard_knapsack(size: int, rows: int, seed: int = 7) -> Problem:
    """A multi-dimensional integer knapsack SCIP cannot prove optimal in under a second."""
    rng = random.Random(seed)  # noqa: S311 - reproducible test data, not crypto
    names = [f"x{i}" for i in range(size)]
    objective = LinearExpression()
    for name in names:
        objective.add_term(rng.randint(10, 100), name)
    problem = Problem(
        objective=ObjectiveFunction(direction=ObjectiveDirection.MAXIMIZE, expression=objective)
    )
    for _ in range(rows):
        row = LinearExpression()
        for name in names:
            row.add_term(rng.randint(5, 60), name)
        problem.add_constraint(
            Constraint(
                expression=row, operator=ConstraintOperator.LESS_EQUAL, rhs=Decimal(size * 8)
            )
        )
    problem.variables = {
        name: Variable(name=name, variable_type=VariableType.INTEGER, upper_bound=Decimal(3))
        for name in names
    }
    return problem


def test_time_limit_returns() -> None:
    start = time.perf_counter()
    solution = solve(_hard_knapsack(size=150, rows=40), time_limit_s=0.5)
    assert time.perf_counter() - start < 3
    assert solution.status in (SolverStatus.FEASIBLE, SolverStatus.TIME_LIMIT)


def test_unknown_variable_is_an_error_not_dropped() -> None:
    problem = parse_lp_problem("max x + y\nsubject to:\nx + y <= 4\n")
    del problem.variables["y"]
    solution = solve(problem)
    assert solution.status is SolverStatus.ERROR
    assert "'y'" in (solution.solver_message or "")


# Objectives the gallery produced before the solver rewrite (recorded at 5325e1e).
_GALLERY_OBJECTIVES = {
    "production_planning": 1800.0,
    "transportation": 4500.0,
    "diet_optimization": 520.0,
    "resource_allocation": 148500.0,
    "facility_location": 95000.0,
}


def test_gallery_objectives_unchanged() -> None:
    examples = yaml.safe_load(_EXAMPLES.read_text(encoding="utf-8"))
    assert set(examples) == set(_GALLERY_OBJECTIVES)
    for key, example in examples.items():
        solution = _solve(example["problem"])
        assert solution.status is SolverStatus.OPTIMAL, key
        assert solution.objective_value == pytest.approx(_GALLERY_OBJECTIVES[key]), key


def test_objective_constant_is_in_the_value() -> None:
    assert _solve("max x + 5\nsubject to:\nx <= 3\n").objective_value == pytest.approx(8)
