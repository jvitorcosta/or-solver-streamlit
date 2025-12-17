"""Tests for solver functions."""

from solver.engine import solve_optimization_problem
from solver.models import Problem, SolverStatus


def test_linear_problem():
    """Test solving a basic linear programming problem."""
    problem_text = """
    maximize 3*x + 2*y

    subject to:
        x + 2*y <= 4
        2*x + y <= 4

    where:
        x, y >= 0
    """

    problem, solution = solve_optimization_problem(problem_text)

    assert isinstance(problem, Problem)
    assert len(problem.variables) == 2
    assert solution.status == SolverStatus.OPTIMAL
    assert solution.objective_value is not None
    assert "x" in solution.variable_values
    assert "y" in solution.variable_values


def test_integer_problem():
    """Test solving an integer programming problem."""
    problem_text = """
    maximize 3*x + 2*y

    subject to:
        x + 2*y <= 4
        2*x + y <= 4

    where:
        x, y >= 0
        integer x, y
    """

    problem, solution = solve_optimization_problem(problem_text)

    assert len(problem.variables) == 2
    assert solution.status == SolverStatus.OPTIMAL
    # Values should be integers
    for value in solution.variable_values.values():
        assert abs(value - round(value)) < 1e-6


def test_minimize_problem():
    """Test solving a minimization problem."""
    problem_text = """
    minimize x + y

    subject to:
        x + y >= 2
        x >= 1

    where:
        x, y >= 0
    """

    problem, solution = solve_optimization_problem(problem_text)

    assert solution.status == SolverStatus.OPTIMAL
    assert solution.objective_value is not None
    assert float(solution.objective_value) >= 2  # Should be at least 2
