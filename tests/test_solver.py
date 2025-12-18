"""Tests for solver functions."""

from solver.engine import parse_text_and_solve_with_backend
from solver.models import Problem, SolverStatus


def test_solve_continuous_linear_program_optimally():
    """Test solving continuous linear programming problem to optimal solution."""
    linear_program_text = """
    maximize 3*x + 2*y

    subject to:
        x + 2*y <= 4
        2*x + y <= 4

    where:
        x, y >= 0
    """

    parsed_linear_problem, optimal_solution = parse_text_and_solve_with_backend(
        linear_program_text
    )

    assert isinstance(parsed_linear_problem, Problem)
    assert len(parsed_linear_problem.variables) == 2
    assert optimal_solution.status == SolverStatus.OPTIMAL
    assert optimal_solution.objective_value is not None
    assert "x" in optimal_solution.variable_values
    assert "y" in optimal_solution.variable_values


def test_solve_integer_program_with_discrete_variables():
    """Test solving integer programming problem with discrete variable constraints."""
    integer_program_text = """
    maximize 3*x + 2*y

    subject to:
        x + 2*y <= 4
        2*x + y <= 4

    where:
        x, y >= 0
        integer x, y
    """

    parsed_integer_problem, integer_solution = parse_text_and_solve_with_backend(
        integer_program_text
    )

    assert len(parsed_integer_problem.variables) == 2
    assert integer_solution.status == SolverStatus.OPTIMAL
    # Values should be integers
    for variable_value in integer_solution.variable_values.values():
        assert abs(variable_value - round(variable_value)) < 1e-6


def test_solve_minimization_problem_successfully():
    """Test solving minimization problem to find lowest objective value."""
    minimization_problem_text = """
    minimize x + y

    subject to:
        x + y >= 2
        x >= 1

    where:
        x, y >= 0
    """

    parsed_minimization_problem, minimization_solution = (
        parse_text_and_solve_with_backend(minimization_problem_text)
    )

    assert minimization_solution.status == SolverStatus.OPTIMAL
    assert minimization_solution.objective_value is not None
    assert float(minimization_solution.objective_value) >= 2  # Should be at least 2
