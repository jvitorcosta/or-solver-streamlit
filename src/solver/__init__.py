"""Solver domain for linear programming problem solving."""

# Only import core models for tests, not the full engine
from solver.models import Constraint, ObjectiveFunction, Problem


# Import functions only when needed to avoid dependency issues
def solve_problem(*args, **kwargs):
    """Lazy import wrapper for solve_problem to avoid UI dependencies in tests."""
    from solver.engine import solve_problem as _solve_problem

    return _solve_problem(*args, **kwargs)


def parse_lp_problem(*args, **kwargs):
    """Lazy import wrapper for parse_lp_problem."""
    from solver.parser import parse_lp_problem as _parse_lp_problem

    return _parse_lp_problem(*args, **kwargs)


def ParseError():
    """Lazy import wrapper for ParseError."""
    from solver.parser import ParseError as _ParseError

    return _ParseError


__all__ = [
    "solve_problem",
    "Problem",
    "ObjectiveFunction",
    "Constraint",
    "parse_lp_problem",
    "ParseError",
]
