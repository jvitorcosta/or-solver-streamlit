"""Text in, answer out. No Streamlit here, so it can be tested and reused without the UI."""

from __future__ import annotations

from dataclasses import dataclass

from solvedor.solver.backends import DEFAULT_TIME_LIMIT_S, solve
from solvedor.solver.models import Problem, Solution
from solvedor.solver.parser import ParseError, parse_lp_problem


@dataclass(frozen=True)
class SolveResult:
    """Either a parsed problem with its solution, or the parse error that stopped it."""

    problem: Problem | None = None
    solution: Solution | None = None
    error: ParseError | None = None


def solve_text(text: str, time_limit_s: float = DEFAULT_TIME_LIMIT_S) -> SolveResult:
    """Parse and solve a problem written in the app's syntax."""
    try:
        problem = parse_lp_problem(text)
    except ParseError as e:
        return SolveResult(error=e)
    return SolveResult(problem=problem, solution=solve(problem, time_limit_s))
