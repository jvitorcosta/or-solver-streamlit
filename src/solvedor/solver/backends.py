"""Solve a Problem with Google OR-Tools: GLOP for linear programs, SCIP when any variable is
integer or binary."""

from __future__ import annotations

import time

from ortools.linear_solver import pywraplp

from solvedor.solver.models import (
    ConstraintOperator,
    ObjectiveDirection,
    Problem,
    Solution,
    SolverStatus,
    VariableType,
)

DEFAULT_TIME_LIMIT_S = 10.0

_STATUSES = {
    pywraplp.Solver.OPTIMAL: SolverStatus.OPTIMAL,
    pywraplp.Solver.FEASIBLE: SolverStatus.FEASIBLE,  # SCIP hit the time limit with a solution
    pywraplp.Solver.INFEASIBLE: SolverStatus.INFEASIBLE,
    pywraplp.Solver.UNBOUNDED: SolverStatus.UNBOUNDED,
    pywraplp.Solver.NOT_SOLVED: SolverStatus.TIME_LIMIT,  # stopped before finding any solution
}


def solve(problem: Problem, time_limit_s: float = DEFAULT_TIME_LIMIT_S) -> Solution:
    """Solve the problem. Never raises for solver trouble; that comes back as status ERROR."""
    start = time.perf_counter()
    solver_name = "SCIP" if problem.is_mip else "GLOP"
    try:
        solver, status = _build_and_solve(problem, solver_name, time_limit_s, presolve=True)
        if status == pywraplp.Solver.INFEASIBLE and solver_name == "GLOP":
            # GLOP's presolve reports unbounded models as infeasible; without it the simplex
            # tells the two apart.
            solver, status = _build_and_solve(problem, solver_name, time_limit_s, presolve=False)
    except Exception as e:  # OR-Tools and bad models: report, never crash the app
        return Solution(
            status=SolverStatus.ERROR,
            solver_message=str(e),
            solve_time=time.perf_counter() - start,
        )

    solution = Solution(
        status=_STATUSES.get(status, SolverStatus.ERROR),
        solve_time=time.perf_counter() - start,
        iterations=solver.iterations(),
        solver_name=solver_name,
    )
    if solution.is_feasible:
        solution.objective_value = solver.Objective().Value()
        solution.variable_values = {
            variable.name(): variable.solution_value() for variable in solver.variables()
        }
    return solution


def _build_and_solve(
    problem: Problem, solver_name: str, time_limit_s: float, *, presolve: bool
) -> tuple[pywraplp.Solver, int]:
    solver = pywraplp.Solver.CreateSolver(solver_name)
    if solver is None:
        raise RuntimeError(f"OR-Tools could not create the {solver_name} solver")
    solver.SetTimeLimit(int(time_limit_s * 1000))
    infinity = solver.infinity()

    variables = {}
    for name in problem.extract_all_variable_names():
        if name not in problem.variables:
            raise ValueError(f"variable '{name}' is used but not defined in problem.variables")
    for name, variable in problem.variables.items():
        low = float(variable.lower_bound) if variable.lower_bound is not None else -infinity
        high = float(variable.upper_bound) if variable.upper_bound is not None else infinity
        is_integer = variable.variable_type is not VariableType.CONTINUOUS
        variables[name] = solver.Var(low, high, is_integer, name)

    for constraint in problem.constraints:
        lhs = sum(
            float(term.coefficient) * variables[term.variable]
            for term in constraint.expression.terms
        )
        rhs = float(constraint.rhs)
        match constraint.operator:
            case ConstraintOperator.LESS_EQUAL:
                solver.Add(lhs <= rhs)
            case ConstraintOperator.GREATER_EQUAL:
                solver.Add(lhs >= rhs)
            case ConstraintOperator.EQUAL:
                solver.Add(lhs == rhs)

    objective = solver.Objective()
    for term in problem.objective.expression.terms:
        objective.SetCoefficient(variables[term.variable], float(term.coefficient))
    if problem.objective.direction is ObjectiveDirection.MAXIMIZE:
        objective.SetMaximization()
    else:
        objective.SetMinimization()

    parameters = pywraplp.MPSolverParameters()
    if not presolve:
        parameters.SetIntegerParam(
            pywraplp.MPSolverParameters.PRESOLVE, pywraplp.MPSolverParameters.PRESOLVE_OFF
        )
    return solver, solver.Solve(parameters)
