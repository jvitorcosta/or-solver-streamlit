"""Solver backend implementations using Google OR-Tools."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ortools.linear_solver import pywraplp

from solver.models import (
    Constraint,
    ConstraintOperator,
    Problem,
    Solution,
    SolverStatus,
    VariableType,
)


@dataclass
class SolverConfig:
    """Configuration for OR-Tools solver backend."""

    solver_name: str = "GLOP"

    @property
    def display_name(self) -> str:
        """Get the display name for this solver."""
        return f"OR-Tools {self.solver_name}"


def check_solver_compatibility_with_problem(
    config: SolverConfig, problem: Problem
) -> bool:
    """Check if solver configuration is compatible with problem variable types."""
    contains_integer_or_binary_variables = any(
        var.variable_type in (VariableType.INTEGER, VariableType.BINARY)
        for var in problem.variables.values()
    )

    # GLOP only supports continuous variables
    if config.solver_name == "GLOP":
        return not contains_integer_or_binary_variables

    # SCIP supports mixed-integer problems
    return config.solver_name == "SCIP"


class OrToolsBackend:
    """OR-Tools solver backend for linear and mixed-integer programming."""

    def __init__(self, solver_name: str = "GLOP") -> None:
        """Initialize with specific OR-Tools solver.

        Args:
            solver_name: OR-Tools solver name (GLOP, SCIP, CP_SAT, etc.)
        """
        self.config = SolverConfig(solver_name=solver_name)

    def get_solver_name(self) -> str:
        """Get the solver name."""
        return self.config.display_name

    def supports_problem_type(self, problem: Problem) -> bool:
        """Check if this backend supports the problem type."""
        return check_solver_compatibility_with_problem(self.config, problem)

    def execute_optimization_with_backend(self, problem: Problem, **kwargs) -> Solution:
        """Execute optimization problem using OR-Tools backend."""
        return execute_problem_with_ortools_backend(self.config, problem, **kwargs)


def select_optimal_solver_for_problem(
    problem: Problem, solver_preference: str | None = None
) -> OrToolsBackend:
    """Select most suitable solver based on problem characteristics.

    Args:
        problem: The optimization problem to solve
        solver_preference: Optional solver preference ('linear', 'integer',
            or specific solver name)
    """
    problem_contains_integer_variables = any(
        var.variable_type in (VariableType.INTEGER, VariableType.BINARY)
        for var in problem.variables.values()
    )

    # Use explicit preference if provided
    if solver_preference and solver_preference.upper() in ("GLOP", "SCIP"):
        backend = OrToolsBackend(solver_preference.upper())
        if backend.supports_problem_type(problem):
            return backend
        else:
            raise ValueError(
                f"Solver {solver_preference} does not support this problem type"
            )

    # Auto-select based on problem type
    if problem_contains_integer_variables:
        return OrToolsBackend("SCIP")  # Mixed-integer programming
    else:
        return OrToolsBackend("GLOP")  # Linear programming


def enumerate_supported_solver_backends() -> dict[str, str]:
    """Enumerate all supported solver backends with descriptions."""
    return {
        "GLOP": "Google Linear Optimization Package (Linear Programming only)",
        "SCIP": "Solving Constraint Integer Programs (Mixed-Integer Programming)",
    }


class SolverFactory:
    """Factory for creating appropriate solver backends."""

    @staticmethod
    def create_solver(
        problem: Problem, solver_preference: str | None = None
    ) -> OrToolsBackend:
        """Create the best solver for the given problem."""
        return select_optimal_solver_for_problem(problem, solver_preference)

    @staticmethod
    def get_available_solvers() -> dict[str, str]:
        """Get available solvers and their descriptions."""
        return enumerate_supported_solver_backends()


def execute_problem_with_ortools_backend(
    config: SolverConfig, problem: Problem, **kwargs
) -> Solution:
    """Execute optimization using OR-Tools backend with specific configuration."""
    start_time = time.time()

    # Create the solver
    solver = pywraplp.Solver.CreateSolver(config.solver_name)
    if not solver:
        raise ValueError(f"Could not create solver: {config.solver_name}")

    try:
        # Create variables
        solver_variable_map = _initialize_decision_variables_in_solver(solver, problem)

        # Add constraints
        for constraint in problem.constraints:
            _apply_constraint_to_solver(solver, solver_variable_map, constraint)

        # Set objective
        _configure_optimization_objective(
            solver, solver_variable_map, problem.objective
        )

        # Solve
        status = solver.Solve()

        # Extract solution
        return _build_solution_from_solver_results(
            solver, solver_variable_map, status, start_time
        )

    except Exception as e:
        solve_time = time.time() - start_time
        return Solution(
            status=SolverStatus.ABNORMAL,
            objective_value=None,
            variable_values={},
            solve_time=solve_time,
            solver_message=f"Error during solving: {str(e)}",
        )


def _initialize_decision_variables_in_solver(
    solver: pywraplp.Solver, problem: Problem
) -> dict[str, Any]:
    """Initialize decision variables in OR-Tools solver based on problem definition."""
    solver_variable_mapping = {}
    for variable_name, variable_definition in problem.variables.items():
        if variable_definition.variable_type == VariableType.BINARY:
            solver_variable = solver.BoolVar(variable_name)
        elif variable_definition.variable_type == VariableType.INTEGER:
            # Use large bounds if not specified
            variable_lower_bound = (
                float(variable_definition.lower_bound)
                if variable_definition.lower_bound is not None
                else -solver.infinity()
            )
            variable_upper_bound = (
                float(variable_definition.upper_bound)
                if variable_definition.upper_bound is not None
                else solver.infinity()
            )
            solver_variable = solver.IntVar(
                variable_lower_bound, variable_upper_bound, variable_name
            )
        else:  # CONTINUOUS
            variable_lower_bound = (
                float(variable_definition.lower_bound)
                if variable_definition.lower_bound is not None
                else -solver.infinity()
            )
            variable_upper_bound = (
                float(variable_definition.upper_bound)
                if variable_definition.upper_bound is not None
                else solver.infinity()
            )
            solver_variable = solver.NumVar(
                variable_lower_bound, variable_upper_bound, variable_name
            )

        solver_variable_mapping[variable_name] = solver_variable

    return solver_variable_mapping


def _apply_constraint_to_solver(
    solver: pywraplp.Solver, solver_variables: dict[str, Any], constraint: Constraint
) -> None:
    """Apply linear constraint to OR-Tools solver with proper bounds."""
    # Build linear expression
    constraint_left_hand_side = solver.Sum(
        [
            float(coeff) * solver_variables[var_name]
            for var_name, coeff in constraint.expression.variable_coefficients_map.items()
            if var_name in solver_variables
        ]
    )

    constraint_right_hand_side = float(constraint.rhs)

    if constraint.operator == ConstraintOperator.LESS_EQUAL:
        solver.Add(constraint_left_hand_side <= constraint_right_hand_side)
    elif constraint.operator == ConstraintOperator.GREATER_EQUAL:
        solver.Add(constraint_left_hand_side >= constraint_right_hand_side)
    elif constraint.operator == ConstraintOperator.EQUAL:
        solver.Add(constraint_left_hand_side == constraint_right_hand_side)


def _configure_optimization_objective(
    solver: pywraplp.Solver, solver_variables: dict[str, Any], objective: Any
) -> None:
    """Configure objective function coefficients and optimization direction."""
    objective_function = solver.Objective()
    for (
        variable_name,
        coefficient_value,
    ) in objective.expression.variable_coefficients_map.items():
        if variable_name in solver_variables:
            objective_function.SetCoefficient(
                solver_variables[variable_name], float(coefficient_value)
            )

    if objective.direction.value in ("maximize", "maximizar"):
        objective_function.SetMaximization()
    else:
        objective_function.SetMinimization()


def _build_solution_from_solver_results(
    solver: pywraplp.Solver,
    solver_variables: dict[str, Any],
    solver_status_code: int,
    optimization_start_time: float,
) -> Solution:
    """Build solution object from OR-Tools solver results and variable values."""
    total_optimization_time = time.time() - optimization_start_time

    # Map OR-Tools status to our enum
    if solver_status_code == pywraplp.Solver.OPTIMAL:
        solution_status = SolverStatus.OPTIMAL
    elif solver_status_code == pywraplp.Solver.INFEASIBLE:
        solution_status = SolverStatus.INFEASIBLE
    elif solver_status_code == pywraplp.Solver.UNBOUNDED:
        solution_status = SolverStatus.UNBOUNDED
    else:
        solution_status = SolverStatus.ABNORMAL

    # Extract variable values
    extracted_variable_values = {}
    optimal_objective_value = None

    if solution_status == SolverStatus.OPTIMAL:
        for variable_name, solver_variable in solver_variables.items():
            extracted_variable_values[variable_name] = solver_variable.solution_value()
        optimal_objective_value = solver.Objective().Value()

    return Solution(
        status=solution_status,
        objective_value=optimal_objective_value,
        variable_values=extracted_variable_values,
        solve_time=total_optimization_time,
        iterations=solver.iterations() if hasattr(solver, "iterations") else None,
    )
