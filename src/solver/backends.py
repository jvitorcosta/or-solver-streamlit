"""Solver backend implementations using Google OR-Tools."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ortools.linear_solver import pywraplp

from solver.models import (
    Constraint,
    ConstraintOperator,
    Problem,
    Solution,
    SolverStatus,
    VariableType,
)


class SolverBackend(ABC):
    """Abstract base class for optimization solver backends."""

    @abstractmethod
    def solve(self, problem: Problem, **kwargs) -> Solution:
        """Solve the optimization problem."""
        pass

    @abstractmethod
    def supports_problem_type(self, problem: Problem) -> bool:
        """Check if this backend can solve the given problem type."""
        pass

    @abstractmethod
    def get_solver_name(self) -> str:
        """Get the name of this solver backend."""
        pass


class OrToolsBackend(SolverBackend):
    """OR-Tools solver backend for linear and mixed-integer programming."""

    def __init__(self, solver_name: str = "GLOP"):
        """Initialize with specific OR-Tools solver.

        Args:
            solver_name: OR-Tools solver name (GLOP, SCIP, CP_SAT, etc.)
        """
        self.solver_name = solver_name

    def get_solver_name(self) -> str:
        """Get the solver name."""
        return f"OR-Tools {self.solver_name}"

    def supports_problem_type(self, problem: Problem) -> bool:
        """Check if this backend supports the problem type."""
        has_integer_vars = any(
            var.var_type in (VariableType.INTEGER, VariableType.BINARY)
            for var in problem.variables.values()
        )

        # GLOP only supports continuous variables
        if self.solver_name == "GLOP":
            return not has_integer_vars

        # SCIP supports mixed-integer problems
        return self.solver_name == "SCIP"

    def solve(self, problem: Problem, **kwargs) -> Solution:
        """Solve the problem using OR-Tools."""
        start_time = time.time()

        # Create the solver
        solver = pywraplp.Solver.CreateSolver(self.solver_name)
        if not solver:
            raise ValueError(f"Could not create solver: {self.solver_name}")

        try:
            # Create variables
            or_vars = {}
            for var_name, var in problem.variables.items():
                if var.var_type == VariableType.BINARY:
                    or_var = solver.BoolVar(var_name)
                elif var.var_type == VariableType.INTEGER:
                    # Use large bounds if not specified
                    lb = (
                        float(var.lower_bound)
                        if var.lower_bound is not None
                        else -solver.infinity()
                    )
                    ub = (
                        float(var.upper_bound)
                        if var.upper_bound is not None
                        else solver.infinity()
                    )
                    or_var = solver.IntVar(lb, ub, var_name)
                else:  # CONTINUOUS
                    lb = (
                        float(var.lower_bound)
                        if var.lower_bound is not None
                        else -solver.infinity()
                    )
                    ub = (
                        float(var.upper_bound)
                        if var.upper_bound is not None
                        else solver.infinity()
                    )
                    or_var = solver.NumVar(lb, ub, var_name)

                or_vars[var_name] = or_var

            # Add constraints
            for constraint in problem.constraints:
                self._add_constraint(solver, or_vars, constraint)

            # Set objective
            objective = solver.Objective()
            for var_name, coeff in problem.objective.expression.coefficients.items():
                objective.SetCoefficient(or_vars[var_name], float(coeff))

            if problem.objective.direction.value in ("maximize", "maximizar"):
                objective.SetMaximization()
            else:
                objective.SetMinimization()

            # Solve
            status = solver.Solve()

            # Extract solution
            return self._extract_solution(solver, or_vars, status, start_time)

        except Exception as e:
            solve_time = time.time() - start_time
            return Solution(
                status=SolverStatus.ABNORMAL,
                objective_value=None,
                variable_values={},
                solve_time=solve_time,
                solver_message=f"Error during solving: {str(e)}",
            )

    def _add_constraint(self, solver, or_vars: Dict[str, Any], constraint: Constraint):
        """Add a constraint to the OR-Tools solver."""
        # Build linear expression
        expr = solver.Sum(
            [
                float(coeff) * or_vars[var_name]
                for var_name, coeff in constraint.expression.coefficients.items()
                if var_name in or_vars
            ]
        )

        rhs = float(constraint.rhs)

        if constraint.operator == ConstraintOperator.LESS_EQUAL:
            solver.Add(expr <= rhs)
        elif constraint.operator == ConstraintOperator.GREATER_EQUAL:
            solver.Add(expr >= rhs)
        elif constraint.operator == ConstraintOperator.EQUAL:
            solver.Add(expr == rhs)

    def _extract_solution(
        self, solver, or_vars: Dict[str, Any], status, start_time: float
    ) -> Solution:
        """Extract solution from OR-Tools solver."""
        solve_time = time.time() - start_time

        # Map OR-Tools status to our enum
        if status == pywraplp.Solver.OPTIMAL:
            solver_status = SolverStatus.OPTIMAL
        elif status == pywraplp.Solver.INFEASIBLE:
            solver_status = SolverStatus.INFEASIBLE
        elif status == pywraplp.Solver.UNBOUNDED:
            solver_status = SolverStatus.UNBOUNDED
        else:
            solver_status = SolverStatus.ABNORMAL

        # Extract variable values
        variable_values = {}
        objective_value = None

        if solver_status == SolverStatus.OPTIMAL:
            for var_name, or_var in or_vars.items():
                variable_values[var_name] = or_var.solution_value()
            objective_value = solver.Objective().Value()

        return Solution(
            status=solver_status,
            objective_value=objective_value,
            variable_values=variable_values,
            solve_time=solve_time,
            iterations=solver.iterations() if hasattr(solver, "iterations") else None,
        )


class SolverFactory:
    """Factory for creating appropriate solver backends."""

    @staticmethod
    def create_solver(
        problem: Problem, solver_preference: Optional[str] = None
    ) -> OrToolsBackend:
        """Create the best solver for the given problem.

        Args:
            problem: The optimization problem to solve
            solver_preference: Optional solver preference ('linear', 'integer',
                or specific solver name)
        """
        has_integer_vars = any(
            var.var_type in (VariableType.INTEGER, VariableType.BINARY)
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
        if has_integer_vars:
            return OrToolsBackend("SCIP")  # Mixed-integer programming
        else:
            return OrToolsBackend("GLOP")  # Linear programming

    @staticmethod
    def get_available_solvers() -> Dict[str, str]:
        """Get available solvers and their descriptions."""
        return {
            "GLOP": "Google Linear Optimization Package (Linear Programming only)",
            "SCIP": "Solving Constraint Integer Programs (Mixed-Integer Programming)",
        }
