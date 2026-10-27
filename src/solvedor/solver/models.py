"""Core domain models for OR-Solver."""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum

import pydantic


class ObjectiveDirection(StrEnum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class ConstraintOperator(StrEnum):
    LESS_EQUAL = "<="
    GREATER_EQUAL = ">="
    EQUAL = "="


class VariableType(StrEnum):
    CONTINUOUS = "continuous"
    INTEGER = "integer"
    BINARY = "binary"


class SolverStatus(StrEnum):
    OPTIMAL = "optimal"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    NOT_SOLVED = "not_solved"
    ABNORMAL = "abnormal"


class Variable(pydantic.BaseModel):
    name: str
    variable_type: VariableType = VariableType.CONTINUOUS
    lower_bound: Decimal | None = Decimal(0)  # None means unbounded below (a free variable)
    upper_bound: Decimal | None = None  # None means unbounded above

    @pydantic.field_validator("name")
    @classmethod
    def ensure_valid_variable_name(cls, variable_name: str) -> str:
        if not variable_name or not isinstance(variable_name, str):
            raise ValueError(f"Variable name must be a non-empty string: {variable_name}")

        if not variable_name.replace("_", "").isalnum():
            raise ValueError(f"Invalid variable name: {variable_name}")

        if variable_name[0].isdigit():
            raise ValueError(f"Variable name cannot start with a number: {variable_name}")

        return variable_name

    @property
    def is_bounded(self) -> bool:
        return self.lower_bound is not None or self.upper_bound is not None

    @property
    def is_non_negative(self) -> bool:
        return self.lower_bound is not None and self.lower_bound >= 0


class Term(pydantic.BaseModel):
    coefficient: Decimal
    variable: str

    def __str__(self) -> str:
        """String representation of the term."""
        if self.coefficient == 1:
            return self.variable
        elif self.coefficient == -1:
            return f"-{self.variable}"
        else:
            return f"{self.coefficient}*{self.variable}"


class LinearExpression(pydantic.BaseModel):
    """Represents a linear expression as a sum of terms."""

    terms: list[Term] = pydantic.Field(default_factory=list, description="List of terms")

    def add_term(self, coefficient: Decimal | float | int, variable_name: str) -> None:
        decimal_coefficient = Decimal(str(coefficient))

        for existing_term in self.terms:
            if existing_term.variable == variable_name:
                existing_term.coefficient += decimal_coefficient
                return

        self.terms.append(Term(coefficient=decimal_coefficient, variable=variable_name))

    def extract_variable_names(self) -> list[str]:
        return [term.variable for term in self.terms]

    @property
    def variable_coefficients_map(self) -> dict[str, Decimal]:
        return {term.variable: term.coefficient for term in self.terms}

    def __str__(self) -> str:
        if not self.terms:
            return "0"

        expression_parts = []

        for term_index, current_term in enumerate(self.terms):
            term_string = str(current_term)
            if term_index == 0:
                expression_parts.append(term_string)
            else:
                if current_term.coefficient >= 0:
                    expression_parts.append(f" + {term_string}")
                else:
                    expression_parts.append(f" - {str(current_term).lstrip('-')}")

        return "".join(expression_parts)


class Constraint(pydantic.BaseModel):
    """Represents a linear constraint."""

    expression: LinearExpression = pydantic.Field(..., description="Left-hand side expression")
    operator: ConstraintOperator = pydantic.Field(..., description="Constraint operator")
    rhs: Decimal = pydantic.Field(..., description="Right-hand side value")
    name: str | None = pydantic.Field(default=None, description="Constraint name")

    def __str__(self) -> str:
        """String representation of the constraint."""
        match self.operator:
            case ConstraintOperator.LESS_EQUAL:
                op_symbol = "<="
            case ConstraintOperator.GREATER_EQUAL:
                op_symbol = ">="
            case ConstraintOperator.EQUAL:
                op_symbol = "="

        constraint_str = f"{self.expression} {op_symbol} {self.rhs}"
        if self.name:
            return f"{self.name}: {constraint_str}"
        return constraint_str

    def extract_variable_names(self) -> list[str]:
        return self.expression.extract_variable_names()


class ObjectiveFunction(pydantic.BaseModel):
    """Represents the objective function."""

    direction: ObjectiveDirection = pydantic.Field(..., description="Optimization direction")
    expression: LinearExpression = pydantic.Field(..., description="Objective expression")

    def __str__(self) -> str:
        return f"{self.direction.value} {self.expression}"

    def extract_variable_names(self) -> list[str]:
        return self.expression.extract_variable_names()


class Problem(pydantic.BaseModel):
    """Represents a complete linear programming problem."""

    objective: ObjectiveFunction = pydantic.Field(..., description="Objective function")
    constraints: list[Constraint] = pydantic.Field(
        default_factory=list, description="Problem constraints"
    )
    variables: dict[str, Variable] = pydantic.Field(
        default_factory=dict, description="Variable definitions"
    )
    name: str | None = pydantic.Field(default=None, description="Problem name")

    def add_constraint(self, new_constraint: Constraint) -> None:
        self.constraints.append(new_constraint)

        constraint_variable_names = new_constraint.extract_variable_names()
        for variable_name in constraint_variable_names:
            if variable_name not in self.variables:
                self.variables[variable_name] = Variable(name=variable_name)

    def add_variable(self, variable: Variable) -> None:
        """Add or update a variable definition."""
        self.variables[variable.name] = variable

    def extract_all_variable_names(self) -> list[str]:
        all_variable_names = set()

        objective_variables = self.objective.extract_variable_names()
        all_variable_names.update(objective_variables)

        for constraint in self.constraints:
            constraint_variables = constraint.extract_variable_names()
            all_variable_names.update(constraint_variables)

        return sorted(all_variable_names)

    @property
    def is_mip(self) -> bool:
        """True when any variable is integer or binary (solved with SCIP instead of GLOP)."""
        return any(v.variable_type is not VariableType.CONTINUOUS for v in self.variables.values())

    def __str__(self) -> str:
        """The model in the same syntax the parser reads, so it can be pasted back."""
        lines = [str(self.objective), ""]
        if self.constraints:
            lines.append("subject to:")
            lines.extend(f"    {constraint}" for constraint in self.constraints)

        non_negative: list[str] = []
        bounds: list[str] = []
        free: list[str] = []
        integer: list[str] = []
        binary: list[str] = []
        for name in self.extract_all_variable_names():
            variable = self.variables.get(name, Variable(name=name))
            low, high = variable.lower_bound, variable.upper_bound
            if variable.variable_type is VariableType.BINARY:
                binary.append(name)
                continue
            if variable.variable_type is VariableType.INTEGER:
                integer.append(name)
            if low is None:
                free.append(name)
                if high is not None:
                    bounds.append(f"{name} <= {high}")
            elif high is None:
                if low == 0:
                    non_negative.append(name)
                else:
                    bounds.append(f"{name} >= {low}")
            else:
                bounds.append(f"{low} <= {name} <= {high}")

        where: list[str] = []
        if non_negative:
            where.append(f"{', '.join(non_negative)} >= 0")
        where.extend(bounds)
        if free:
            where.append(f"free {', '.join(free)}")
        if integer:
            where.append(f"integer {', '.join(integer)}")
        if binary:
            where.append(f"binary {', '.join(binary)}")
        if where:
            lines.extend(["", "where:"])
            lines.extend(f"    {item}" for item in where)
        return "\n".join(lines)


class Solution(pydantic.BaseModel):
    """Represents a solution to an optimization problem."""

    status: SolverStatus = pydantic.Field(..., description="Solver status")
    objective_value: Decimal | None = pydantic.Field(
        default=None, description="Optimal objective value"
    )
    variable_values: dict[str, float] = pydantic.Field(
        default_factory=dict, description="Variable values in the solution"
    )
    solver_message: str | None = pydantic.Field(
        default=None, description="Optional solver message or error details"
    )
    solve_time: float | None = pydantic.Field(
        default=None, description="Time taken to solve (in seconds)"
    )
    iterations: int | None = pydantic.Field(default=None, description="Number of iterations")

    @property
    def is_optimal(self) -> bool:
        """Check if solution is optimal."""
        return self.status == SolverStatus.OPTIMAL

    @property
    def is_feasible(self) -> bool:
        """Check if solution is feasible."""
        return self.status in [SolverStatus.OPTIMAL]

    def extract_variable_value(self, variable_name: str) -> float | None:
        return self.variable_values.get(variable_name)

    def __str__(self) -> str:
        solution_display_lines = []

        if self.status == SolverStatus.OPTIMAL:
            solution_display_lines.append("🎯 Optimal Solution Found!")
            if self.objective_value is not None:
                solution_display_lines.append(f"   Objective Value: {self.objective_value}")
            solution_display_lines.append("")
            solution_display_lines.append("   Variables:")
            for variable_name, variable_value in sorted(self.variable_values.items()):
                solution_display_lines.append(f"   🐾 {variable_name} = {variable_value}")
            solution_display_lines.append("")
            solution_display_lines.append("   Status: OPTIMAL 😺")
        elif self.status == SolverStatus.INFEASIBLE:
            solution_display_lines.append("🙀 Problem is Infeasible!")
            solution_display_lines.append("   No solution exists that satisfies all constraints.")
        elif self.status == SolverStatus.UNBOUNDED:
            solution_display_lines.append("😿 Problem is Unbounded!")
            solution_display_lines.append("   The objective function can be improved infinitely.")
        else:
            solution_display_lines.append(f"😾 Solver Status: {self.status}")

        if self.solve_time is not None:
            solution_display_lines.append(f"   ⏱️ Solve Time: {self.solve_time:.3f}s")

        return "\n".join(solution_display_lines)


def add_term_to_linear_expression(
    target_expression: LinearExpression,
    coefficient: Decimal | float | int,
    variable_name: str,
) -> None:
    target_expression.add_term(coefficient, variable_name)


def extract_variables_from_expression(source_expression: LinearExpression) -> list[str]:
    return source_expression.extract_variable_names()
