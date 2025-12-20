"""Core domain models for SolvedOR."""

from __future__ import annotations

from decimal import Decimal
from enum import Enum

import pydantic


class ObjectiveDirection(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"
    MINIMIZAR = "minimizar"  # Portuguese
    MAXIMIZAR = "maximizar"  # Portuguese


class ConstraintOperator(str, Enum):
    LESS_EQUAL = "<="
    GREATER_EQUAL = ">="
    EQUAL = "="


class VariableType(str, Enum):
    CONTINUOUS = "continuous"
    INTEGER = "integer"
    BINARY = "binary"


class SolverStatus(str, Enum):
    OPTIMAL = "optimal"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    NOT_SOLVED = "not_solved"
    ABNORMAL = "abnormal"


class Variable(pydantic.BaseModel):
    name: str
    variable_type: VariableType = VariableType.CONTINUOUS
    lower_bound: Decimal | None = None
    upper_bound: Decimal | None = None

    @pydantic.field_validator("name")
    @classmethod
    def ensure_valid_variable_name(cls, variable_name):
        if not variable_name or not isinstance(variable_name, str):
            raise ValueError(
                f"Variable name must be a non-empty string: {variable_name}"
            )

        if not variable_name.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"Invalid variable name: {variable_name}")

        if variable_name[0].isdigit():
            raise ValueError(
                f"Variable name cannot start with a number: {variable_name}"
            )

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

    terms: list[Term] = pydantic.Field(
        default_factory=list, description="List of terms"
    )
    constant: Decimal = pydantic.Field(
        default=Decimal("0"), description="Constant term"
    )

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
        if not self.terms and self.constant == 0:
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

        if self.constant != 0:
            if expression_parts:
                if self.constant > 0:
                    expression_parts.append(f" + {self.constant}")
                else:
                    expression_parts.append(f" - {abs(self.constant)}")
            else:
                expression_parts.append(str(self.constant))

        return "".join(expression_parts)


class Constraint(pydantic.BaseModel):
    """Represents a linear constraint."""

    expression: LinearExpression = pydantic.Field(
        ..., description="Left-hand side expression"
    )
    operator: ConstraintOperator = pydantic.Field(
        ..., description="Constraint operator"
    )
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

    direction: ObjectiveDirection = pydantic.Field(
        ..., description="Optimization direction"
    )
    expression: LinearExpression = pydantic.Field(
        ..., description="Objective expression"
    )

    def __str__(self) -> str:
        match self.direction:
            case ObjectiveDirection.MINIMIZAR:
                english_direction = "minimize"
            case ObjectiveDirection.MAXIMIZAR:
                english_direction = "maximize"
            case _:
                english_direction = self.direction.value

        return f"{english_direction} {self.expression}"

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

    def __str__(self) -> str:
        problem_lines = []

        problem_lines.append(str(self.objective))
        problem_lines.append("")

        if self.constraints:
            problem_lines.append("subject to:")
            for constraint in self.constraints:
                problem_lines.append(f"    {constraint}")

        non_negative_variable_names = []
        bounded_variable_constraints = []
        integer_variable_names = []
        binary_variable_names = []

        all_variables = self.extract_all_variable_names()
        for variable_name in all_variables:
            if variable_name in self.variables:
                variable_definition = self.variables[variable_name]

                if variable_definition.variable_type == VariableType.INTEGER:
                    integer_variable_names.append(variable_name)
                elif variable_definition.variable_type == VariableType.BINARY:
                    binary_variable_names.append(variable_name)

                if variable_definition.is_bounded:
                    if (
                        variable_definition.lower_bound == 0
                        and variable_definition.upper_bound is None
                    ):
                        non_negative_variable_names.append(variable_name)
                    else:
                        bound_constraint_parts = []
                        if variable_definition.lower_bound is not None:
                            bound_constraint_parts.append(
                                str(variable_definition.lower_bound)
                            )
                        bound_constraint_parts.append("<=")
                        bound_constraint_parts.append(variable_name)
                        if variable_definition.upper_bound is not None:
                            bound_constraint_parts.append("<=")
                            bound_constraint_parts.append(
                                str(variable_definition.upper_bound)
                            )
                        bounded_variable_constraints.append(
                            " ".join(bound_constraint_parts)
                        )
                else:
                    non_negative_variable_names.append(variable_name)
            else:
                non_negative_variable_names.append(variable_name)

        has_variable_constraints = any(
            [
                non_negative_variable_names,
                bounded_variable_constraints,
                integer_variable_names,
                binary_variable_names,
            ]
        )
        if has_variable_constraints:
            problem_lines.append("")
            problem_lines.append("where:")

            if non_negative_variable_names:
                problem_lines.append(
                    f"    {', '.join(non_negative_variable_names)} >= 0"
                )

            for bound_constraint in bounded_variable_constraints:
                problem_lines.append(f"    {bound_constraint}")

            if integer_variable_names:
                problem_lines.append(f"    integer {', '.join(integer_variable_names)}")

            if binary_variable_names:
                problem_lines.append(f"    binary {', '.join(binary_variable_names)}")

        return "\n".join(problem_lines)


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
    iterations: int | None = pydantic.Field(
        default=None, description="Number of iterations"
    )

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
                solution_display_lines.append(
                    f"   Objective Value: {self.objective_value}"
                )
            solution_display_lines.append("")
            solution_display_lines.append("   Variables:")
            for variable_name, variable_value in sorted(self.variable_values.items()):
                solution_display_lines.append(
                    f"   🐾 {variable_name} = {variable_value}"
                )
            solution_display_lines.append("")
            solution_display_lines.append("   Status: OPTIMAL 😺")
        elif self.status == SolverStatus.INFEASIBLE:
            solution_display_lines.append("🙀 Problem is Infeasible!")
            solution_display_lines.append(
                "   No solution exists that satisfies all constraints."
            )
        elif self.status == SolverStatus.UNBOUNDED:
            solution_display_lines.append("😿 Problem is Unbounded!")
            solution_display_lines.append(
                "   The objective function can be improved infinitely."
            )
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
