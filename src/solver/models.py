"""Core domain models for OR-Solver."""

from __future__ import annotations

from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field, field_validator


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


class Variable(BaseModel):
    name: str
    var_type: VariableType = VariableType.CONTINUOUS
    lower_bound: Decimal | None = None
    upper_bound: Decimal | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError(f"Variable name must be a non-empty string: {v}")
        
        # Check for invalid characters (only allow alphanumeric, underscore)
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"Invalid variable name: {v}")
            
        # Check for names starting with numbers
        if v[0].isdigit():
            raise ValueError(f"Variable name cannot start with a number: {v}")
            
        return v

    @property
    def is_bounded(self) -> bool:
        return self.lower_bound is not None or self.upper_bound is not None

    @property
    def is_non_negative(self) -> bool:
        return self.lower_bound is not None and self.lower_bound >= 0


class Term(BaseModel):
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


class LinearExpression(BaseModel):
    """Represents a linear expression as a sum of terms."""

    terms: list[Term] = Field(default_factory=list, description="List of terms")
    constant: Decimal = Field(default=Decimal("0"), description="Constant term")

    def add_term(self, coefficient: Decimal | float | int, variable: str) -> None:
        """Add a term to the expression."""
        coeff = Decimal(str(coefficient))

        # Check if variable already exists and combine coefficients
        for term in self.terms:
            if term.variable == variable:
                term.coefficient += coeff
                return

        # Add new term
        self.terms.append(Term(coefficient=coeff, variable=variable))

    def get_variables(self) -> list[str]:
        """Get list of all variable names in the expression."""
        return [term.variable for term in self.terms]

    def __str__(self) -> str:
        """String representation of the expression."""
        if not self.terms and self.constant == 0:
            return "0"

        parts = []

        # Add terms
        for i, term in enumerate(self.terms):
            term_str = str(term)
            if i == 0:
                parts.append(term_str)
            else:
                if term.coefficient >= 0:
                    parts.append(f" + {term_str}")
                else:
                    parts.append(f" - {str(term).lstrip('-')}")

        # Add constant if non-zero
        if self.constant != 0:
            if parts:
                if self.constant > 0:
                    parts.append(f" + {self.constant}")
                else:
                    parts.append(f" - {abs(self.constant)}")
            else:
                parts.append(str(self.constant))

        return "".join(parts)


class Constraint(BaseModel):
    """Represents a linear constraint."""

    expression: LinearExpression = Field(..., description="Left-hand side expression")
    operator: ConstraintOperator = Field(..., description="Constraint operator")
    rhs: Decimal = Field(..., description="Right-hand side value")
    name: str | None = Field(default=None, description="Constraint name")

    def __str__(self) -> str:
        """String representation of the constraint."""
        op_symbol = {
            ConstraintOperator.LESS_EQUAL: "<=",
            ConstraintOperator.GREATER_EQUAL: ">=",
            ConstraintOperator.EQUAL: "=",
        }[self.operator]
        
        constraint_str = f"{self.expression} {op_symbol} {self.rhs}"
        if self.name:
            return f"{self.name}: {constraint_str}"
        return constraint_str

    def get_variables(self) -> list[str]:
        """Get list of all variables in the constraint."""
        return self.expression.get_variables()


class ObjectiveFunction(BaseModel):
    """Represents the objective function."""

    direction: ObjectiveDirection = Field(..., description="Optimization direction")
    expression: LinearExpression = Field(..., description="Objective expression")

    def __str__(self) -> str:
        """String representation of the objective."""
        direction_str = self.direction.value
        if direction_str in ["minimizar", "maximizar"]:
            # Use English for string representation
            direction_str = "minimize" if direction_str == "minimizar" else "maximize"

        return f"{direction_str} {self.expression}"

    def get_variables(self) -> list[str]:
        """Get list of all variables in the objective."""
        return self.expression.get_variables()


class Problem(BaseModel):
    """Represents a complete linear programming problem."""

    objective: ObjectiveFunction = Field(..., description="Objective function")
    constraints: list[Constraint] = Field(
        default_factory=list, description="Problem constraints"
    )
    variables: dict[str, Variable] = Field(
        default_factory=dict, description="Variable definitions"
    )
    name: str | None = Field(default=None, description="Problem name")

    def add_constraint(self, constraint: Constraint) -> None:
        """Add a constraint to the problem."""
        self.constraints.append(constraint)

        # Auto-register variables from constraint
        for var_name in constraint.get_variables():
            if var_name not in self.variables:
                self.variables[var_name] = Variable(name=var_name)

    def add_variable(self, variable: Variable) -> None:
        """Add or update a variable definition."""
        self.variables[variable.name] = variable

    def get_all_variables(self) -> list[str]:
        """Get list of all variable names used in the problem."""
        var_names = set()

        # Variables from objective
        var_names.update(self.objective.get_variables())

        # Variables from constraints
        for constraint in self.constraints:
            var_names.update(constraint.get_variables())

        return sorted(var_names)

    def __str__(self) -> str:
        """String representation of the problem."""
        lines = []

        # Objective
        lines.append(str(self.objective))
        lines.append("")

        # Constraints
        if self.constraints:
            lines.append("subject to:")
            for constraint in self.constraints:
                lines.append(f"    {constraint}")

        # Variable bounds and types
        non_negative_vars = []
        bounded_vars = []
        integer_vars = []
        binary_vars = []

        for var_name in self.get_all_variables():
            if var_name in self.variables:
                var = self.variables[var_name]

                if var.var_type == VariableType.INTEGER:
                    integer_vars.append(var_name)
                elif var.var_type == VariableType.BINARY:
                    binary_vars.append(var_name)

                if var.is_bounded:
                    if var.lower_bound == 0 and var.upper_bound is None:
                        non_negative_vars.append(var_name)
                    else:
                        bound_parts = []
                        if var.lower_bound is not None:
                            bound_parts.append(str(var.lower_bound))
                        bound_parts.append("<=")
                        bound_parts.append(var_name)
                        if var.upper_bound is not None:
                            bound_parts.append("<=")
                            bound_parts.append(str(var.upper_bound))
                        bounded_vars.append(" ".join(bound_parts))
                else:
                    non_negative_vars.append(var_name)
            else:
                # Default to non-negative
                non_negative_vars.append(var_name)

        if any([non_negative_vars, bounded_vars, integer_vars, binary_vars]):
            lines.append("")
            lines.append("where:")

            if non_negative_vars:
                lines.append(f"    {', '.join(non_negative_vars)} >= 0")

            if bounded_vars:
                for bound in bounded_vars:
                    lines.append(f"    {bound}")

            if integer_vars:
                lines.append(f"    integer {', '.join(integer_vars)}")

            if binary_vars:
                lines.append(f"    binary {', '.join(binary_vars)}")

        return "\n".join(lines)


class Solution(BaseModel):
    """Represents a solution to an optimization problem."""

    status: SolverStatus = Field(..., description="Solver status")
    objective_value: Decimal | None = Field(
        default=None, description="Optimal objective value"
    )
    variables: dict[str, Decimal] = Field(
        default_factory=dict, description="Variable values in the solution"
    )
    solve_time: float | None = Field(
        default=None, description="Time taken to solve (in seconds)"
    )
    iterations: int | None = Field(default=None, description="Number of iterations")

    @property
    def is_optimal(self) -> bool:
        """Check if solution is optimal."""
        return self.status == SolverStatus.OPTIMAL

    @property
    def is_feasible(self) -> bool:
        """Check if solution is feasible."""
        return self.status in [SolverStatus.OPTIMAL]

    def get_variable_value(self, variable_name: str) -> Decimal | None:
        """Get value of a specific variable."""
        return self.variables.get(variable_name)

    def __str__(self) -> str:
        """String representation of the solution."""
        lines = []

        if self.status == SolverStatus.OPTIMAL:
            lines.append("🎯 Optimal Solution Found!")
            if self.objective_value is not None:
                lines.append(f"   Objective Value: {self.objective_value}")
            lines.append("")
            lines.append("   Variables:")
            for var_name, value in sorted(self.variables.items()):
                lines.append(f"   🐾 {var_name} = {value}")
            lines.append("")
            lines.append("   Status: OPTIMAL 😺")
        elif self.status == SolverStatus.INFEASIBLE:
            lines.append("🙀 Problem is Infeasible!")
            lines.append("   No solution exists that satisfies all constraints.")
        elif self.status == SolverStatus.UNBOUNDED:
            lines.append("😿 Problem is Unbounded!")
            lines.append("   The objective function can be improved infinitely.")
        else:
            lines.append(f"😾 Solver Status: {self.status}")

        if self.solve_time is not None:
            lines.append(f"   ⏱️ Solve Time: {self.solve_time:.3f}s")

        return "\n".join(lines)
