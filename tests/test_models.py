"""Test the domain models."""

from decimal import Decimal

import pytest

from solver.models import (
    Constraint,
    ConstraintOperator,
    LinearExpression,
    ObjectiveDirection,
    ObjectiveFunction,
    Problem,
    Solution,
    SolverStatus,
    Variable,
    VariableType,
    add_term_to_linear_expression,
)


class TestVariable:
    """Test Variable model."""

    def test_instantiate_continuous_variable_with_defaults(self):
        """Test creating a basic continuous variable with default settings."""
        basic_continuous_variable = Variable(name="x1")
        assert basic_continuous_variable.name == "x1"
        assert basic_continuous_variable.variable_type == VariableType.CONTINUOUS
        assert basic_continuous_variable.lower_bound is None
        assert basic_continuous_variable.upper_bound is None

    def test_instantiate_bounded_integer_variable(self):
        """Test creating integer variable with explicit bounds."""
        bounded_integer_variable = Variable(
            name="y1",
            variable_type=VariableType.INTEGER,
            lower_bound=Decimal("0"),
            upper_bound=Decimal("10"),
        )
        assert bounded_integer_variable.name == "y1"
        assert bounded_integer_variable.variable_type == VariableType.INTEGER
        assert bounded_integer_variable.lower_bound == Decimal("0")
        assert bounded_integer_variable.upper_bound == Decimal("10")

    def test_instantiate_zero_one_binary_variable(self):
        """Test creating binary variable constrained to 0-1 values."""
        zero_one_binary_variable = Variable(
            name="z1",
            variable_type=VariableType.BINARY,
            lower_bound=Decimal("0"),
            upper_bound=Decimal("1"),
        )
        assert zero_one_binary_variable.name == "z1"
        assert zero_one_binary_variable.variable_type == VariableType.BINARY
        assert zero_one_binary_variable.is_bounded

    def test_reject_invalid_variable_name_format(self):
        """Test validation rejects invalid variable name formats."""
        with pytest.raises(ValueError):
            Variable(name="123invalid")

    def test_identify_non_negative_variables_correctly(self):
        """Test identification of non-negative vs negative-allowed variables."""
        non_negative_variable = Variable(name="x1", lower_bound=Decimal("0"))
        assert non_negative_variable.is_non_negative

        negative_allowed_variable = Variable(name="x2", lower_bound=Decimal("-5"))
        assert not negative_allowed_variable.is_non_negative


class TestLinearExpression:
    """Test LinearExpression model."""

    def test_instantiate_empty_linear_expression(self):
        """Test creating empty linear expression with default values."""
        empty_linear_expression = LinearExpression()
        assert len(empty_linear_expression.terms) == 0
        assert empty_linear_expression.constant == Decimal("0")
        assert str(empty_linear_expression) == "0"

    def test_add_coefficient_variable_term_to_expression(self):
        """Test adding single coefficient-variable term to linear expression."""
        linear_expression_with_one_term = LinearExpression()
        add_term_to_linear_expression(
            linear_expression_with_one_term, Decimal("2"), "x1"
        )

        assert len(linear_expression_with_one_term.terms) == 1
        assert linear_expression_with_one_term.terms[0].coefficient == Decimal("2")
        assert linear_expression_with_one_term.terms[0].variable == "x1"
        assert str(linear_expression_with_one_term) == "2*x1"

    def test_build_multi_variable_linear_expression(self):
        """Test building linear expression with multiple variable terms."""
        multi_variable_expression = LinearExpression()
        multi_variable_expression.add_term(Decimal("2"), "x1")
        multi_variable_expression.add_term(Decimal("3"), "x2")
        multi_variable_expression.add_term(Decimal("-1"), "x3")

        assert len(multi_variable_expression.terms) == 3
        extracted_variable_names = multi_variable_expression.extract_variable_names()
        assert "x1" in extracted_variable_names
        assert "x2" in extracted_variable_names
        assert "x3" in extracted_variable_names

    def test_combine_same_variable(self):
        """Test combining coefficients for the same variable."""
        expr = LinearExpression()
        expr.add_term(Decimal("2"), "x1")
        expr.add_term(Decimal("3"), "x1")  # Should combine with previous

        assert len(expr.terms) == 1
        assert expr.terms[0].coefficient == Decimal("5")
        assert expr.terms[0].variable == "x1"

    def test_expression_string_representation(self):
        """Test string representation of expressions."""
        expr = LinearExpression()
        expr.add_term(Decimal("1"), "x1")
        assert str(expr) == "x1"  # Coefficient 1 is implicit

        expr.add_term(Decimal("-1"), "x2")
        assert "- x2" in str(expr)  # Coefficient -1 shows as negative


class TestConstraint:
    """Test Constraint model."""

    def test_create_constraint(self):
        """Test creating a constraint."""
        expr = LinearExpression()
        expr.add_term(Decimal("2"), "x1")
        expr.add_term(Decimal("3"), "x2")

        constraint = Constraint(
            expression=expr, operator=ConstraintOperator.LESS_EQUAL, rhs=Decimal("10")
        )

        assert constraint.operator == ConstraintOperator.LESS_EQUAL
        assert constraint.rhs == Decimal("10")
        assert len(constraint.extract_variable_names()) == 2
        assert "2*x1 + 3*x2 <= 10" in str(constraint)


class TestObjectiveFunction:
    """Test ObjectiveFunction model."""

    def test_minimize_objective(self):
        """Test minimize objective function."""
        expr = LinearExpression()
        expr.add_term(Decimal("2"), "x1")
        expr.add_term(Decimal("3"), "x2")

        obj = ObjectiveFunction(direction=ObjectiveDirection.MINIMIZE, expression=expr)

        assert obj.direction == ObjectiveDirection.MINIMIZE
        assert len(obj.extract_variable_names()) == 2
        assert "minimize" in str(obj)

    def test_maximize_objective(self):
        """Test maximize objective function."""
        expr = LinearExpression()
        expr.add_term(Decimal("5"), "profit")

        obj = ObjectiveFunction(direction=ObjectiveDirection.MAXIMIZE, expression=expr)

        assert obj.direction == ObjectiveDirection.MAXIMIZE
        assert "maximize" in str(obj)


class TestProblem:
    """Test Problem model."""

    def test_create_basic_problem(self):
        """Test creating a basic problem."""
        # Create objective
        obj_expr = LinearExpression()
        obj_expr.add_term(Decimal("2"), "x1")
        obj_expr.add_term(Decimal("3"), "x2")

        objective = ObjectiveFunction(
            direction=ObjectiveDirection.MINIMIZE, expression=obj_expr
        )

        # Create constraint
        constr_expr = LinearExpression()
        constr_expr.add_term(Decimal("1"), "x1")
        constr_expr.add_term(Decimal("1"), "x2")

        constraint = Constraint(
            expression=constr_expr,
            operator=ConstraintOperator.LESS_EQUAL,
            rhs=Decimal("10"),
        )

        # Create problem
        problem = Problem(objective=objective)
        problem.add_constraint(constraint)

        assert len(problem.constraints) == 1
        assert len(problem.variables) == 2
        assert "x1" in problem.extract_all_variable_names()
        assert "x2" in problem.extract_all_variable_names()

    def test_problem_string_representation(self):
        """Test string representation of complete problem."""
        # Simple problem
        obj_expr = LinearExpression()
        obj_expr.add_term(Decimal("1"), "cat_food")

        objective = ObjectiveFunction(
            direction=ObjectiveDirection.MINIMIZE, expression=obj_expr
        )

        problem = Problem(objective=objective)
        problem_str = str(problem)

        assert "minimize cat_food" in problem_str
        assert "cat_food >= 0" in problem_str  # Default non-negative


class TestSolution:
    """Test Solution model."""

    def test_optimal_solution(self):
        """Test optimal solution."""
        solution = Solution(
            status=SolverStatus.OPTIMAL,
            objective_value=Decimal("15.5"),
            variable_values={"x1": 2.5, "x2": 3.0},
            solve_time=0.123,
        )

        assert solution.is_optimal
        assert solution.is_feasible
        assert solution.variable_values["x1"] == 2.5
        assert solution.extract_variable_value("nonexistent") is None
        assert "🎯 Optimal Solution Found!" in str(solution)

    def test_infeasible_solution(self):
        """Test infeasible solution."""
        solution = Solution(status=SolverStatus.INFEASIBLE)

        assert not solution.is_optimal
        assert not solution.is_feasible
        assert "🙀 Problem is Infeasible!" in str(solution)

    def test_unbounded_solution(self):
        """Test unbounded solution."""
        solution = Solution(status=SolverStatus.UNBOUNDED)

        assert not solution.is_optimal
        assert "😿 Problem is Unbounded!" in str(solution)
