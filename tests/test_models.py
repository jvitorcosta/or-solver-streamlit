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
)


class TestVariable:
    """Test Variable model."""

    def test_create_basic_variable(self):
        """Test creating a basic variable."""
        var = Variable(name="x1")
        assert var.name == "x1"
        assert var.var_type == VariableType.CONTINUOUS
        assert var.lower_bound is None
        assert var.upper_bound is None

    def test_create_integer_variable(self):
        """Test creating an integer variable."""
        var = Variable(
            name="y1",
            var_type=VariableType.INTEGER,
            lower_bound=Decimal("0"),
            upper_bound=Decimal("10"),
        )
        assert var.name == "y1"
        assert var.var_type == VariableType.INTEGER
        assert var.lower_bound == Decimal("0")
        assert var.upper_bound == Decimal("10")

    def test_create_binary_variable(self):
        """Test creating a binary variable."""
        var = Variable(
            name="z1",
            var_type=VariableType.BINARY,
            lower_bound=Decimal("0"),
            upper_bound=Decimal("1"),
        )
        assert var.name == "z1"
        assert var.var_type == VariableType.BINARY
        assert var.is_bounded

    def test_invalid_variable_name(self):
        """Test variable name validation."""
        with pytest.raises(ValueError):
            Variable(name="123invalid")

    def test_is_non_negative(self):
        """Test non-negative property."""
        var = Variable(name="x1", lower_bound=Decimal("0"))
        assert var.is_non_negative

        var_neg = Variable(name="x2", lower_bound=Decimal("-5"))
        assert not var_neg.is_non_negative


class TestLinearExpression:
    """Test LinearExpression model."""

    def test_create_empty_expression(self):
        """Test creating an empty expression."""
        expr = LinearExpression()
        assert len(expr.terms) == 0
        assert expr.constant == Decimal("0")
        assert str(expr) == "0"

    def test_add_single_term(self):
        """Test adding a single term."""
        expr = LinearExpression()
        expr.add_term(Decimal("2"), "x1")

        assert len(expr.terms) == 1
        assert expr.terms[0].coefficient == Decimal("2")
        assert expr.terms[0].variable == "x1"
        assert str(expr) == "2*x1"

    def test_add_multiple_terms(self):
        """Test adding multiple terms."""
        expr = LinearExpression()
        expr.add_term(Decimal("2"), "x1")
        expr.add_term(Decimal("3"), "x2")
        expr.add_term(Decimal("-1"), "x3")

        assert len(expr.terms) == 3
        variables = expr.get_variables()
        assert "x1" in variables
        assert "x2" in variables
        assert "x3" in variables

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
        assert len(constraint.get_variables()) == 2
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
        assert len(obj.get_variables()) == 2
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
        assert len(problem.get_all_variables()) == 2
        assert "x1" in problem.get_all_variables()
        assert "x2" in problem.get_all_variables()

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
        assert solution.get_variable_value("x1") == 2.5
        assert solution.get_variable_value("nonexistent") is None
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
