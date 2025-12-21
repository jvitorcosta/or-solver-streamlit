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


# Variable tests with comprehensive parametrization
@pytest.mark.parametrize(
    "input_data,expected",
    [
        # Continuous variable (default)
        (
            {"name": "x1"},
            {
                "name": "x1",
                "variable_type": VariableType.CONTINUOUS,
                "lower_bound": None,
                "upper_bound": None,
                "is_non_negative": False,
            },
        ),
        # Integer variable with bounds
        (
            {
                "name": "y1",
                "variable_type": VariableType.INTEGER,
                "lower_bound": Decimal("0"),
                "upper_bound": Decimal("10"),
            },
            {
                "name": "y1",
                "variable_type": VariableType.INTEGER,
                "lower_bound": Decimal("0"),
                "upper_bound": Decimal("10"),
                "is_non_negative": True,
            },
        ),
        # Binary variable
        (
            {
                "name": "z1",
                "variable_type": VariableType.BINARY,
                "lower_bound": Decimal("0"),
                "upper_bound": Decimal("1"),
            },
            {
                "name": "z1",
                "variable_type": VariableType.BINARY,
                "lower_bound": Decimal("0"),
                "upper_bound": Decimal("1"),
                "is_non_negative": True,
            },
        ),
        # Variable with negative lower bound
        (
            {"name": "x2", "lower_bound": Decimal("-5")},
            {
                "name": "x2",
                "variable_type": VariableType.CONTINUOUS,
                "lower_bound": Decimal("-5"),
                "is_non_negative": False,
            },
        ),
    ],
)
def test_variable_creation_and_properties(input_data: dict, expected: dict) -> None:
    """Test variable creation with different configurations and property validation."""
    variable = Variable(**input_data)

    assert variable.name == expected["name"]
    assert variable.variable_type == expected["variable_type"]

    if "lower_bound" in expected:
        assert variable.lower_bound == expected["lower_bound"]
    if "upper_bound" in expected:
        assert variable.upper_bound == expected["upper_bound"]
    if "is_non_negative" in expected:
        assert variable.is_non_negative == expected["is_non_negative"]


@pytest.mark.parametrize(
    "input_data,expected_exception",
    [
        ({"name": "123invalid"}, ValueError),
        ({"name": ""}, ValueError),
        ({"name": "valid_name"}, None),  # Should not raise
    ],
)
def test_variable_name_validation(input_data: dict, expected_exception: type) -> None:
    """Test variable name validation."""
    if expected_exception is None:
        # Should not raise exception
        variable = Variable(**input_data)
        assert variable.name == input_data["name"]
    else:
        # Should raise specified exception
        with pytest.raises(expected_exception):
            Variable(**input_data)


# LinearExpression tests with comprehensive parametrization
@pytest.mark.parametrize(
    "input_data,expected",
    [
        # Empty expression
        (
            {},
            {
                "terms_count": 0,
                "constant": Decimal("0"),
                "str_repr": "0",
                "variables": [],
            },
        ),
        # Single term with coefficient 1
        (
            {"terms": [(Decimal("1"), "x1")]},
            {"terms_count": 1, "str_repr": "x1", "variables": ["x1"]},
        ),
        # Single term with coefficient != 1
        (
            {"terms": [(Decimal("2"), "x1")]},
            {"terms_count": 1, "str_repr": "2*x1", "variables": ["x1"]},
        ),
        # Single term with negative coefficient
        (
            {"terms": [(Decimal("-1"), "x2")]},
            {"terms_count": 1, "str_repr": "-x2", "variables": ["x2"]},
        ),
        # Multiple terms
        (
            {"terms": [(Decimal("2"), "x1"), (Decimal("-1"), "x2")]},
            {"terms_count": 2, "variables": ["x1", "x2"]},
        ),
        # Terms that should combine (same variable)
        (
            {"terms": [(Decimal("2"), "x1"), (Decimal("3"), "x1")]},
            {
                "terms_count": 1,
                "combined_coefficient": Decimal("5"),
                "variables": ["x1"],
            },
        ),
    ],
)
def test_linear_expression_operations(input_data: dict, expected: dict) -> None:
    """Test linear expression creation, term addition, and properties."""
    expression = LinearExpression()

    # Add terms if provided
    for coeff, var_name in input_data.get("terms", []):
        add_term_to_linear_expression(expression, coeff, var_name)

    assert len(expression.terms) == expected["terms_count"]

    if "constant" in expected:
        assert expression.constant == expected["constant"]

    if "str_repr" in expected:
        assert str(expression) == expected["str_repr"]

    variables = expression.extract_variable_names()
    assert set(variables) == set(expected["variables"])

    # Check combined coefficient for same-variable cases
    if "combined_coefficient" in expected and expression.terms:
        assert expression.terms[0].coefficient == expected["combined_coefficient"]


# Constraint tests with parametrization
@pytest.mark.parametrize(
    "input_data,expected",
    [
        # Less than or equal constraint
        (
            {
                "terms": [(Decimal("2"), "x1"), (Decimal("3"), "x2")],
                "operator": ConstraintOperator.LESS_EQUAL,
                "rhs": Decimal("10"),
            },
            {
                "operator": ConstraintOperator.LESS_EQUAL,
                "rhs": Decimal("10"),
                "variables": ["x1", "x2"],
                "str_contains": "<=",
            },
        ),
        # Greater than or equal constraint
        (
            {
                "terms": [(Decimal("1"), "a"), (Decimal("1"), "b")],
                "operator": ConstraintOperator.GREATER_EQUAL,
                "rhs": Decimal("5"),
            },
            {
                "operator": ConstraintOperator.GREATER_EQUAL,
                "rhs": Decimal("5"),
                "variables": ["a", "b"],
                "str_contains": ">=",
            },
        ),
    ],
)
def test_constraint_creation_and_properties(input_data: dict, expected: dict) -> None:
    """Test constraint creation with different operators and expressions."""
    # Create expression
    expr = LinearExpression()
    for coeff, var_name in input_data["terms"]:
        expr.add_term(coeff, var_name)

    constraint = Constraint(
        expression=expr, operator=input_data["operator"], rhs=input_data["rhs"]
    )

    assert constraint.operator == expected["operator"]
    assert constraint.rhs == expected["rhs"]

    variables = constraint.extract_variable_names()
    assert set(variables) == set(expected["variables"])

    if "str_contains" in expected:
        assert expected["str_contains"] in str(constraint)


# ObjectiveFunction tests with parametrization
@pytest.mark.parametrize(
    "input_data,expected",
    [
        # Maximization objective
        (
            {
                "direction": ObjectiveDirection.MAXIMIZE,
                "terms": [(Decimal("3"), "x"), (Decimal("2"), "y")],
            },
            {
                "direction": ObjectiveDirection.MAXIMIZE,
                "variables": ["x", "y"],
                "str_contains": "maximize",
            },
        ),
        # Minimization objective
        (
            {
                "direction": ObjectiveDirection.MINIMIZE,
                "terms": [(Decimal("1"), "cost")],
            },
            {
                "direction": ObjectiveDirection.MINIMIZE,
                "variables": ["cost"],
                "str_contains": "minimize",
            },
        ),
    ],
)
def test_objective_function_creation_and_properties(
    input_data: dict, expected: dict
) -> None:
    """Test objective function creation with different directions."""
    # Create expression
    expr = LinearExpression()
    for coeff, var_name in input_data["terms"]:
        expr.add_term(coeff, var_name)

    objective = ObjectiveFunction(direction=input_data["direction"], expression=expr)

    assert objective.direction == expected["direction"]

    variables = objective.extract_variable_names()
    assert set(variables) == set(expected["variables"])

    if "str_contains" in expected:
        assert expected["str_contains"] in str(objective)


# Problem tests with parametrization
@pytest.mark.parametrize(
    "input_data,expected",
    [
        # Basic minimization problem
        (
            {
                "objective": {
                    "direction": ObjectiveDirection.MINIMIZE,
                    "terms": [(Decimal("2"), "x1"), (Decimal("3"), "x2")],
                },
                "constraints": [
                    {
                        "terms": [(Decimal("1"), "x1"), (Decimal("1"), "x2")],
                        "operator": ConstraintOperator.LESS_EQUAL,
                        "rhs": Decimal("10"),
                    }
                ],
            },
            {
                "objective_direction": ObjectiveDirection.MINIMIZE,
                "variables_count": 2,
                "constraints_count": 1,
                "variable_names": ["x1", "x2"],
                "str_contains": "minimize",
            },
        )
    ],
)
def test_problem_creation_and_properties(input_data: dict, expected: dict) -> None:
    """Test complete problem creation and validation."""
    # Create objective
    obj_expr = LinearExpression()
    for coeff, var_name in input_data["objective"]["terms"]:
        obj_expr.add_term(coeff, var_name)

    objective = ObjectiveFunction(
        direction=input_data["objective"]["direction"], expression=obj_expr
    )

    # Create problem
    problem = Problem(objective=objective)

    # Add constraints
    for constraint_data in input_data["constraints"]:
        constr_expr = LinearExpression()
        for coeff, var_name in constraint_data["terms"]:
            constr_expr.add_term(coeff, var_name)

        constraint = Constraint(
            expression=constr_expr,
            operator=constraint_data["operator"],
            rhs=constraint_data["rhs"],
        )
        problem.add_constraint(constraint)

    assert problem.objective.direction == expected["objective_direction"]
    assert len(problem.constraints) == expected["constraints_count"]
    assert len(problem.variables) == expected["variables_count"]

    all_vars = problem.extract_all_variable_names()
    assert set(all_vars) == set(expected["variable_names"])

    if "str_contains" in expected:
        assert expected["str_contains"] in str(problem)


# Solution tests with comprehensive parametrization
@pytest.mark.parametrize(
    "input_data,expected",
    [
        # Optimal solution
        (
            {
                "status": SolverStatus.OPTIMAL,
                "objective_value": Decimal("15.5"),
                "variable_values": {"x1": 2.5, "x2": 3.0},
                "solve_time": 0.123,
            },
            {
                "status": SolverStatus.OPTIMAL,
                "is_optimal": True,
                "is_feasible": True,
                "str_contains": "🎯 Optimal Solution Found!",
                "has_variable_values": True,
            },
        ),
        # Infeasible solution
        (
            {"status": SolverStatus.INFEASIBLE},
            {
                "status": SolverStatus.INFEASIBLE,
                "is_optimal": False,
                "is_feasible": False,
                "str_contains": "🙀 Problem is Infeasible!",
                "has_variable_values": False,
            },
        ),
        # Unbounded solution
        (
            {"status": SolverStatus.UNBOUNDED},
            {
                "status": SolverStatus.UNBOUNDED,
                "is_optimal": False,
                "is_feasible": False,
                "str_contains": "😿 Problem is Unbounded!",
                "has_variable_values": False,
            },
        ),
    ],
)
def test_solution_status_and_properties(input_data: dict, expected: dict) -> None:
    """Test solution creation with different statuses and property validation."""
    solution = Solution(**input_data)

    assert solution.status == expected["status"]
    assert solution.is_optimal == expected["is_optimal"]
    assert solution.is_feasible == expected["is_feasible"]

    solution_str = str(solution)
    assert expected["str_contains"] in solution_str

    if expected["has_variable_values"] and "variable_values" in input_data:
        for var_name, var_value in input_data["variable_values"].items():
            assert solution.variable_values[var_name] == var_value

    # Test variable extraction
    assert solution.extract_variable_value("nonexistent") is None
