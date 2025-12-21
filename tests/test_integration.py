from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from solver import parser
from solver.backends import SolverFactory
from solver.models import Problem, Solution, SolverStatus, VariableType

# Import only the core functions without UI dependencies
from solver.parser import ParseError


def parse_and_solve_without_ui_dependencies(
    problem_text: str,
) -> tuple[Problem, Solution]:
    """Parse LP text and solve without UI dependencies for testing."""
    problem = parser.parse_lp_problem(problem_text)
    solver = SolverFactory.create_solver(problem)
    solution = solver.execute_optimization_with_backend(problem)
    return problem, solution


@pytest.fixture
def examples_data() -> dict:
    examples_path = Path("resources") / "examples" / "en.yaml"
    with open(examples_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


# Comprehensive problem type tests with input,expected structure
@pytest.mark.parametrize(
    "input_data,expected",
    [
        # Basic maximization problem
        (
            {
                "problem_text": """
                maximize 3*x + 2*y
                subject to:
                    x + y <= 4
                    2*x + y <= 6
                    x <= 3
                where:
                    x, y >= 0
                """
            },
            {
                "objective_direction": "MAXIMIZE",
                "variables": ["x", "y"],
                "min_constraints": 3,
                "status": SolverStatus.OPTIMAL,
                "should_have_solution": True,
            },
        ),
        # Basic minimization problem
        (
            {
                "problem_text": """
                minimize 2*x + 3*y
                subject to:
                    x + 2*y >= 5
                    3*x + y >= 6
                where:
                    x, y >= 0
                """
            },
            {
                "objective_direction": "MINIMIZE",
                "variables": ["x", "y"],
                "min_constraints": 2,
                "status": SolverStatus.OPTIMAL,
                "should_have_solution": True,
            },
        ),
        # Infeasible problem
        (
            {
                "problem_text": """
                maximize x + y
                subject to:
                    x + y >= 10
                    x + y <= 5
                where:
                    x, y >= 0
                """
            },
            {
                "objective_direction": "MAXIMIZE",
                "variables": ["x", "y"],
                "status": SolverStatus.INFEASIBLE,
                "should_have_solution": False,
                "is_feasible": False,
            },
        ),
        # Unbounded problem
        (
            {
                "problem_text": """
                maximize x + y
                subject to:
                    x >= 1
                    y >= 0.1
                where:
                    x, y >= 0
                """
            },
            {
                "objective_direction": "MAXIMIZE",
                "variables": ["x", "y"],
                "allowed_statuses": [
                    SolverStatus.OPTIMAL,
                    SolverStatus.UNBOUNDED,
                    SolverStatus.INFEASIBLE,
                ],
                "should_parse": True,
            },
        ),
        # Mixed integer problem
        (
            {
                "problem_text": """
                maximize 3*x + 2*y
                subject to:
                    x + 2*y <= 10
                    2*x + y <= 12
                where:
                    x >= 0
                    integer y
                    y >= 0
                """
            },
            {
                "objective_direction": "MAXIMIZE",
                "variables": ["x", "y"],
                "variable_types": {
                    "x": VariableType.CONTINUOUS,
                    "y": VariableType.INTEGER,
                },
                "status": SolverStatus.OPTIMAL,
                "should_have_solution": True,
                "integer_variables": ["y"],
            },
        ),
        # Problem with only objective (no constraints)
        (
            {
                "problem_text": """
                maximize x + y
                where:
                    x, y >= 0
                """
            },
            {
                "objective_direction": "MAXIMIZE",
                "variables": ["x", "y"],
                "allowed_statuses": [
                    SolverStatus.UNBOUNDED,
                    SolverStatus.OPTIMAL,
                    SolverStatus.INFEASIBLE,
                ],
                "should_parse": True,
            },
        ),
    ],
)
def test_comprehensive_problem_solving_flow(input_data: dict, expected: dict) -> None:
    """Test complete parse and solve flow for various problem types and edge cases."""
    problem, solution = parse_and_solve_without_ui_dependencies(
        input_data["problem_text"]
    )

    # Verify parsing
    assert problem is not None
    assert problem.objective.direction.name == expected["objective_direction"]

    # Verify variables
    for var_name in expected["variables"]:
        assert var_name in problem.variables

    # Verify variable types if specified
    if "variable_types" in expected:
        for var_name, expected_type in expected["variable_types"].items():
            assert problem.variables[var_name].variable_type == expected_type

    # Verify constraints count if specified
    if "min_constraints" in expected:
        assert len(problem.constraints) >= expected["min_constraints"]

    # Verify solution status
    if "status" in expected:
        assert solution.status == expected["status"]
    elif "allowed_statuses" in expected:
        assert solution.status in expected["allowed_statuses"]

    # Verify solution properties
    if expected.get("should_have_solution", False):
        assert solution.objective_value is not None
        assert solution.variable_values is not None
        assert solution.solve_time is not None

        # Verify variable values are present for all variables
        for var_name in expected["variables"]:
            assert var_name in solution.variable_values

    # Verify feasibility
    if "is_feasible" in expected:
        assert solution.is_feasible == expected["is_feasible"]

    # Check integer constraints if specified
    if "integer_variables" in expected:
        for var_name in expected["integer_variables"]:
            if var_name in solution.variable_values:
                var_value = solution.variable_values[var_name]
                assert abs(var_value - round(var_value)) < 1e-6


# YAML-based example tests with parametrization
@pytest.mark.parametrize(
    "example_key,expected",
    [
        (
            "production_planning",
            {
                "objective_direction": "MAXIMIZE",
                "min_variables": 2,
                "min_constraints": 4,
                "variable_type": VariableType.CONTINUOUS,
                "expected_profit": True,
            },
        ),
        (
            "transportation",
            {
                "objective_direction": "MINIMIZE",
                "min_variables": 6,
                "variable_type": VariableType.CONTINUOUS,
                "expected_cost": True,
            },
        ),
        (
            "diet_optimization",
            {
                "objective_direction": "MINIMIZE",
                "min_variables": 4,
                "variable_type": VariableType.INTEGER,
                "expected_cost": True,
            },
        ),
        (
            "resource_allocation",
            {
                "objective_direction": "MAXIMIZE",
                "min_variables": 3,
                "variable_type": VariableType.CONTINUOUS,
                "expected_profit": True,
            },
        ),
        (
            "facility_location",
            {
                "objective_direction": "MINIMIZE",
                "min_variables": 4,
                "expected_cost": True,
            },
        ),
    ],
)
def test_yaml_examples_comprehensive(
    examples_data: dict, example_key: str, expected: dict
) -> None:
    """Test all YAML example problems with comprehensive validation."""
    problem_text = examples_data[example_key]["problem"]

    problem, solution = parse_and_solve_without_ui_dependencies(problem_text)

    # Basic problem validation
    assert problem is not None
    assert problem.objective.direction.name == expected["objective_direction"]
    assert len(problem.variables) >= expected["min_variables"]

    if "min_constraints" in expected:
        assert len(problem.constraints) >= expected["min_constraints"]

    # Variable type validation if specified
    if "variable_type" in expected:
        for _var in problem.variables.values():
            # Allow mixed types, but check that continuous variables exist
            if expected["variable_type"] == VariableType.CONTINUOUS:
                assert any(
                    v.variable_type == VariableType.CONTINUOUS
                    for v in problem.variables.values()
                )

    # Solution validation
    assert solution is not None
    assert solution.status == SolverStatus.OPTIMAL
    assert solution.objective_value is not None
    assert solution.variable_values is not None
    assert len(solution.variable_values) == len(problem.variables)

    # Objective value validation
    if expected.get("expected_profit", False):
        assert (
            solution.objective_value > 0
        )  # Should have positive value for maximization
    elif expected.get("expected_cost", False):
        assert (
            solution.objective_value > 0
        )  # Should have positive cost for minimization

    # Variable value validation
    for _var_name, var_value in solution.variable_values.items():
        assert isinstance(var_value, (int, float, Decimal))
        assert var_value >= -1e-6  # Should be non-negative (with tolerance)


# Error handling tests with input,expected structure
@pytest.mark.parametrize(
    "input_data,expected_exception",
    [
        # Syntax errors
        (
            {
                "problem_text": """
                maximize x + y
                subject to:
                    x + invalid_operator 10
                where:
                    x, y >= 0
                """
            },
            ParseError,
        ),
        # Empty problem
        ({"problem_text": ""}, ParseError),
        # Missing objective
        (
            {
                "problem_text": """
                subject to:
                    x + y <= 10
                where:
                    x, y >= 0
                """
            },
            ParseError,
        ),
    ],
)
def test_error_handling_comprehensive(
    input_data: dict, expected_exception: type
) -> None:
    """Test comprehensive error handling for invalid problem formats."""
    with pytest.raises(expected_exception):
        parse_and_solve_without_ui_dependencies(input_data["problem_text"])


# Solution formatting and properties tests
@pytest.mark.parametrize(
    "input_data,expected",
    [
        # Test solution string representation
        (
            {"example_key": "production_planning"},
            {
                "str_contains": [
                    "Optimal Solution Found!",
                    "Objective Value:",
                    "Variables:",
                ],
                "has_variable_names": True,
            },
        )
    ],
)
def test_solution_formatting_comprehensive(
    examples_data: dict, input_data: dict, expected: dict
) -> None:
    """Test solution formatting and string representation."""
    problem_text = examples_data[input_data["example_key"]]["problem"]
    problem, solution = parse_and_solve_without_ui_dependencies(problem_text)

    solution_str = str(solution)

    # Check for required string components
    for required_str in expected["str_contains"]:
        assert any(
            req in solution_str
            for req in [
                required_str,
                required_str.lower(),
                required_str.replace(" ", ""),
            ]
        )

    # Check for variable names if expected
    if expected.get("has_variable_names", False):
        for var_name in problem.variables:
            assert var_name in solution_str


# Problem variable extraction tests
def test_problem_variable_extraction_comprehensive(examples_data: dict) -> None:
    """Test comprehensive variable extraction from problems."""
    problem_text = examples_data["production_planning"]["problem"]
    problem, solution = parse_and_solve_without_ui_dependencies(problem_text)

    # Test variable extraction from problem
    assert len(problem.variables) > 0

    # Test variable extraction from objective
    objective_vars = problem.objective.extract_variable_names()
    assert len(objective_vars) > 0

    # Should be subset of problem variables
    problem_var_names = set(problem.variables.keys())
    objective_var_names = set(objective_vars)
    assert objective_var_names.issubset(problem_var_names)

    # Test variable extraction from constraints
    for constraint in problem.constraints:
        constraint_vars = constraint.extract_variable_names()
        constraint_var_names = set(constraint_vars)
        assert constraint_var_names.issubset(problem_var_names)


# Comprehensive test for all examples (parametrized)
@pytest.mark.parametrize(
    "example_key",
    [
        "production_planning",
        "transportation",
        "diet_optimization",
        "resource_allocation",
        "facility_location",
    ],
)
def test_all_examples_solve_successfully_comprehensive(
    examples_data: dict, example_key: str
) -> None:
    """Test that all example problems solve successfully with comprehensive validation."""
    problem_text = examples_data[example_key]["problem"]

    # Should not raise any exceptions
    problem, solution = parse_and_solve_without_ui_dependencies(problem_text)

    # Basic validations for all examples
    assert problem is not None
    assert solution is not None
    assert len(problem.variables) > 0
    assert len(problem.constraints) > 0

    # Should have optimal solution for our examples
    assert solution.status == SolverStatus.OPTIMAL
    assert solution.objective_value is not None
    assert solution.variable_values is not None
    assert len(solution.variable_values) == len(problem.variables)

    # All variable values should be valid numbers
    for _var_name, var_value in solution.variable_values.items():
        assert isinstance(var_value, (int, float, Decimal))
        assert not (
            isinstance(var_value, float) and (var_value != var_value)
        )  # Not NaN
        assert var_value >= -1e-6  # Non-negative with tolerance
