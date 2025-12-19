import pytest

from solver import parser
from solver.models import ObjectiveDirection, VariableType
from solver.parser import ParseError


# Core parsing tests with input,expected structure
@pytest.mark.parametrize(
    "input_data,expected",
    [
        # Basic maximization problem
        (
            {
                "problem_text": """
                maximize 2*x1 + 3*x2
                subject to:
                    x1 + x2 <= 10
                    x1 >= 0
                    x2 >= 0
                where:
                    x1, x2 >= 0
                """
            },
            {
                "objective_direction": ObjectiveDirection.MAXIMIZE,
                "variables_count": 2,
                "min_constraints": 2,
                "variable_names": ["x1", "x2"],
            },
        ),
        # Basic minimization problem
        (
            {
                "problem_text": """
                minimize x1 + 2*x2
                subject to:
                    2*x1 + x2 >= 3
                where:
                    x1, x2 >= 0
                """
            },
            {
                "objective_direction": ObjectiveDirection.MINIMIZE,
                "variables_count": 2,
                "min_constraints": 1,
                "variable_names": ["x1", "x2"],
            },
        ),
        # Portuguese keywords
        (
            {
                "problem_text": """
                maximizar 8*comida_gato + 10*brinquedos_gato
                sujeito a:
                    0.5*comida_gato + 0.5*brinquedos_gato <= 150
                onde:
                    comida_gato, brinquedos_gato >= 0
                """
            },
            {
                "objective_direction": ObjectiveDirection.MAXIMIZE,
                "variables_count": 2,
                "min_constraints": 1,
                "variable_names": ["comida_gato", "brinquedos_gato"],
            },
        ),
        # Implicit coefficients
        (
            {
                "problem_text": """
                maximize x1 + x2  # Coefficient 1 is implicit
                subject to:
                    x1 + 2*x2 <= 5
                where:
                    x1, x2 >= 0
                """
            },
            {
                "objective_direction": ObjectiveDirection.MAXIMIZE,
                "variables_count": 2,
                "min_constraints": 1,
                "variable_names": ["x1", "x2"],
            },
        ),
    ],
)
def test_parse_linear_programs(input_data, expected):
    """Test parsing linear programs with different objective directions and formats."""
    problem = parser.parse_lp_problem(input_data["problem_text"])

    assert problem.objective.direction == expected["objective_direction"]

    objective_vars = problem.objective.extract_variable_names()
    assert len(objective_vars) == expected["variables_count"]
    assert set(objective_vars) == set(expected["variable_names"])

    assert len(problem.constraints) >= expected["min_constraints"]


# Variable type declaration tests
@pytest.mark.parametrize(
    "input_data,expected",
    [
        # Basic continuous variables (default)
        (
            {
                "problem_text": """
                maximize x + y
                subject to:
                    x + y <= 10
                where:
                    x, y >= 0
                """
            },
            {
                "variables": {
                    "x": {"type": VariableType.CONTINUOUS},
                    "y": {"type": VariableType.CONTINUOUS},
                }
            },
        ),
        # Integer variables
        (
            {
                "problem_text": """
                maximize x + y
                subject to:
                    x + y <= 10
                where:
                    integer x, y
                    x, y >= 0
                """
            },
            {
                "variables": {
                    "x": {"type": VariableType.INTEGER},
                    "y": {"type": VariableType.INTEGER},
                }
            },
        ),
        # Binary variables
        (
            {
                "problem_text": """
                maximize x + y
                subject to:
                    x + y <= 1
                where:
                    binary x, y
                """
            },
            {
                "variables": {
                    "x": {"type": VariableType.BINARY},
                    "y": {"type": VariableType.BINARY},
                }
            },
        ),
        # Mixed variable types
        (
            {
                "problem_text": """
                maximize x + y + z
                subject to:
                    x + y + z <= 10
                where:
                    integer y
                    binary z
                    x >= 0
                    y >= 0
                """
            },
            {
                "variables": {
                    "x": {"type": VariableType.CONTINUOUS},
                    "y": {"type": VariableType.INTEGER},
                    "z": {"type": VariableType.BINARY},
                }
            },
        ),
    ],
)
def test_variable_type_declarations(input_data, expected):
    """Test parsing of different variable type declarations."""
    problem = parser.parse_lp_problem(input_data["problem_text"])

    for var_name, var_props in expected["variables"].items():
        assert var_name in problem.variables
        assert problem.variables[var_name].variable_type == var_props["type"]


# Problem structure and formatting tests
@pytest.mark.parametrize(
    "input_data,expected",
    [
        # Comments handling
        (
            {
                "problem_text": """
                # This is a comment
                maximize 2*x + 3*y  # Inline comment
                subject to:
                    x + y <= 4  # Another comment
                    # Comment line
                where:
                    x, y >= 0
                """
            },
            {
                "objective_direction": ObjectiveDirection.MAXIMIZE,
                "variables_count": 2,
                "should_parse": True,
            },
        ),
        # Extra whitespace handling
        (
            {
                "problem_text": """


                maximize     x   +    y

                subject   to:

                    x   +   y   <=   5

                where:

                    x,   y   >=   0

                """
            },
            {
                "objective_direction": ObjectiveDirection.MAXIMIZE,
                "variables_count": 2,
                "should_parse": True,
            },
        ),
    ],
)
def test_problem_formatting_tolerance(input_data, expected):
    """Test parsing tolerance for comments and whitespace."""
    if expected["should_parse"]:
        problem = parser.parse_lp_problem(input_data["problem_text"])
        assert problem.objective.direction == expected["objective_direction"]

        objective_vars = problem.objective.extract_variable_names()
        assert len(objective_vars) == expected["variables_count"]
    else:
        with pytest.raises(ParseError):
            parser.parse_lp_problem(input_data["problem_text"])


# Error handling tests with input,expected structure
@pytest.mark.parametrize(
    "input_data,expected_exception",
    [
        # Invalid syntax
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
        # Invalid variable name
        (
            {
                "problem_text": """
                maximize 123invalid + y
                subject to:
                    123invalid + y <= 10
                where:
                    123invalid, y >= 0
                """
            },
            ParseError,
        ),
    ],
)
def test_parsing_error_handling(input_data, expected_exception):
    """Test that invalid syntax raises appropriate parsing errors."""
    with pytest.raises(expected_exception):
        parser.parse_lp_problem(input_data["problem_text"])


# Complex problem structure tests
@pytest.mark.parametrize(
    "input_data,expected",
    [
        # Multi-constraint problem
        (
            {
                "problem_text": """
                minimize 2*cost + 3*time
                subject to:
                    cost + time >= 10
                    2*cost <= 20
                    time >= 5
                    cost + 2*time <= 25
                where:
                    cost, time >= 0
                """
            },
            {
                "objective_direction": ObjectiveDirection.MINIMIZE,
                "variables": ["cost", "time"],
                "min_constraints": 4,
            },
        ),
        # Problem with many variables
        (
            {
                "problem_text": """
                maximize a + b + c + d + e
                subject to:
                    a + b + c <= 10
                    d + e >= 5
                    a + c + e <= 8
                where:
                    a, b, c, d, e >= 0
                """
            },
            {
                "objective_direction": ObjectiveDirection.MAXIMIZE,
                "variables": ["a", "b", "c", "d", "e"],
                "min_constraints": 3,
            },
        ),
    ],
)
def test_complex_problem_structures(input_data, expected):
    """Test parsing of complex problems with multiple variables and constraints."""
    problem = parser.parse_lp_problem(input_data["problem_text"])

    assert problem.objective.direction == expected["objective_direction"]

    objective_vars = problem.objective.extract_variable_names()
    assert set(objective_vars) == set(expected["variables"])

    assert len(problem.constraints) >= expected["min_constraints"]


@pytest.mark.parametrize(
    "input_data,expected_vars",
    [
        (
            """
            minimize 2*x1 - 3*x2 + x3
            subject to:
                x1 - x2 + 2*x3 <= 10
            where:
                x1, x2, x3 >= 0
            """,
            3,
        )
    ],
)
def test_parse_coefficients(input_data, expected_vars):
    """Test parsing with implicit and negative coefficients."""
    problem = parser.parse_lp_problem(input_data)
    obj_vars = problem.objective.extract_variable_names()
    assert len(obj_vars) == expected_vars


def test_comments_ignored():
    """Test that comments are properly ignored."""
    problem_text = """
        # This is a cat food optimization problem
        maximize 8*cat_food + 10*cat_toys  # Maximize happiness

        subject to:
            0.5*cat_food + 0.5*cat_toys <= 150  # Budget constraint
            # cat_food >= 30  # This constraint is commented out

        where:
            cat_food, cat_toys >= 0  # Non-negative variables
        """

    problem = parser.parse_lp_problem(problem_text)

    # Should parse successfully despite comments
    assert problem.objective.direction == ObjectiveDirection.MAXIMIZE
    variables = list(problem.variables.keys())
    assert "cat_food" in variables
    assert "cat_toys" in variables


def test_invalid_syntax_raises_error():
    """Test that invalid syntax raises ParseError."""
    invalid_problems = [
        "invalid problem text",
        "maximize",  # Missing expression
        "maximize x1\nsubject to:\n x1 <=",  # Missing RHS
        "maximize x1\nsubject to:\n x1 ?? 5",  # Invalid operator
    ]

    for invalid_text in invalid_problems:
        with pytest.raises(ParseError):
            parser.parse_lp_problem(invalid_text)
