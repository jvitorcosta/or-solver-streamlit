from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from solvedor.solver import parser
from solvedor.solver.models import ConstraintOperator, ObjectiveDirection, VariableType
from solvedor.solver.parser import ParseError


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
def test_parse_linear_programs(input_data, expected) -> None:
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
def test_variable_type_declarations(input_data, expected) -> None:
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
def test_problem_formatting_tolerance(input_data, expected) -> None:
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
def test_parsing_error_handling(input_data, expected_exception) -> None:
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
def test_complex_problem_structures(input_data, expected) -> None:
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
def test_parse_coefficients(input_data, expected_vars) -> None:
    """Test parsing with implicit and negative coefficients."""
    problem = parser.parse_lp_problem(input_data)
    obj_vars = problem.objective.extract_variable_names()
    assert len(obj_vars) == expected_vars


def test_comments_ignored() -> None:
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


def test_invalid_syntax_raises_error() -> None:
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


# --- Refactor Task 2: parser correctness and error messages -------------------------------

_EXAMPLES = Path(__file__).parents[1] / "src" / "solvedor" / "resources" / "examples.yaml"


@pytest.mark.parametrize(
    ("text", "rhs"),
    [
        ("max x\nsubject to:\nx <= -5\n", Decimal("-5")),
        ("max x\nsubject to:\nx >= -2.5\n", Decimal("-2.5")),
        ("max x\nsubject to:\nx <= 1e-3\n", Decimal("0.001")),
    ],
)
def test_signed_and_scientific_rhs(text: str, rhs: Decimal) -> None:
    problem = parser.parse_lp_problem(text)
    assert problem.constraints[0].rhs == rhs


def test_unicode_inequalities() -> None:
    problem = parser.parse_lp_problem("max x + y\nsubject to:\nx + y ≤ 4\nx ≥ 1\n")
    assert [c.operator for c in problem.constraints] == [
        ConstraintOperator.LESS_EQUAL,
        ConstraintOperator.GREATER_EQUAL,
    ]


@pytest.mark.parametrize(
    "header", ["subject to", "subject to:", "s.t.", "s.t.:", "st:", "sujeito a"]
)
def test_constraint_header_variants(header: str) -> None:
    problem = parser.parse_lp_problem(f"max x\n{header}\nx <= 4\n")
    assert len(problem.constraints) == 1


def test_general_bounds() -> None:
    problem = parser.parse_lp_problem(
        "max x + y + z\nsubject to:\nx + y + z <= 20\nwhere:\n0 <= x <= 10\ny >= 2\nz <= 3\n"
    )
    v = problem.variables
    assert (v["x"].lower_bound, v["x"].upper_bound) == (Decimal("0"), Decimal("10"))
    assert (v["y"].lower_bound, v["y"].upper_bound) == (Decimal("2"), None)
    assert (v["z"].lower_bound, v["z"].upper_bound) == (Decimal("0"), Decimal("3"))


def test_free_keyword() -> None:
    problem = parser.parse_lp_problem("min x\nsubject to:\nx >= -3\nwhere:\nfree x\n")
    assert problem.variables["x"].lower_bound is None


def test_objective_only_variable_is_registered() -> None:
    problem = parser.parse_lp_problem("max x + y\nsubject to:\nx <= 5\n")
    assert set(problem.variables) == {"x", "y"}


def test_unknown_variable_in_where() -> None:
    with pytest.raises(ParseError, match="yy") as err:
        parser.parse_lp_problem("min x + y\nsubject to:\nx + y >= 2\nwhere:\nx, yy >= 0\n")
    assert err.value.line == 5


def test_error_points_at_bad_constraint() -> None:
    with pytest.raises(ParseError) as err:
        parser.parse_lp_problem("max x\nsubject to:\nx <= 4\nx === 3\n")
    assert err.value.line == 4


def test_two_constraints_on_one_line_rejected() -> None:
    with pytest.raises(ParseError) as err:
        parser.parse_lp_problem("max x\nsubject to:\nx <= 4 x >= 1\n")
    assert err.value.line == 3


def test_where_lists_still_parse() -> None:
    problem = parser.parse_lp_problem(
        "max x + y + a + b\nsubject to:\nx + y + a + b <= 10\nwhere:\nx, y >= 0\ninteger a, b\n"
    )
    assert problem.variables["a"].variable_type.value == "integer"
    assert problem.variables["b"].variable_type.value == "integer"


def test_all_gallery_examples_parse() -> None:
    examples = yaml.safe_load(_EXAMPLES.read_text(encoding="utf-8"))
    for key, example in examples.items():
        problem = parser.parse_lp_problem(example["problem"])
        assert problem.constraints, key


def test_coefficient_values() -> None:
    problem = parser.parse_lp_problem("max 3x - y + 2.5z\nsubject to:\nx + y + z <= 1\n")
    coeffs = problem.objective.expression.variable_coefficients_map
    assert coeffs == {"x": Decimal("3"), "y": Decimal("-1"), "z": Decimal("2.5")}
