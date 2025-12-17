import pytest

from solver.models import (
    ObjectiveDirection,
    VariableType,
)
from solver.parser import (
    ParseError,
    parse_lp_problem,
)


class TestLPParser:
    """Test the modern LP parser."""

    def test_simple_maximize_problem(self):
        """Test parsing a simple maximize problem."""
        problem_text = """
        maximize 2*x1 + 3*x2

        subject to:
            x1 + x2 <= 10
            x1 >= 0
            x2 >= 0

        where:
            x1, x2 >= 0
        """

        problem = parse_lp_problem(problem_text)

        assert problem.objective.direction == ObjectiveDirection.MAXIMIZE
        assert len(problem.objective.get_variables()) == 2
        assert len(problem.constraints) >= 2  # At least the explicit constraints

    def test_simple_minimize_problem(self):
        """Test parsing a simple minimize problem."""
        problem_text = """
        minimize x1 + 2*x2

        subject to:
            2*x1 + x2 >= 3

        where:
            x1, x2 >= 0
        """

        problem = parse_lp_problem(problem_text)

        assert problem.objective.direction == ObjectiveDirection.MINIMIZE
        assert len(problem.constraints) >= 1

    def test_portuguese_syntax(self):
        """Test parsing Portuguese syntax."""
        problem_text = """
        maximizar 8*comida_gato + 10*brinquedos_gato

        sujeito a:
            0.5*comida_gato + 0.5*brinquedos_gato <= 150

        onde:
            comida_gato, brinquedos_gato >= 0
        """

        problem = parse_lp_problem(problem_text)

        assert (
            problem.objective.direction == ObjectiveDirection.MAXIMIZE
        )  # Converted to English internally
        variables = problem.get_all_variables()
        assert "comida_gato" in variables
        assert "brinquedos_gato" in variables

    def test_implicit_coefficients(self):
        """Test parsing with implicit coefficients."""
        problem_text = """
        maximize x1 + x2  # Implicit coefficient 1

        subject to:
            x1 + 2*x2 <= 5

        where:
            x1, x2 >= 0
        """

        problem = parse_lp_problem(problem_text)

        # Check that implicit coefficients are handled
        obj_vars = problem.objective.get_variables()
        assert len(obj_vars) == 2

    def test_negative_coefficients(self):
        """Test parsing negative coefficients."""
        problem_text = """
        minimize 2*x1 - 3*x2 + x3

        subject to:
            x1 - x2 + 2*x3 <= 10

        where:
            x1, x2, x3 >= 0
        """

        problem = parse_lp_problem(problem_text)

        assert len(problem.objective.get_variables()) == 3
        assert len(problem.constraints) >= 1

    def test_integer_variables(self):
        """Test parsing integer variable declarations."""
        problem_text = """
        minimize x1 + x2

        subject to:
            x1 + x2 <= 5

        where:
            integer x1, x2
            x1, x2 >= 0
        """

        problem = parse_lp_problem(problem_text)

        # Check that variables are marked as integer
        for var_name in ["x1", "x2"]:
            if var_name in problem.variables:
                assert problem.variables[var_name].var_type == VariableType.INTEGER

    def test_binary_variables(self):
        """Test parsing binary variable declarations."""
        problem_text = """
        maximize 5*y1 + 3*y2

        subject to:
            y1 + y2 <= 1

        where:
            binary y1, y2
        """

        problem = parse_lp_problem(problem_text)

        # Check that variables are marked as binary
        for var_name in ["y1", "y2"]:
            if var_name in problem.variables:
                assert problem.variables[var_name].var_type == VariableType.BINARY

    def test_comments_ignored(self):
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

        problem = parse_lp_problem(problem_text)

        # Should parse successfully despite comments
        assert problem.objective.direction == ObjectiveDirection.MAXIMIZE
        variables = problem.get_all_variables()
        assert "cat_food" in variables
        assert "cat_toys" in variables

    def test_invalid_syntax_raises_error(self):
        """Test that invalid syntax raises ParseError."""
        invalid_problems = [
            "invalid problem text",
            "maximize",  # Missing expression
            "maximize x1\nsubject to:\n x1 <=",  # Missing RHS
            "maximize x1\nsubject to:\n x1 ?? 5",  # Invalid operator
        ]

        for invalid_text in invalid_problems:
            with pytest.raises(ParseError):
                parse_lp_problem(invalid_text)
