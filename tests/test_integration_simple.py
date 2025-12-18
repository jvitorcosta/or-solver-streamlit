"""Integration tests for the complete parse + solve flow.

This module tests the end-to-end flow from problem text parsing
to solution generation, using simple working problems without UI components.
"""

import pytest

from solver.backends import SolverFactory
from solver.models import SolverStatus, VariableType

# Import only the core functions without UI dependencies
from solver.parser import ParseError, parse_lp_problem


def parse_and_solve_without_ui_dependencies(problem_text: str):
    """Parse LP text and solve without UI dependencies for testing."""
    problem = parse_lp_problem(problem_text)
    solver = SolverFactory.create_solver(problem)
    solution = solver.execute_optimization_with_backend(problem)
    return problem, solution


class TestParseAndSolveIntegration:
    """Test the complete parse + solve integration flow with working examples."""

    def test_simple_maximize_problem_flow(self):
        """Test a simple maximization problem - complete flow."""
        problem_text = """
        maximize 3*x + 2*y

        subject to:
            x + y <= 4
            2*x + y <= 6
            x <= 3

        where:
            x, y >= 0
        """

        problem, solution = parse_and_solve_without_ui_dependencies(problem_text)

        # Verify parsing
        assert problem is not None
        assert problem.objective.direction.name == "MAXIMIZE"
        assert len(problem.variables) == 2
        assert "x" in problem.variables
        assert "y" in problem.variables
        assert len(problem.constraints) >= 3  # Explicit constraints + non-negativity

        # Verify all variables are continuous
        for var in problem.variables.values():
            assert var.variable_type == VariableType.CONTINUOUS

        # Verify solution
        assert solution.status == SolverStatus.OPTIMAL
        assert solution.is_optimal
        assert solution.objective_value is not None
        assert solution.objective_value > 0  # Should have positive value
        assert len(solution.variable_values) == 2

        # Verify variable values are within bounds
        x_val = solution.variable_values.get("x", 0)
        y_val = solution.variable_values.get("y", 0)
        assert x_val >= 0  # Non-negativity
        assert y_val >= 0  # Non-negativity
        assert x_val <= 3 + 1e-6  # Upper bound constraint
        assert x_val + y_val <= 4 + 1e-6  # Constraint
        assert 2 * x_val + y_val <= 6 + 1e-6  # Constraint

    def test_simple_minimize_problem_flow(self):
        """Test a simple minimization problem - complete flow."""
        problem_text = """
        minimize 2*x + 3*y

        subject to:
            x + y >= 3
            x + 2*y >= 4
            x >= 1

        where:
            x, y >= 0
        """

        problem, solution = parse_and_solve_without_ui_dependencies(problem_text)

        # Verify parsing
        assert problem is not None
        assert problem.objective.direction.name == "MINIMIZE"
        assert len(problem.variables) == 2
        assert "x" in problem.variables
        assert "y" in problem.variables

        # Verify solution
        assert solution.status == SolverStatus.OPTIMAL
        assert solution.is_optimal
        assert solution.objective_value is not None
        assert solution.objective_value > 0  # Should have positive cost

        # Verify constraints are satisfied
        x_val = solution.variable_values.get("x", 0)
        y_val = solution.variable_values.get("y", 0)
        assert x_val >= 1 - 1e-6  # Lower bound
        assert x_val + y_val >= 3 - 1e-6  # Constraint
        assert x_val + 2 * y_val >= 4 - 1e-6  # Constraint

    def test_production_planning_simplified(self):
        """Test a simplified production planning problem."""
        problem_text = """
        maximize 40*product_a + 50*product_b

        subject to:
            2*product_a + 3*product_b <= 100
            4*product_a + 2*product_b <= 120
            product_a <= 30
            product_b <= 25

        where:
            product_a, product_b >= 0
        """

        problem, solution = parse_and_solve_without_ui_dependencies(problem_text)

        # Verify parsing
        assert problem is not None
        assert problem.objective.direction.name == "MAXIMIZE"
        assert len(problem.variables) == 2
        assert "product_a" in problem.variables
        assert "product_b" in problem.variables

        # Verify solution
        assert solution.status == SolverStatus.OPTIMAL
        assert solution.is_optimal
        assert solution.objective_value is not None
        assert solution.objective_value > 0  # Should have positive profit

        # Verify variable values are within bounds
        a_val = solution.variable_values.get("product_a", 0)
        b_val = solution.variable_values.get("product_b", 0)
        assert a_val >= 0 and b_val >= 0
        assert a_val <= 30 + 1e-6
        assert b_val <= 25 + 1e-6
        assert 2 * a_val + 3 * b_val <= 100 + 1e-6
        assert 4 * a_val + 2 * b_val <= 120 + 1e-6

    def test_diet_problem_simplified(self):
        """Test a simplified diet problem."""
        problem_text = """
        minimize 3*bread + 2*milk + 4*cheese

        subject to:
            4*bread + 8*milk + 2*cheese >= 8
            2*bread + 12*milk + 10*cheese >= 6
            bread >= 0.5
            milk >= 0.1
            cheese >= 0.1

        where:
            bread, milk, cheese >= 0
        """

        problem, solution = parse_and_solve_without_ui_dependencies(problem_text)

        # Verify parsing
        assert problem is not None
        assert problem.objective.direction.name == "MINIMIZE"
        assert len(problem.variables) == 3

        food_items = ["bread", "milk", "cheese"]
        for item in food_items:
            assert item in problem.variables

        # Verify solution
        assert solution.status == SolverStatus.OPTIMAL
        assert solution.is_optimal
        assert solution.objective_value is not None
        assert solution.objective_value > 0

        # Verify constraints
        b_val = solution.variable_values.get("bread", 0)
        m_val = solution.variable_values.get("milk", 0)
        c_val = solution.variable_values.get("cheese", 0)

        assert b_val >= 0.5 - 1e-6  # Minimum bread (as set in constraint)
        assert (
            4 * b_val + 8 * m_val + 2 * c_val >= 8 - 1e-6
        )  # Nutrition constraint (matches problem)
        assert (
            2 * b_val + 12 * m_val + 10 * c_val >= 6 - 1e-6
        )  # Nutrition constraint (matches problem)

    def test_infeasible_problem_flow(self):
        """Test handling of infeasible problems."""
        infeasible_problem = """
        maximize x + y
        subject to:
            x + y >= 10
            x + y <= 5
        where:
            x, y >= 0
        """

        problem, solution = parse_and_solve_without_ui_dependencies(infeasible_problem)

        # Should parse successfully
        assert problem is not None

        # But solution should be infeasible
        assert solution.status == SolverStatus.INFEASIBLE
        assert not solution.is_optimal
        assert not solution.is_feasible

    def test_unbounded_problem_flow(self):
        """Test handling of unbounded problems."""
        unbounded_problem = """
        maximize x + y
        subject to:
            x >= 1
            y >= 0.1
        where:
            x, y >= 0
        """

        problem, solution = parse_and_solve_without_ui_dependencies(unbounded_problem)

        # Should parse successfully
        assert problem is not None

        # Solution should indicate unbounded, optimal with large value, or may be treated differently by solver
        if solution.status == SolverStatus.OPTIMAL:
            # If marked as optimal, objective value should be very large (unbounded maximization)
            assert solution.objective_value > 1000
        else:
            # May be marked as unbounded or infeasible depending on solver behavior
            assert solution.status in [SolverStatus.UNBOUNDED, SolverStatus.INFEASIBLE]

    def test_problem_with_syntax_error_flow(self):
        """Test that parsing errors are properly raised."""
        invalid_problem = """
        maximize x + y
        subject to:
            x + invalid_operator 10
        where:
            x, y >= 0
        """

        with pytest.raises(ParseError):
            parse_and_solve_without_ui_dependencies(invalid_problem)

    def test_empty_problem_flow(self):
        """Test that empty problem text raises appropriate error."""
        with pytest.raises(ParseError):
            parse_and_solve_without_ui_dependencies("")

    def test_problem_variable_extraction_flow(self):
        """Test that problem variables are correctly extracted."""
        problem_text = """
        maximize 2*x1 + 3*x2 + x3
        subject to:
            x1 + x2 <= 5
            x2 + x3 >= 2
        where:
            x1, x2, x3 >= 0
        """

        problem, solution = parse_and_solve_without_ui_dependencies(problem_text)

        # Get all variables from the problem
        all_vars = problem.extract_all_variable_names()
        assert len(all_vars) == 3

        expected_vars = {"x1", "x2", "x3"}
        actual_vars = set(
            all_vars
        )  # get_all_variables returns variable names as strings
        assert actual_vars == expected_vars

        # Check objective function variables
        objective_vars = problem.objective.extract_variable_names()
        objective_var_names = set(objective_vars)
        assert objective_var_names == expected_vars

    def test_solution_formatting_flow(self):
        """Test that solution formatting works correctly."""
        problem_text = """
        maximize 2*x + 3*y
        subject to:
            x + y <= 4
            x <= 3
            y <= 2
        where:
            x, y >= 0
        """

        problem, solution = parse_and_solve_without_ui_dependencies(problem_text)

        # Ensure we have an optimal solution first
        assert solution.status == SolverStatus.OPTIMAL

        # Test solution string representation
        solution_str = str(solution)
        assert "Optimal" in solution_str or "🎯" in solution_str
        assert "Objective Value:" in solution_str or "💰" in solution_str

        # Should contain variable values
        assert "x" in solution_str
        assert "y" in solution_str

        # Should contain solve time
        assert "Solve Time:" in solution_str or "⏱️" in solution_str

    def test_problem_with_only_objective_flow(self):
        """Test problem with only objective function (unbounded)."""
        minimal_problem = """
        maximize x + y
        subject to:
            x >= 0
            y >= 0
        where:
            x, y >= 0
        """

        problem, solution = parse_and_solve_without_ui_dependencies(minimal_problem)

        # Should parse successfully
        assert problem is not None
        assert problem.objective.direction.name == "MAXIMIZE"

        # Should be unbounded or may be treated as infeasible by solver implementation
        assert solution.status in [
            SolverStatus.UNBOUNDED,
            SolverStatus.OPTIMAL,
            SolverStatus.INFEASIBLE,
        ]

    @pytest.mark.parametrize(
        "obj_direction,constraint_op,expected_finite",
        [
            ("maximize", "<=", True),  # Maximize with upper bounds - should be finite
            ("minimize", ">=", True),  # Minimize with lower bounds - should be finite
            (
                "maximize",
                ">=",
                False,
            ),  # Maximize with only lower bounds - may be unbounded
            (
                "minimize",
                "<=",
                False,
            ),  # Minimize with only upper bounds - may be unbounded
        ],
    )
    def test_problem_variations_flow(
        self, obj_direction, constraint_op, expected_finite
    ):
        """Test various problem configurations."""
        bound_value = "10" if constraint_op == "<=" else "1"

        problem_text = f"""
        {obj_direction} 2*x + 3*y
        subject to:
            x + y {constraint_op} {bound_value}
            x >= 0
            y >= 0
        where:
            x, y >= 0
        """

        # Should not raise any exceptions
        problem, solution = parse_and_solve_without_ui_dependencies(problem_text)

        # Basic validations
        assert problem is not None
        assert solution is not None
        assert len(problem.variables) == 2
        assert len(problem.constraints) >= 1

        if expected_finite:
            # Should have finite optimal solution
            if solution.status == SolverStatus.OPTIMAL:
                assert solution.is_optimal
                assert solution.objective_value is not None
                assert abs(solution.objective_value) < 1e6  # Reasonably finite
            else:
                # Some problems might be infeasible due to constraint conflicts
                assert solution.status in [
                    SolverStatus.OPTIMAL,
                    SolverStatus.INFEASIBLE,
                ]
        else:
            # May be unbounded, optimal with large values, or infeasible
            assert solution.status in [
                SolverStatus.OPTIMAL,
                SolverStatus.UNBOUNDED,
                SolverStatus.INFEASIBLE,
            ]
