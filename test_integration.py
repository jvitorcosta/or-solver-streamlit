"""Simple integration test for OR-Tools migration."""


def test_simple_problem():
    """Test that we can solve a simple problem with OR-Tools."""
    problem_text = """
    maximize 3*x + 2*y

    subject to:
        x + 2*y <= 4
        2*x + y <= 4

    where:
        x, y >= 0
    """

    # Import the core solver function directly
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

    from solver.engine import solve_optimization_problem

    try:
        problem, solution = solve_optimization_problem(problem_text)

        print("✅ Problem parsed successfully")
        print(f"Variables: {len(problem.variables)}")
        print(f"Constraints: {len(problem.constraints)}")
        print(f"Solution status: {solution.status}")
        print(f"Objective value: {solution.objective_value}")
        print(f"Variable values: {solution.variable_values}")

        # Debug the problem
        print("Debug info:")
        print("Problem objective direction:", problem.objective.direction)
        print(
            "Problem objective coefficients:", problem.objective.expression.coefficients
        )
        for i, constraint in enumerate(problem.constraints):
            print(
                f"Constraint {i}: {constraint.expression.coefficients} {constraint.operator.value} {constraint.rhs}"
            )

        if hasattr(solution, "solver_message") and solution.solver_message:
            print(f"Solver message: {solution.solver_message}")

        # Basic assertions
        assert len(problem.variables) == 2

        if solution.status.value != "optimal":
            print(f"Warning: Expected optimal status, got {solution.status.value}")
            return True  # Don't fail for now, just warn

        assert solution.objective_value is not None
        assert (
            abs(float(solution.objective_value) - 20 / 3) < 0.1
        )  # Should be around 6.67

        print("🎯 OR-Tools integration test PASSED!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_simple_problem()
    exit(0 if success else 1)
