"""Modern optimization engine using Google OR-Tools."""

from __future__ import annotations

from solver import parser
from solver.backends import SolverFactory
from solver.models import Problem, Solution, SolverStatus, VariableType


def solve_optimization_problem(problem_text: str) -> tuple[Problem, Solution]:
    """Core solver function that returns Problem and Solution objects."""
    # Parse the problem
    problem = parser.parse_lp_problem(problem_text)

    # Create and use solver
    solver = SolverFactory.create_solver(problem)
    solution = solver.solve(problem)

    return problem, solution


def solve_problem(
    problem_text: str, language: str, *, variable_type: str = "continuous"
) -> None:
    """Solve the optimization problem using OR-Tools with Streamlit UI."""
    # Import UI dependencies only when this function is called
    import streamlit as st

    from config import localization

    translations = localization.load_language(language)

    if not problem_text or not problem_text.strip():
        st.warning(translations.messages.empty_problem_solve)
        return

    # Update session state
    st.session_state.solver_status = "solving"

    # Use modern status container with progress bar
    with st.status("Solving optimization problem...", expanded=True) as status:
        progress_bar = st.progress(0)
        progress_text = st.empty()

        try:
            # Use core solver function
            progress_text.text(translations.status.parsing)
            progress_bar.progress(20)

            progress_text.text(translations.status.setting_up)
            progress_bar.progress(40)

            progress_text.text(translations.status.solving)
            progress_bar.progress(80)

            problem, solution = solve_optimization_problem(problem_text)

            # Determine problem type for display
            has_integer_vars = any(
                var.var_type in (VariableType.INTEGER, VariableType.BINARY)
                for var in problem.variables.values()
            )
            problem_type = "Mixed-Integer" if has_integer_vars else "Linear"

            solver = SolverFactory.create_solver(problem)
            solver_name = solver.get_solver_name()

            # Complete
            progress_text.text(translations.status.solution_found)
            progress_bar.progress(100)
            status.update(label=translations.status.complete, state="complete")

            # Update session state
            st.session_state.solver_status = "ready"

            # Display results using OR-Tools solution
            _display_ortools_results(
                problem, solution, translations, solver_name, problem_type
            )

            # Generate visualization if applicable
            if len(problem.variables) == 2 and solution.status == SolverStatus.OPTIMAL:
                _create_2d_visualization(problem, solution)

            # Success toast
            st.toast(
                ":material/check_circle: Problem solved successfully!",
                icon=":material/analytics:",
            )

        except parser.ParseError as e:
            # Update session state and show error.
            st.session_state.solver_status = "ready"
            status.update(label=translations.status.syntax_error, state="error")

            st.error(f"**{translations.errors.syntax_error}:** {e}")
            st.toast(
                f":material/error: {translations.status.syntax_error}",
                icon=":material/warning:",
            )

        except Exception as e:
            # Update session state and show error
            st.session_state.solver_status = "ready"
            status.update(label=translations.status.solving_failed, state="error")

            st.error(f"**{translations.errors.solving_error}:** {e}")
            st.toast(
                f":material/error_outline: {translations.status.solving_failed}",
                icon=":material/bug_report:",
            )


def _display_ortools_results(
    problem, solution, translations, solver_name, problem_type
):
    """Display OR-Tools solution results in Streamlit interface."""
    import pandas as pd
    import streamlit as st

    st.subheader(f"🎯 {translations.results.solution}")

    # Solver information
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Problem Type", problem_type)
    with col2:
        st.metric("Solver", solver_name.replace("OR-Tools ", ""))
    with col3:
        st.metric("Variables", len(problem.variables))

    if solution.status == SolverStatus.OPTIMAL:
        st.success(f"✅ **{translations.results.optimal_found}**")

        # Objective value
        st.metric(
            "Objective Value",
            f"{solution.objective_value:.6f}" if solution.objective_value else "N/A",
            help="Optimal value of the objective function",
        )

        # Solve time
        if solution.solve_time:
            st.metric(
                "Solve Time",
                f"{solution.solve_time:.3f}s",
                help="Time taken to solve the problem",
            )

        # Variable values table
        st.subheader(f"📊 {translations.results.variable_values}")
        if solution.variable_values:
            df_vars = pd.DataFrame(
                [
                    {
                        "Variable": var_name,
                        "Value": _format_variable_value(var_name, value, problem),
                        "Type": problem.variables[var_name].var_type.value.title(),
                    }
                    for var_name, value in solution.variable_values.items()
                ]
            )

            st.dataframe(
                df_vars,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Variable": st.column_config.TextColumn("Variable", width="medium"),
                    "Value": st.column_config.NumberColumn(
                        "Value", width="medium", format="%.6f"
                    ),
                    "Type": st.column_config.TextColumn("Type", width="small"),
                },
            )
        else:
            st.info("No variable values to display")

    elif solution.status == SolverStatus.INFEASIBLE:
        st.error("❌ **Problem is infeasible**")
        st.info(
            "The constraints cannot be satisfied simultaneously. Check your constraint definitions."
        )

    elif solution.status == SolverStatus.UNBOUNDED:
        st.warning("⚠️ **Problem is unbounded**")
        st.info(
            "The objective function can be improved infinitely. Add upper/lower bounds to variables."
        )

    else:
        st.error(f"❌ **Solver Error**: {solution.status.value}")
        if hasattr(solution, "solver_message") and solution.solver_message:
            st.info(f"Details: {solution.solver_message}")


def _format_variable_value(var_name: str, value: float, problem) -> str:
    """Format variable value based on its type."""
    if var_name in problem.variables:
        var_type = problem.variables[var_name].var_type
        if var_type in (VariableType.INTEGER, VariableType.BINARY):
            return f"{int(round(value))}"
        else:
            return f"{value:.6f}"
    return f"{value:.6f}"


def _create_2d_visualization(problem, solution):
    """Create 2D visualization for two-variable linear programming problems."""
    # Import streamlit when needed to avoid import issues in tests
    import streamlit as st

    var_names = list(problem.variables.keys())
    if len(var_names) != 2:
        return

    st.subheader("📈 Problem Visualization")

    # Check if variables are continuous (visualization works best for continuous)
    has_integer = any(
        problem.variables[var].var_type in (VariableType.INTEGER, VariableType.BINARY)
        for var in var_names
    )

    if has_integer:
        st.info(
            "🔢 Visualization shows continuous relaxation for integer/binary variables"
        )

    try:
        # TODO: Implement 2D visualization for OR-Tools solutions
        st.info("2D visualization will be implemented in a future update")
    except Exception as e:
        st.warning(f"Could not generate visualization: {e}")
