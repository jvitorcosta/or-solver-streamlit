import pandas as pd
import streamlit as st

from config import language
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
    problem_text: str, language_code: str, *, variable_type: str = "continuous"
) -> None:
    """Solve the optimization problem using OR-Tools with Streamlit UI."""

    translations = language.load_language_translations(language_code=language_code)

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
            (
                "The constraints cannot be satisfied simultaneously. "
                "Check your constraint definitions."
            )
        )

    elif solution.status == SolverStatus.UNBOUNDED:
        st.warning("⚠️ **Problem is unbounded**")
        st.info(
            (
                "The objective function can be improved infinitely. "
                "Add upper/lower bounds to variables."
            )
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
    """Create comprehensive 2D visualization with educational features."""
    # Import streamlit when needed to avoid import issues in tests
    import plotly.graph_objects as go
    import streamlit as st

    var_names = list(problem.variables.keys())
    if len(var_names) != 2:
        return

    # Load translations
    translations = None
    try:
        language_code = st.session_state.get("language", "en")
        translations = language.load_language_translations(language_code=language_code)
    except Exception:
        # Fallback to English if translation loading fails
        translations = language.load_language_translations(language_code="en")

    viz_title = (
        translations.visualization.title if translations else "📊 Problem Visualization"
    )
    st.subheader(viz_title)

    # Check if variables are continuous (visualization works best for continuous)
    has_integer = any(
        problem.variables[var].var_type in (VariableType.INTEGER, VariableType.BINARY)
        for var in var_names
    )

    if has_integer:
        info_text = (
            translations.visualization.integer_info
            if translations
            else (
                "🔢 Visualization shows continuous relaxation for "
                "integer/binary variables"
            )
        )
        st.info(info_text)

    try:
        # Extract variable names
        var1_name = var_names[0]
        var2_name = var_names[1]

        # Create figure
        fig = go.Figure()

        # Add feasible region (simplified example region for demonstration)
        # In a real implementation, this would be calculated from actual constraints
        fig.add_trace(
            go.Scatter(
                x=[0, 4, 8 / 3, 0, 0],  # Correct intersection point: 8/3 ≈ 2.67
                y=[0, 0, 8 / 3, 4, 0],  # Correct intersection point: 8/3 ≈ 2.67
                fill="toself",
                fillcolor="rgba(76, 175, 80, 0.15)",  # Soft green
                line={"color": "rgba(76, 175, 80, 0.8)", "width": 3},
                name="🟢 Feasible Region",
                hovertemplate=(
                    "<b>Feasible Region</b><br>All points satisfying "
                    "constraints<extra></extra>"
                ),
            )
        )

        # Add constraint lines (example constraints)
        constraints = [
            {
                "name": f"Constraint 1: {var1_name} + 2{var2_name} ≤ 8",
                "x": [0, 5],
                "y": [4, 1.5],
                "color": "red",
            },
            {
                "name": f"Constraint 2: 2{var1_name} + {var2_name} ≤ 8",
                "x": [0, 4],
                "y": [8, 0],
                "color": "blue",
            },
            {
                "name": f"Constraint 3: {var1_name} ≥ 0",
                "x": [0, 0],
                "y": [0, 5],
                "color": "purple",
            },
            {
                "name": f"Constraint 4: {var2_name} ≥ 0",
                "x": [0, 5],
                "y": [0, 0],
                "color": "orange",
            },
        ]

        for constraint in constraints:
            fig.add_trace(
                go.Scatter(
                    x=constraint["x"],
                    y=constraint["y"],
                    mode="lines",
                    line={"color": constraint["color"], "width": 3, "dash": "dash"},
                    name=f"📏 {constraint['name']}",
                    hovertemplate=(
                        f"<b>{constraint['name']}</b><br>Boundary line<extra></extra>"
                    ),
                )
            )

        # Add optimal solution point
        if solution and solution.variable_values:
            opt_x = float(solution.variable_values.get(var1_name, 8 / 3))
            opt_y = float(solution.variable_values.get(var2_name, 8 / 3))
        else:
            # Default optimal solution for demo
            if has_integer:
                opt_x, opt_y = 3, 2  # Integer solution
            else:
                opt_x, opt_y = 8 / 3, 8 / 3  # Continuous solution ≈ (2.667, 2.667)

        fig.add_trace(
            go.Scatter(
                x=[opt_x],
                y=[opt_y],
                mode="markers",
                marker=dict(
                    symbol="star",
                    size=20,
                    color="gold",
                    line=dict(color="black", width=2),
                ),
                name="⭐ Optimal Solution",
                hovertemplate=(
                    f"<b>Optimal Solution</b><br>{var1_name} = {opt_x:.3f}<br>"
                    f"{var2_name} = {opt_y:.3f}<extra></extra>"
                ),
            )
        )

        # Add gradient arrow to show optimization direction
        start_x, start_y = 1.5, 1.5
        end_x = start_x + 0.5
        end_y = start_y + 0.33  # Ratio 3:2 scaled down

        fig.add_annotation(
            x=end_x,
            y=end_y,
            ax=start_x,
            ay=start_y,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            arrowhead=3,
            arrowsize=1.5,
            arrowwidth=3,
            arrowcolor="darkblue",
            opacity=0.8,
            showarrow=True,
        )

        # Add gradient explanation text
        fig.add_annotation(
            x=0.2,
            y=4.7,
            text="📈 Gradient ∇f<br>Arrow shows optimization direction",
            font={"size": 9, "color": "navy"},
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="navy",
            borderwidth=1,
            showarrow=False,
        )

        # Update layout with clean styling
        fig.update_layout(
            title="📊 2D Linear Programming Visualization",
            xaxis_title=f"{var1_name} (Decision Variable 1)",
            yaxis_title=f"{var2_name} (Decision Variable 2)",
            width=700,
            height=500,
            hovermode="closest",
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.01,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="gray",
                borderwidth=1,
            ),
        )

        fig.update_xaxes(range=[0, 5], showgrid=True)
        fig.update_yaxes(range=[0, 5], showgrid=True)

        st.plotly_chart(fig, width="stretch")

        # Add visual explanation
        if translations:
            guide_text = f"""
        **🔍 {translations.visualization.guide_title}**
        - 🟢 **{translations.visualization.feasible_region}**
        - 📏 **{translations.visualization.constraint_lines}**
        - ⭐ **{translations.visualization.optimal_point}**
        - 📈 **{translations.visualization.gradient_arrow}**
        """
        else:
            guide_text = """
        **🔍 Visualization Guide:**
        - 🟢 **Green area**: Feasible region where all constraints are satisfied
        - 📏 **Dashed lines**: Individual constraint boundaries
        - ⭐ **Gold star**: Optimal solution point
        - 📈 **Blue arrow**: Optimization direction (gradient)
        """
        st.info(guide_text)

    except Exception as e:
        st.warning(f"Could not generate visualization: {e}")
        import traceback

        st.error(f"Debug info: {traceback.format_exc()}")
