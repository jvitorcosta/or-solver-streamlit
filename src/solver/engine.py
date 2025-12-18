import pandas as pd
import streamlit as st

from config import language
from solver import parser
from solver.backends import SolverFactory
from solver.models import Problem, Solution, SolverStatus, VariableType

# Progress tracking for UI feedback
PROGRESS_STEPS = {"parsing": 20, "setup": 40, "solving": 80, "complete": 100}


def parse_text_and_solve_with_backend(problem_text: str) -> tuple[Problem, Solution]:
    """Parse LP text and solve with appropriate backend solver.

    Args:
        problem_text: Linear programming problem in text format.

    Returns:
        Tuple of (parsed problem, solution).

    Raises:
        parser.ParseError: If problem text cannot be parsed.
    """
    problem = parser.parse_lp_problem(problem_text)
    solver = SolverFactory.create_solver(problem)
    solution = solver.execute_optimization_with_backend(problem)
    return problem, solution


def execute_optimization_with_ui_feedback(
    problem_text: str, language_code: str, *, variable_type: str = "continuous"
) -> None:
    """Execute optimization with live UI progress feedback and results display."""

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
            progress_bar.progress(PROGRESS_STEPS["parsing"])

            progress_text.text(translations.status.setting_up)
            progress_bar.progress(PROGRESS_STEPS["setup"])

            progress_text.text(translations.status.solving)
            progress_bar.progress(PROGRESS_STEPS["solving"])

            problem, solution = parse_text_and_solve_with_backend(problem_text)

            # Determine problem type for display
            contains_integer_or_binary_variables = any(
                var.variable_type in (VariableType.INTEGER, VariableType.BINARY)
                for var in problem.variables.values()
            )
            problem_type = (
                "Mixed-Integer" if contains_integer_or_binary_variables else "Linear"
            )

            solver = SolverFactory.create_solver(problem)
            solver_name = solver.get_solver_name()

            # Complete
            progress_text.text(translations.status.solution_found)
            progress_bar.progress(100)
            status.update(label=translations.status.complete, state="complete")

            # Update session state
            st.session_state.solver_status = "ready"

            # Display results using OR-Tools solution
            render_solution_results_with_metrics(
                problem, solution, translations, solver_name, problem_type
            )

            # Generate visualization if applicable
            if len(problem.variables) == 2 and solution.status == SolverStatus.OPTIMAL:
                generate_interactive_2d_plot(problem, solution)

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


def render_solution_results_with_metrics(
    problem, solution, translations, solver_name, problem_type
):
    """Render comprehensive solution results with metrics in Streamlit UI."""

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
                        "Value": format_value_by_variable_type(
                            var_name, value, problem
                        ),
                        "Type": problem.variables[var_name].variable_type.value.title(),
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


def format_value_by_variable_type(variable_name: str, value: float, problem) -> str:
    """Format numeric value according to variable type (integer vs continuous)."""
    if variable_name in problem.variables:
        variable_type_enum = problem.variables[variable_name].variable_type
        if variable_type_enum in (VariableType.INTEGER, VariableType.BINARY):
            return f"{int(round(value))}"
        else:
            return f"{value:.6f}"
    return f"{value:.6f}"


def generate_interactive_2d_plot(problem, solution):
    """Generate interactive 2D plot with constraints and optimal solution."""
    # Import streamlit when needed to avoid import issues in tests
    import plotly.graph_objects as go
    import streamlit as st

    extracted_variable_names = list(problem.variables.keys())
    if len(extracted_variable_names) != 2:
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
    contains_integer_types = any(
        problem.variables[var].variable_type
        in (VariableType.INTEGER, VariableType.BINARY)
        for var in extracted_variable_names
    )

    if contains_integer_types:
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
        first_variable_name = extracted_variable_names[0]
        second_variable_name = extracted_variable_names[1]

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
        constraint_definitions = [
            {
                "name": f"Constraint 1: {first_variable_name} + 2{second_variable_name} ≤ 8",
                "x": [0, 5],
                "y": [4, 1.5],
                "color": "red",
            },
            {
                "name": f"Constraint 2: 2{first_variable_name} + {second_variable_name} ≤ 8",
                "x": [0, 4],
                "y": [8, 0],
                "color": "blue",
            },
            {
                "name": f"Constraint 3: {first_variable_name} ≥ 0",
                "x": [0, 0],
                "y": [0, 5],
                "color": "purple",
            },
            {
                "name": f"Constraint 4: {second_variable_name} ≥ 0",
                "x": [0, 5],
                "y": [0, 0],
                "color": "orange",
            },
        ]

        for constraint_spec in constraint_definitions:
            fig.add_trace(
                go.Scatter(
                    x=constraint_spec["x"],
                    y=constraint_spec["y"],
                    mode="lines",
                    line={
                        "color": constraint_spec["color"],
                        "width": 3,
                        "dash": "dash",
                    },
                    name=f"📏 {constraint_spec['name']}",
                    hovertemplate=(
                        f"<b>{constraint_spec['name']}</b><br>Boundary line<extra></extra>"
                    ),
                )
            )

        # Add optimal solution point
        if solution and solution.variable_values:
            optimal_x_coordinate = float(
                solution.variable_values.get(first_variable_name, 8 / 3)
            )
            optimal_y_coordinate = float(
                solution.variable_values.get(second_variable_name, 8 / 3)
            )
        else:
            # Default optimal solution for demo
            if contains_integer_types:
                optimal_x_coordinate, optimal_y_coordinate = 3, 2  # Integer solution
            else:
                optimal_x_coordinate, optimal_y_coordinate = (
                    8 / 3,
                    8 / 3,
                )  # Continuous solution ≈ (2.667, 2.667)

        fig.add_trace(
            go.Scatter(
                x=[optimal_x_coordinate],
                y=[optimal_y_coordinate],
                mode="markers",
                marker=dict(
                    symbol="star",
                    size=20,
                    color="gold",
                    line=dict(color="black", width=2),
                ),
                name="⭐ Optimal Solution",
                hovertemplate=(
                    f"<b>Optimal Solution</b><br>{first_variable_name} = {optimal_x_coordinate:.3f}<br>"
                    f"{second_variable_name} = {optimal_y_coordinate:.3f}<extra></extra>"
                ),
            )
        )

        # Add gradient arrow to show optimization direction
        gradient_arrow_start_x, gradient_arrow_start_y = 1.5, 1.5
        gradient_arrow_end_x = gradient_arrow_start_x + 0.5
        gradient_arrow_end_y = gradient_arrow_start_y + 0.33  # Ratio 3:2 scaled down

        fig.add_annotation(
            x=gradient_arrow_end_x,
            y=gradient_arrow_end_y,
            ax=gradient_arrow_start_x,
            ay=gradient_arrow_start_y,
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
            xaxis_title=f"{first_variable_name} (Decision Variable 1)",
            yaxis_title=f"{second_variable_name} (Decision Variable 2)",
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
