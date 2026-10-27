"""Show a SolveResult in Streamlit."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from solvedor.config.language import TranslationSchema
from solvedor.solver.api import SolveResult
from solvedor.solver.models import Problem, Solution, SolverStatus, VariableType


def format_number(value: float, *, whole: bool = False) -> str:
    """Integers without decimals; other values with up to 6 significant decimals, no -0."""
    if whole:
        return str(round(value))
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def render_results(result: SolveResult, t: TranslationSchema) -> None:
    if result.error is not None:
        # st.text, not st.error with Markdown: the message quotes the user's own input.
        st.error(f"**{t.errors.syntax_error}**", icon=":material/error:")
        st.text(str(result.error))
        return
    if result.problem is None or result.solution is None:
        return
    problem, solution = result.problem, result.solution

    st.subheader(t.results.solution)
    columns = st.columns(3)
    columns[0].metric(
        t.results.problem_type, t.results.mixed_integer if problem.is_mip else t.results.linear
    )
    columns[1].metric(t.results.solver, solution.solver_name or "-")
    columns[2].metric(t.results.variables, len(problem.variables))

    match solution.status:
        case SolverStatus.OPTIMAL:
            st.success(t.results.optimal_found, icon=":material/check_circle:")
        case SolverStatus.FEASIBLE:
            st.warning(t.results.feasible_found, icon=":material/timer:")
        case SolverStatus.INFEASIBLE:
            st.error(t.results.problem_infeasible, icon=":material/block:")
            st.info(t.results.infeasible_help)
        case SolverStatus.UNBOUNDED:
            st.warning(t.results.problem_unbounded, icon=":material/all_inclusive:")
            st.info(t.results.unbounded_help)
        case SolverStatus.TIME_LIMIT:
            st.warning(t.results.time_limit, icon=":material/timer_off:")
            st.info(t.results.time_limit_help)
        case SolverStatus.ERROR:
            st.error(t.results.solver_error, icon=":material/error:")
            if solution.solver_message:
                st.text(f"{t.results.error_details}: {solution.solver_message}")

    if solution.is_feasible:
        _render_values(problem, solution, t)


def _render_values(problem: Problem, solution: Solution, t: TranslationSchema) -> None:
    columns = st.columns(2)
    if solution.objective_value is not None:
        columns[0].metric(
            t.results.objective_value,
            format_number(solution.objective_value),
            help=t.results.objective_value_help,
        )
    if solution.solve_time is not None:
        columns[1].metric(
            t.results.solve_time, f"{solution.solve_time:.3f}s", help=t.results.solve_time_help
        )

    st.subheader(t.results.variable_values)
    rows = []
    for name, value in sorted(solution.variable_values.items()):
        variable_type = problem.variables[name].variable_type
        rows.append(
            {
                "variable": name,
                "value": format_number(value, whole=variable_type is not VariableType.CONTINUOUS),
                "type": variable_type.value,
            }
        )
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={
            "variable": st.column_config.TextColumn(t.results.variable),
            "value": st.column_config.TextColumn(t.results.value),
            "type": st.column_config.TextColumn(t.results.type),
        },
    )
