"""The gallery and the workspace.

Everything the page shows comes from session state, so a rerun (a language switch, a click
elsewhere) redraws the same results instead of losing them:

- ``problem_text``: the text area's content (its widget key)
- ``example_key``: the gallery pill that is selected (its widget key)
- ``loaded_example``: the gallery example last copied into the workspace, if any
- ``last_result``: the last ``SolveResult``, rendered on every run
"""

import logging

import streamlit as st

from solvedor.config import language, settings
from solvedor.solver.api import solve_text
from solvedor.ui.results import render_results

logger = logging.getLogger(__name__)

MAX_PROBLEM_CHARS = 20_000

_EXAMPLE_ICONS = {
    "production_planning": ":material/factory:",
    "transportation": ":material/local_shipping:",
    "diet_optimization": ":material/restaurant:",
    "resource_allocation": ":material/account_balance:",
    "facility_location": ":material/location_on:",
}


def init_state() -> None:
    for key, default in {
        "problem_text": "",
        "example_key": None,
        "loaded_example": None,
        "last_result": None,
    }.items():
        st.session_state.setdefault(key, default)


def display_optimization_problem_gallery(*, translations: language.TranslationSchema) -> None:
    examples = settings.extract_examples_from_resources_directory()
    if not examples:
        return

    selected = st.pills(
        translations.gallery.choose_text,
        options=[key for key in _EXAMPLE_ICONS if key in examples],
        format_func=lambda key: f"{_EXAMPLE_ICONS[key]} {translations.gallery.examples[key]}",
        help=translations.gallery.select_help,
        key="example_key",
    )
    if selected is None:
        return

    example = examples[selected]
    with st.container(border=True):
        st.markdown(f"**:material/assignment: {translations.gallery.examples[selected]}**")
        st.code(example["problem"], language="text")
        _, center, _ = st.columns([1, 2, 1])
        with center:
            st.button(
                f":material/content_copy: {translations.gallery.copy_button}",
                key="load_example",
                type="primary",
                width="stretch",
                help=translations.gallery.copy_help,
                on_click=_load_example,
                args=(selected, example["problem"]),
            )


def _load_example(key: str, problem_text: str) -> None:
    # Callbacks run before the widgets are drawn, so the text area can still be set here.
    st.session_state.problem_text = problem_text
    st.session_state.loaded_example = key
    st.session_state.last_result = None


def _clear_workspace() -> None:
    st.session_state.problem_text = ""
    st.session_state.example_key = None
    st.session_state.loaded_example = None
    st.session_state.last_result = None


def display_optimization_workspace_interface(*, translations: language.TranslationSchema) -> None:
    st.markdown(f"#### :material/edit_note: {translations.interface.workspace_title}")
    loaded = st.session_state.loaded_example
    if loaded in translations.gallery.examples:
        st.caption(
            f":material/lightbulb: {translations.interface.using_example}: "
            f"**{translations.gallery.examples[loaded]}**"
        )

    with st.form("problem_form", border=False):
        st.text_area(
            translations.interface.workspace_title,
            key="problem_text",
            height=300,
            max_chars=MAX_PROBLEM_CHARS,
            label_visibility="collapsed",
            placeholder=translations.interface.placeholder,
            help=translations.interface.help_text,
        )
        solve_column, clear_column = st.columns(2)
        with solve_column:
            solve_clicked = st.form_submit_button(
                f":material/play_arrow: **{translations.interface.solve_button}**",
                key="solve",
                type="primary",
                width="stretch",
                help=translations.interface.solve_help,
            )
        with clear_column:
            st.form_submit_button(
                f":material/restart_alt: {translations.interface.clear_workspace}",
                key="clear",
                width="stretch",
                help=translations.interface.clear_help,
                on_click=_clear_workspace,
            )

    if solve_clicked:
        _solve(st.session_state.problem_text, translations)

    if st.session_state.last_result is not None:
        render_results(st.session_state.last_result, translations)


def _solve(problem_text: str, translations: language.TranslationSchema) -> None:
    if not problem_text.strip():
        st.session_state.last_result = None
        st.toast(translations.messages.empty_problem_solve, icon=":material/warning:")
        return
    try:
        with st.spinner(translations.status.solving):
            st.session_state.last_result = solve_text(problem_text)
    except Exception:
        # Unexpected bug: keep the traceback in the server log, not on the page.
        logger.exception("solving failed")
        st.session_state.last_result = None
        st.error(translations.errors.solving_error, icon=":material/error:")
