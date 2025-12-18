from typing import Any

import streamlit as st

from config import language, settings
from solver import engine


def render_examples_section(*, translations: dict[str, Any]) -> None:
    """Render the optimization problem examples gallery section.

    Args:
        translations: Dictionary containing translation data.
    """
    optimization_examples = settings.load_optimization_examples()
    if not optimization_examples:
        return

    example_template_options = {
        "production_planning": ":material/factory: Production",
        "transportation": ":material/local_shipping: Transport",
        "diet_optimization": ":material/restaurant: Diet",
        "resource_allocation": ":material/account_balance: Resource",
        "facility_location": ":material/location_on: Location",
    }

    selected_template_key = st.pills(
        translations.gallery.choose_text,
        options=list(example_template_options.keys()),
        format_func=lambda key: example_template_options[key],
        help=translations.gallery.select_help,
    )

    # Display selected example preview and copy functionality
    if selected_template_key and selected_template_key in optimization_examples:
        selected_example = optimization_examples[selected_template_key]

        st.markdown("")
        with st.container(border=True):
            st.markdown(f"**:material/assignment: {selected_example['name']}**")
            st.code(selected_example["problem"], language="text")

            # Center-aligned copy button
            _, center_col, _ = st.columns([1, 2, 1])
            with center_col:
                copy_button_pressed = st.button(
                    (f":material/content_copy: {translations.gallery.copy_button}"),
                    key="load_template",
                    type="primary",
                    use_container_width=True,
                    help=translations.gallery.copy_help,
                )

                if copy_button_pressed:
                    st.session_state.example_text = selected_example["problem"]
                    st.session_state.example_name = selected_example["name"]
                    # Update text area directly
                    st.session_state.problem_text = selected_example["problem"]
                    st.toast(
                        (
                            f":material/check_circle: **{selected_example['name']}** "
                            f"{translations.gallery.copied_toast}"
                        ),
                        icon=":material/content_copy:",
                    )
                    st.rerun()


def render_workspace_section(*, translations: dict[str, Any]) -> None:
    """Render the optimization problem workspace section.

    Args:
        translations: Dictionary containing translation data.
    """
    workspace_text, example_is_loaded = _get_workspace_initial_state()
    current_language_code = st.session_state.get("language", "en")

    _render_workspace_header(example_is_loaded=example_is_loaded)
    render_optimization_problem_form(
        initial_text=workspace_text, language_code=current_language_code
    )
    _process_optimization_problem_solution()


def render_optimization_problem_form(*, initial_text: str, language_code: str) -> None:
    """Render form for optimization problem input and submission.

    Args:
        initial_text: Initial text to display in the text area.
        language_code: Language code for processing.
    """
    translations = language.load_language_translations(language_code=language_code)

    with st.form("problem_form", border=False):
        # To ensure we won't erase the text area content on rerun
        # (e.g. changing language)
        if "problem_text" not in st.session_state:
            st.session_state.problem_text = initial_text

        problem_text = st.text_area(
            "",
            height=300,
            label_visibility="collapsed",
            placeholder=translations.interface.placeholder,
            help=translations.interface.help_text,
            key="problem_text",
        )

        st.markdown("")
        button_col1, button_col2 = st.columns(2)

        with button_col1:
            solve_button_pressed = st.form_submit_button(
                (f":material/play_arrow: **{translations.interface.solve_button}**"),
                type="primary",
                use_container_width=True,
                help=translations.interface.solve_help,
            )

        with button_col2:
            clear_button_pressed = st.form_submit_button(
                (f":material/restart_alt: {translations.interface.clear_workspace}"),
                type="secondary",
                use_container_width=True,
                help=translations.interface.clear_help,
                on_click=lambda: st.session_state.update(
                    {"problem_text": "", "example_text": None, "example_name": None}
                ),
            )

        if solve_button_pressed:
            if problem_text.strip():
                _store_problem_for_processing(
                    problem_text=problem_text, language_code=language_code
                )
                st.toast(
                    (f":material/calculate: {translations.interface.processing_toast}"),
                    icon=":material/play_arrow:",
                )
                st.rerun()
            else:
                st.toast(
                    (
                        f":material/warning: "
                        f"{translations.messages.empty_problem_solve}!"
                    ),
                    icon=":material/error:",
                )

        if clear_button_pressed:
            st.toast(
                (f":material/refresh: {translations.interface.clear_workspace}!"),
                icon=":material/restart_alt:",
            )


def _get_workspace_initial_state() -> tuple[str, bool]:
    """Get initial workspace state from session storage.

    Returns:
        Tuple of (initial_text, example_is_loaded) for workspace
        initialization.
    """
    workspace_initial_text = ""
    example_is_loaded = False

    if "example_text" in st.session_state:
        workspace_initial_text = st.session_state.example_text
        example_is_loaded = True

    return workspace_initial_text, example_is_loaded


def _render_workspace_header(*, example_is_loaded: bool) -> None:
    """Render workspace header.

    Args:
        example_is_loaded: Whether an example is currently loaded in workspace.
    """
    st.markdown("#### :material/edit_note: Workspace")
    if example_is_loaded and "example_name" in st.session_state:
        st.caption(
            (f":material/lightbulb: Using example: **{st.session_state.example_name}**")
        )


def _store_problem_for_processing(*, problem_text: str, language_code: str) -> None:
    """Store optimization problem data in session state for processing.

    Args:
        problem_text: The optimization problem text input.
        language_code: Language code for processing.
    """
    # Auto-detect variable type from problem syntax
    variable_type = _detect_variable_type_from_text(problem_text=problem_text)

    st.session_state.current_problem = {
        "text": problem_text,
        "language": language_code,
        "variable_type": variable_type,
    }


def _detect_variable_type_from_text(*, problem_text: str) -> str:
    """Detect variable type from optimization problem text.

    Args:
        problem_text: The optimization problem text to analyze.

    Returns:
        Variable type: 'integer' if integer/binary keywords found,
        else 'continuous'.
    """
    text_lower = problem_text.lower()
    integer_keywords = ["integer", "binary"]

    return (
        "integer"
        if any(keyword in text_lower for keyword in integer_keywords)
        else "continuous"
    )


def _process_optimization_problem_solution() -> None:
    """Process and display optimization problem solution if available."""
    if "current_problem" in st.session_state:
        problem_data = st.session_state.current_problem
        engine.solve_problem(
            problem_data["text"],
            language_code=problem_data["language"],
            variable_type=problem_data["variable_type"],
        )
        # Clear the problem data after processing
        del st.session_state.current_problem
