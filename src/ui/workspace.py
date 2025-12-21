from pathlib import Path
from typing import Any

import streamlit as st
import yaml

import language
from solver import engine

# Detection keywords for variable types.
_INTEGER_KEYWORDS = frozenset(["integer", "binary"])


def _extract_examples_from_resources_directory(*, language_code: str) -> dict[str, Any]:
    """Extract optimization problem examples from language-specific YAML file.

    Args:
        language_code: Language code (e.g., 'en', 'pt') to load examples for.

    Returns:
        Dictionary containing example optimization problems.
        Returns empty dict if examples file is not found or invalid.
    """
    examples_yaml_file_path = (
        Path(__file__).parents[2] / "resources" / "examples" / f"{language_code}.yaml"
    )

    try:
        examples_file_content = examples_yaml_file_path.read_text(encoding="utf-8")
        return yaml.safe_load(examples_file_content) or {}
    except (FileNotFoundError, yaml.YAMLError, OSError):
        return {}


def display_optimization_problem_gallery(
    *, translations: language.TranslationSchema
) -> None:
    """Display interactive gallery of optimization problem examples with selection.

    Args:
        translations: Translation schema for the current language.
    """
    # Extract language code from translations (default to 'en' if not found)
    language_code = getattr(st.session_state, "language", "en")
    optimization_examples_dictionary = _extract_examples_from_resources_directory(
        language_code=language_code
    )
    if not optimization_examples_dictionary:
        return

    # Create format function that combines icon with translated name from YAML
    def format_example_name(key: str) -> str:
        example_data = optimization_examples_dictionary.get(key, {})
        icon = example_data.get("icon", ":material/help:")
        name = example_data.get("name", key)
        return f"{icon} {name}"

    user_selected_example_key = st.pills(
        translations.gallery.choose_text,
        options=list(optimization_examples_dictionary.keys()),
        format_func=format_example_name,
        help=translations.gallery.select_help,
    )

    # Store selected example in session state for description display
    if user_selected_example_key:
        st.session_state["selected_example"] = optimization_examples_dictionary.get(
            user_selected_example_key
        )
    else:
        st.session_state["selected_example"] = None

    # Early return if no selection
    if (
        not user_selected_example_key
        or user_selected_example_key not in optimization_examples_dictionary
    ):
        return

    # Example preview will be handled separately in the app flow


def display_example_preview_with_copy_button(
    example: dict[str, str], translations: language.TranslationSchema
) -> None:
    """Display example preview with interactive copy-to-workspace functionality.

    Args:
        example: Example data dictionary.
        translations: Translation schema.
    """
    st.markdown("")
    with st.container(border=True):
        st.markdown(f"**:material/assignment: {example['name']}**")
        st.code(example["problem"], language="text")

        # Center-aligned copy button
        _, center_column, _ = st.columns([1, 2, 1])
        with center_column:
            copy_button_clicked = st.button(
                f":material/content_copy: {translations.gallery.copy_button}",
                key="load_template",
                type="primary",
                use_container_width=True,
                help=translations.gallery.copy_help,
            )
            if copy_button_clicked:
                copy_example_to_active_workspace(example, translations)


def copy_example_to_active_workspace(
    example: dict[str, str], translations: language.TranslationSchema
) -> None:
    """Copy selected example to active workspace with user feedback.

    Args:
        example: Example data to copy.
        translations: Translation schema for messages.
    """
    # Update session state with example data
    st.session_state.update(
        {
            "example_text": example["problem"],
            "example_name": example["name"],
            "problem_text": example["problem"],
        }
    )

    # Show success feedback
    st.toast(
        f":material/check_circle: **{example['name']}** {translations.gallery.copied_toast}",
        icon=":material/content_copy:",
    )
    st.rerun()


def display_selected_example_description() -> None:
    """Display description of the currently selected example."""
    selected_example = st.session_state.get("selected_example")
    if selected_example and "description" in selected_example:
        with st.container(border=True):
            st.markdown(f"**:material/info: {selected_example['name']}**")
            st.markdown(selected_example["description"])


def display_selected_example_preview(translations: language.TranslationSchema) -> None:
    """Display mathematical formulation preview of the currently selected example."""
    selected_example = st.session_state.get("selected_example")
    if selected_example and "problem" in selected_example:
        display_example_preview_with_copy_button(selected_example, translations)


def display_optimization_workspace_interface(
    *, translations: language.TranslationSchema
) -> None:
    """Display complete optimization workspace interface with form and processing.

    Args:
        translations: Dictionary containing translation data.
    """
    initial_workspace_text, example_currently_loaded = (
        get_workspace_initial_state_from_session()
    )
    active_language_code = st.session_state.get("language", "en")

    display_workspace_header_with_example_status(
        example_is_loaded=example_currently_loaded, translations=translations
    )
    display_problem_input_form_with_buttons(
        initial_text=initial_workspace_text, language_code=active_language_code
    )
    execute_queued_optimization_problem()


def display_problem_input_form_with_buttons(
    *, initial_text: str, language_code: str
) -> None:
    """Display form with text input area and solve/clear buttons.

    Args:
        initial_text: Initial text to display in the text area.
        language_code: Language code for processing.
    """
    interface_translations = language.load_language_translations(
        language_code=language_code
    )

    with st.form("problem_form", border=False):
        # To ensure we won't erase the text area content on rerun
        # (e.g. changing language)
        if "problem_text" not in st.session_state:
            st.session_state.problem_text = initial_text

        user_input_problem_text = st.text_area(
            "",
            height=300,
            label_visibility="collapsed",
            placeholder=interface_translations.interface.placeholder,
            help=interface_translations.interface.help_text,
            key="problem_text",
        )

        st.markdown("")
        solve_button_column, clear_button_column = st.columns(2)

        with solve_button_column:
            solve_button_clicked = st.form_submit_button(
                (
                    f":material/play_arrow: **{interface_translations.interface.solve_button}**"
                ),
                type="primary",
                use_container_width=True,
                help=interface_translations.interface.solve_help,
            )

        with clear_button_column:
            clear_button_clicked = st.form_submit_button(
                (
                    f":material/restart_alt: {interface_translations.interface.clear_workspace}"
                ),
                type="secondary",
                use_container_width=True,
                help=interface_translations.interface.clear_help,
                on_click=lambda: st.session_state.update(
                    {"problem_text": "", "example_text": None, "example_name": None}
                ),
            )

        if solve_button_clicked:
            if user_input_problem_text.strip():
                queue_problem_text_for_optimization(
                    problem_text=user_input_problem_text, language_code=language_code
                )
                st.toast(
                    (
                        f":material/calculate: {interface_translations.interface.processing_toast}"
                    ),
                    icon=":material/play_arrow:",
                )
                st.rerun()
            else:
                st.toast(
                    (
                        f":material/warning: "
                        f"{interface_translations.messages.empty_problem_solve}!"
                    ),
                    icon=":material/error:",
                )

        if clear_button_clicked:
            st.toast(
                (
                    f":material/refresh: {interface_translations.interface.clear_workspace}!"
                ),
                icon=":material/restart_alt:",
            )


def get_workspace_initial_state_from_session() -> tuple[str, bool]:
    """Retrieve initial workspace state from Streamlit session storage.

    Returns:
        Tuple of (initial_text, example_is_loaded) for workspace
        initialization.
    """
    initial_text_content = ""
    example_currently_loaded = False

    if "example_text" in st.session_state:
        initial_text_content = st.session_state.example_text
        example_currently_loaded = True

    return initial_text_content, example_currently_loaded


def display_workspace_header_with_example_status(
    *, example_is_loaded: bool, translations: language.TranslationSchema
) -> None:
    """Display workspace header showing current example status.

    Args:
        example_is_loaded: Whether an example is currently loaded in workspace.
        translations: Translation schema for localized text.
    """
    st.markdown(f"#### :material/edit_note: {translations.interface.workspace_title}")
    if example_is_loaded and "example_name" in st.session_state:
        st.caption(
            (
                f":material/lightbulb: {translations.interface.using_example}: **{st.session_state.example_name}**"
            )
        )


def queue_problem_text_for_optimization(
    *, problem_text: str, language_code: str
) -> None:
    """Queue optimization problem in session state for processing by solver.

    Args:
        problem_text: The optimization problem text input.
        language_code: Language code for processing.
    """
    # Auto-detect variable type from problem syntax
    detected_variable_type = analyze_text_for_variable_type_hints(
        problem_text=problem_text
    )

    st.session_state.current_problem = {
        "text": problem_text,
        "language": language_code,
        "variable_type": detected_variable_type,
    }


def analyze_text_for_variable_type_hints(*, problem_text: str) -> str:
    """Analyze problem text for variable type hints and keywords.

    Args:
        problem_text: The optimization problem text to analyze.

    Returns:
        Variable type: 'integer' if integer/binary keywords found,
        else 'continuous'.
    """
    problem_text_lowercase = problem_text.lower()
    return (
        "integer"
        if any(
            integer_keyword in problem_text_lowercase
            for integer_keyword in _INTEGER_KEYWORDS
        )
        else "continuous"
    )


def execute_queued_optimization_problem() -> None:
    """Execute and display queued optimization problem solution if available."""
    if "current_problem" in st.session_state:
        queued_problem_data = st.session_state.current_problem
        engine.execute_optimization_with_ui_feedback(
            queued_problem_data["text"],
            language_code=queued_problem_data["language"],
            variable_type=queued_problem_data["variable_type"],
        )
        # Clear the problem data after processing
        del st.session_state.current_problem
