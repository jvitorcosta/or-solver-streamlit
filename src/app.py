"""Main web interface for OR-Solver."""

from pathlib import Path
from typing import Any

import streamlit as st

from config import localization, settings
from solver import engine
from ui import components


def _configure_streamlit_application() -> None:
    """Configure Streamlit application settings and initialize session state."""
    st.set_page_config(
        page_title="SolvedOR",
        page_icon=":material/analytics:",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://github.com/jvitorcosta/or-solver-streamlit",
            "Report a bug": "https://github.com/jvitorcosta/or-solver-streamlit/issues",
            "About": "# OR-Solver\nModern Operations Research Solver",
        },
    )

    # Enhanced styling for Material Icons and modern UI
    st.markdown(
        """
    <style>
    /* Improve Material Icon alignment and spacing */
    .stTabs [data-baseweb="tab-list"] button div p {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Better header spacing for icons */
    h1 span[aria-label], h2 span[aria-label], h3 span[aria-label], h4 span[aria-label] {
        margin-right: 0.5rem;
        vertical-align: middle;
        line-height: 1;
    }

    /* Enhanced visual hierarchy for gallery section */
    .element-container:has(h4) {
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }

    /* Improved pills component styling */
    .stPills div[data-baseweb="button-group"] button {
        border-radius: 1rem;
        padding: 0.5rem 1rem;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


def _initialize_session_state() -> None:
    """Initialize default values for Streamlit session state variables."""
    if "solver_status" not in st.session_state:
        st.session_state.solver_status = "ready"
    if "language" not in st.session_state:
        st.session_state.language = "en"


def render_main_interface(*, language_code: str) -> None:
    """Render the main problem input and solving interface.

    Args:
        language_code: Language code for translations (e.g., 'en', 'pt').
    """
    translations = localization.load_language(language_code)

    # Render application header with translations
    st.markdown(
        f"""
        # :material/analytics: **{translations.app.title}**
        ### :material/auto_graph: {translations.app.subtitle}
        """
    )

    # Configure tab navigation with query parameter support
    tab_names = [
        ":material/work: WORKSPACE",
        ":material/school: PAPER",
        ":material/help: GUIDE",
        ":material/description: README",
        ":material/group: CONTRIBUTING",
    ]

    _process_tab_query_parameters()
    main_tabs = st.tabs(tab_names)

    # Render each tab content
    with main_tabs[0]:
        st.markdown("#### :material/collections: Gallery")
        _render_examples_section(translations=translations)
        st.markdown("---")
        _render_workspace_section(translations=translations)

    with main_tabs[1]:
        _render_thesis_tab()

    with main_tabs[2]:
        _render_guide_tab()

    with main_tabs[3]:
        _render_readme_tab()

    with main_tabs[4]:
        _render_contributing_tab()


def _process_tab_query_parameters() -> None:
    """Process URL query parameters for programmatic tab navigation."""
    query_params = st.query_params
    requested_tab = query_params.get("tab", "").lower()

    tab_mapping = {
        "workspace": 0,
        "paper": 1,
        "guide": 2,
        "readme": 3,
        "contributing": 4,
    }

    if requested_tab in tab_mapping:
        # Clear the query param after processing to avoid sticky state
        if "tab" in st.query_params:
            del st.query_params["tab"]
        st.session_state["active_tab"] = tab_mapping[requested_tab]


def _render_examples_section(*, translations: dict[str, Any]) -> None:
    """Render the optimization problem examples gallery section.

    Args:
        translations: Dictionary containing translation data.
    """
    optimization_examples = settings.load_examples()
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
        "Choose from gallery to get started:",
        options=list(example_template_options.keys()),
        format_func=lambda template_key: example_template_options[template_key],
        help="Select a problem from the gallery to load into your workspace",
    )

    # Display selected example preview and copy functionality
    if selected_template_key and selected_template_key in optimization_examples:
        selected_example = optimization_examples[selected_template_key]

        st.markdown("")
        with st.container(border=True):
            st.markdown(f"**:material/assignment: {selected_example['name']}**")
            st.code(selected_example["problem"], language="text")

            # Center-aligned copy button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                copy_button_pressed = st.button(
                    ":material/content_copy: Copy to Workspace",
                    key="load_template",
                    type="primary",
                    use_container_width=True,
                    help="Copy this example to your workspace",
                )

                if copy_button_pressed:
                    st.session_state.example_text = selected_example["problem"]
                    st.session_state.example_name = selected_example["name"]
                    st.toast(
                        f":material/check_circle: **{selected_example['name']}** copied to workspace!",
                        icon=":material/content_copy:",
                    )
                    st.rerun()


def _render_workspace_section(*, translations: dict[str, Any]) -> None:
    """Render the optimization problem workspace section.

    Args:
        translations: Dictionary containing translation data.
    """
    workspace_text, example_is_loaded = _get_workspace_initial_state()
    current_language_code = st.session_state.get("language", "en")

    _render_workspace_header(example_is_loaded=example_is_loaded)
    _render_optimization_problem_form(
        initial_text=workspace_text, language_code=current_language_code
    )
    _process_optimization_problem_solution()


def _get_workspace_initial_state() -> tuple[str, bool]:
    """Get initial workspace state from session storage.

    Returns:
        Tuple of (initial_text, example_is_loaded) for workspace initialization.
    """
    workspace_initial_text = ""
    example_is_loaded = False

    if "example_text" in st.session_state:
        workspace_initial_text = st.session_state.example_text
        example_is_loaded = True

    return workspace_initial_text, example_is_loaded


def _render_workspace_header(*, example_is_loaded: bool) -> None:
    """Render workspace header with optional clear button.

    Args:
        example_is_loaded: Whether an example is currently loaded in workspace.
    """
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("#### :material/edit_note: Workspace")
        if example_is_loaded and "example_name" in st.session_state:
            st.caption(
                f":material/lightbulb: Using example: **{st.session_state.example_name}**"
            )

    with col2:
        pass  # Removed clear button from here


def _clear_workspace_session_state() -> None:
    """Clear workspace-related session state variables."""
    session_keys_to_remove = ["example_text", "example_name"]
    for key in session_keys_to_remove:
        if key in st.session_state:
            del st.session_state[key]


def _render_optimization_problem_form(*, initial_text: str, language_code: str) -> None:
    """Render form for optimization problem input and submission.

    Args:
        initial_text: Initial text to display in the text area.
        language_code: Language code for processing.
    """
    with st.form("problem_form", border=False):
        problem_text = st.text_area(
            "",
            value=initial_text,
            height=300,
            label_visibility="collapsed",
            placeholder="Enter your optimization problem here...",
            help="Enter your linear or integer programming problem using mathematical notation",
        )

        # Action buttons side by side
        st.markdown("")
        button_col1, button_col2 = st.columns(2)

        with button_col1:
            solve_button_pressed = st.form_submit_button(
                ":material/play_arrow: **Solve Problem**",
                type="primary",
                use_container_width=True,
                help="Analyze and solve your optimization problem",
            )

        with button_col2:
            clear_button_pressed = st.form_submit_button(
                ":material/restart_alt: Clear Workspace",
                type="secondary",
                use_container_width=True,
                help="Clear workspace and start fresh",
            )

        if solve_button_pressed:
            if problem_text.strip():
                _store_problem_for_processing(
                    problem_text=problem_text, language_code=language_code
                )
                st.toast(
                    ":material/calculate: Processing your optimization problem...",
                    icon=":material/play_arrow:",
                )
                st.rerun()
            else:
                st.toast(
                    ":material/warning: Please enter a problem to solve!",
                    icon=":material/error:",
                )

        if clear_button_pressed:
            _clear_workspace_session_state()
            st.toast(
                ":material/refresh: Workspace cleared successfully!",
                icon=":material/restart_alt:",
            )
            st.rerun()


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
        Variable type: 'integer' if integer/binary keywords found, else 'continuous'.
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
            problem_data["language"],
            variable_type=problem_data["variable_type"],
        )
        # Clear the problem data after processing
        del st.session_state.current_problem


def _render_markdown_file(*, file_name: str, not_found_message: str) -> None:
    """Render a markdown file from the project root directory.

    Args:
        file_name: Name of the markdown file to render.
        not_found_message: Warning message to display if file is not found.
    """
    file_path = Path(__file__).parent.parent / file_name

    if file_path.exists():
        try:
            with open(file_path, encoding="utf-8") as markdown_file:
                file_content = markdown_file.read()
            st.markdown(file_content)
        except OSError:
            st.warning(f"Could not read {file_name}.")
    else:
        st.warning(not_found_message)


def _render_thesis_tab() -> None:
    """Render the research thesis PDF tab with viewer and download functionality."""
    resources_directory_path = Path(__file__).parent.parent / "resources"
    thesis_file_path = resources_directory_path / "solvedor_article.pdf"

    if thesis_file_path.exists():
        _render_thesis_header_with_download(thesis_file_path=thesis_file_path)
        st.markdown("")
        _render_pdf_viewer(pdf_file_path=thesis_file_path)
    else:
        st.warning("Thesis PDF not found.")


def _render_thesis_header_with_download(*, thesis_file_path: Path) -> None:
    """Render thesis header section with download button.

    Args:
        thesis_file_path: Path to the thesis PDF file.
    """
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown(
            "**:material/school: Research Thesis - SolvedOR: Modern Operations Research Interface**"
        )
        st.caption(
            "This thesis introduces the OR-Solver application and its methodology."
        )

    with col2:
        try:
            with open(thesis_file_path, "rb") as pdf_file:
                download_pressed = st.download_button(
                    ":material/download: Download PDF",
                    pdf_file.read(),
                    file_name="solvedor_article.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

                if download_pressed:
                    st.toast(
                        ":material/file_download_done: Thesis PDF ready for download!",
                        icon=":material/download:",
                    )
        except OSError:
            st.error("Could not load PDF for download.")


def _render_pdf_viewer(*, pdf_file_path: Path) -> None:
    """Render PDF viewer with error handling.

    Args:
        pdf_file_path: Path to the PDF file to display.
    """
    try:
        st.pdf(str(pdf_file_path))
    except Exception as pdf_error:
        st.warning(
            "PDF viewer not available. You can download the thesis using the button above."
        )
        st.info(f"Error: {str(pdf_error)}")


def _render_readme_tab() -> None:
    """Render the project README tab."""
    _render_markdown_file(
        file_name="README.md", not_found_message="README.md not found."
    )


def _render_contributing_tab() -> None:
    """Render the contributing guide tab."""
    _render_markdown_file(
        file_name="CONTRIBUTING.md", not_found_message="CONTRIBUTING.md not found."
    )


def _render_guide_tab() -> None:
    """Render the visual elements guide tab."""
    st.markdown("### :material/help: Visual Elements Guide")
    st.markdown("Understanding the optimization visualization components:")

    st.markdown("""
    #### :material/search: Visual Elements Explained

    ---

    ##### :material/circle: **Feasible Region (Green Area)**
    - All points that satisfy **ALL** constraints simultaneously
    - The optimal solution must lie within or on the boundary
    - Represents the solution space where all requirements are met

    ---

    ##### :material/straighten: **Constraint Lines (Dashed)**
    - Each colored line represents one constraint boundary
    - Different colors help distinguish between constraints
    - Lines separate feasible from infeasible regions
    - Intersections often contain optimal solutions

    ---

    ##### :material/navigation: **Gradient Arrows (Navy Blue)**
    - Show the direction of steepest increase in objective value
    - Point toward higher objective function values
    - Arrow demonstrates optimization direction
    - Help visualize how the algorithm "climbs" toward optimum

    ---

    ##### :material/star: **Optimal Solution (Gold Star)**
    - The **best possible solution** within the feasible region
    - Located at vertex where gradient direction is blocked by constraints
    - Point where you cannot improve without violating constraints
    - Hover to see exact values and objective score

    ---

    #### :material/target: **How They Work Together**

    1. **Follow gradient arrows** from any point in feasible region
    2. **Arrows guide you** toward increasing objective values
    3. **Optimal star marks** where this process terminates
    4. **Mathematical guarantee**: This vertex gives the best possible solution

    ---

    #### :material/lightbulb: **Pro Tips**

    - **Vertices matter**: Optimal solutions for linear programs always occur at vertices
    - **Direction matters**: Gradient shows which way to improve the objective
    - **Constraints define limits**: Feasible region boundaries prevent further improvement
    - **Interactive exploration**: Hover over points to see exact coordinates and values
    """)


def _run_or_solver_application() -> None:
    """Main application entry point for OR-Solver."""
    _configure_streamlit_application()
    _initialize_session_state()

    language_code, precision_value, solver_name = components.render_sidebar()

    render_main_interface(language_code=language_code)

    st.divider()
    footer_translations = localization.load_language(language_code)
    st.markdown(footer_translations.app.footer)


if __name__ == "__main__":
    _run_or_solver_application()
