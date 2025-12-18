import streamlit as st

from config import language
from ui import components, tabs, workspace


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


def _initialize_session_state() -> None:
    """Initialize default values for Streamlit session state variables."""
    if "solver_status" not in st.session_state:
        st.session_state.solver_status = "ready"
    if "language" not in st.session_state:
        st.session_state.language = "en"


def render_main_interface(*, translations: language.TranslationSchema) -> None:
    """Render the main problem input and solving interface.

    Args:
        translations: Loaded translation schema for the current language.
    """

    # Render application header with translations
    st.markdown(
        f"""
        # :material/analytics: **{translations.app.title}**
        ### :material/auto_graph: {translations.app.subtitle}
        """
    )

    # Configure tab navigation with query parameter support
    tab_names = [
        f":material/work: {translations.tabs.workspace}",
        f":material/school: {translations.tabs.paper}",
        f":material/help: {translations.tabs.guide}",
        f":material/description: {translations.tabs.readme}",
        f":material/group: {translations.tabs.contributing}",
    ]

    workspace_tab, paper_tab, guide_tab, readme_tab, contributing_tab = st.tabs(
        tab_names
    )

    with workspace_tab:
        st.markdown(f"#### :material/collections: {translations.gallery.title}")
        workspace.render_examples_section(translations=translations)
        st.markdown("---")
        workspace.render_workspace_section(translations=translations)

    with paper_tab:
        tabs.render_thesis_tab()

    with guide_tab:
        tabs.render_guide_tab()

    with readme_tab:
        tabs._render_markdown_file(
            file_name="README.md", not_found_message="README.md not found."
        )

    with contributing_tab:
        tabs._render_markdown_file(
            file_name="CONTRIBUTING.md", not_found_message="CONTRIBUTING.md not found."
        )


if __name__ == "__main__":
    _configure_streamlit_application()
    _initialize_session_state()

    translations = language.load_language_translations(
        language_code=st.session_state.get("language")
    )

    components.render_sidebar(translations=translations)
    render_main_interface(translations=translations)

    st.divider()
    st.markdown(translations.app.footer)
