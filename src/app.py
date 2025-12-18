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
            "Get Help": ("https://github.com/jvitorcosta/or-solver-streamlit"),
            "Report a bug": (
                "https://github.com/jvitorcosta/or-solver-streamlit/issues"
            ),
            "About": "# OR-Solver\nModern Operations Research Solver",
        },
    )


def _initialize_session_state() -> None:
    """Initialize default values for Streamlit session state variables."""
    # Use setdefault for cleaner initialization
    st.session_state.setdefault("solver_status", "ready")
    st.session_state.setdefault("language", "en")


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

    # Configure tab navigation - explicit mapping for clarity
    tab_config = {
        "workspace": (":material/work:", translations.tabs.workspace),
        "paper": (":material/school:", translations.tabs.paper),
        "guide": (":material/help:", translations.tabs.guide),
        "readme": (":material/description:", translations.tabs.readme),
        "contributing": (":material/group:", translations.tabs.contributing),
    }

    tab_names = [f"{icon} {label}" for icon, label in tab_config.values()]
    tabs_dict = dict(zip(tab_config.keys(), st.tabs(tab_names), strict=True))

    # Render tabs using clean, functional approach
    with tabs_dict["workspace"]:
        st.markdown(f"#### :material/collections: {translations.gallery.title}")
        workspace.render_examples_section(translations=translations)
        st.markdown("---")
        workspace.render_workspace_section(translations=translations)

    with tabs_dict["paper"]:
        tabs.render_thesis_tab()

    with tabs_dict["guide"]:
        tabs.render_guide_tab()

    with tabs_dict["readme"]:
        tabs.render_markdown_file("README.md")

    with tabs_dict["contributing"]:
        tabs.render_markdown_file("CONTRIBUTING.md")


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
