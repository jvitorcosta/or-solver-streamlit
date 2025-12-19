import streamlit as st

from config import language
from ui import components, tabs, workspace


def initialize_streamlit_page_configuration() -> None:
    """Initialize Streamlit page configuration with title, layout, and menu options."""
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


def setup_default_session_state_variables() -> None:
    """Setup default values for Streamlit session state variables across application."""
    # Use setdefault for cleaner initialization
    st.session_state.setdefault("solver_status", "ready")
    st.session_state.setdefault("language", "en")


def display_tabbed_application_interface(
    *, translations: language.TranslationSchema
) -> None:
    """Display main tabbed interface with workspace, documentation, and guide sections.

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
    tab_configuration_mapping = {
        "workspace": (":material/work:", translations.tabs.workspace),
        "paper": (":material/school:", translations.tabs.paper),
        "guide": (":material/help:", translations.tabs.guide),
        "readme": (":material/description:", translations.tabs.readme),
        "contributing": (":material/group:", translations.tabs.contributing),
    }

    formatted_tab_names = [
        f"{icon} {label}" for icon, label in tab_configuration_mapping.values()
    ]
    streamlit_tabs_dictionary = dict(
        zip(tab_configuration_mapping.keys(), st.tabs(formatted_tab_names), strict=True)
    )

    # Render tabs using clean, functional approach
    with streamlit_tabs_dictionary["workspace"]:
        st.markdown(f"#### :material/collections: {translations.gallery.title}")
        workspace.display_optimization_problem_gallery(translations=translations)
        st.markdown("---")
        workspace.display_optimization_workspace_interface(translations=translations)

    with streamlit_tabs_dictionary["paper"]:
        tabs.display_research_thesis_with_viewer(translations)

    with streamlit_tabs_dictionary["guide"]:
        tabs.display_visualization_elements_guide()

    with streamlit_tabs_dictionary["readme"]:
        tabs.display_markdown_content_from_file("README.md", translations)

    with streamlit_tabs_dictionary["contributing"]:
        tabs.display_markdown_content_from_file("CONTRIBUTING.md", translations)


if __name__ == "__main__":
    initialize_streamlit_page_configuration()
    setup_default_session_state_variables()

    current_language_translations = language.load_language_translations(
        language_code=st.session_state.get("language")
    )

    components.display_language_selection_sidebar(
        translations=current_language_translations
    )
    display_tabbed_application_interface(translations=current_language_translations)

    st.divider()
    st.markdown(current_language_translations.app.footer)
