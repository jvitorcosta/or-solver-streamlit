import streamlit as st

from solvedor.config import language
from solvedor.ui import components, tabs, workspace

_REPO = "https://github.com/jvitorcosta/or-solver-streamlit"


def initialize_streamlit_page_configuration(translations: language.TranslationSchema) -> None:
    st.set_page_config(
        page_title="SolvedOR",
        page_icon=":material/analytics:",
        layout="wide",
        initial_sidebar_state="auto",  # closed on phones and when embedded in a narrow page
        menu_items={
            "Get Help": _REPO,
            "Report a bug": f"{_REPO}/issues",
            "About": f"### {translations.app.title}\n{translations.app.about}",
        },
    )


def setup_default_session_state_variables() -> None:
    st.session_state.setdefault("language", language.LanguageCode.ENGLISH.value)
    workspace.init_state()


def display_tabbed_application_interface(
    *, translations: language.TranslationSchema, language_code: str
) -> None:
    st.markdown(
        f"""
        # :material/analytics: **{translations.app.title}**
        ### :material/auto_graph: {translations.app.subtitle}
        """
    )

    workspace_tab, paper_tab, guide_tab, readme_tab, contributing_tab = st.tabs(
        [
            f":material/work: {translations.tabs.workspace}",
            f":material/school: {translations.tabs.paper}",
            f":material/help: {translations.tabs.guide}",
            f":material/description: {translations.tabs.readme}",
            f":material/group: {translations.tabs.contributing}",
        ]
    )

    with workspace_tab:
        st.markdown(f"#### :material/collections: {translations.gallery.title}")
        workspace.display_optimization_problem_gallery(translations=translations)
        st.markdown("---")
        workspace.display_optimization_workspace_interface(translations=translations)

    with paper_tab:
        tabs.display_research_thesis_with_viewer(translations)

    with guide_tab:
        tabs.display_guide(language_code)

    with readme_tab:
        tabs.display_markdown_content_from_file("README.md", translations)

    with contributing_tab:
        tabs.display_markdown_content_from_file("CONTRIBUTING.md", translations)


def main() -> None:
    """Render the whole app (called by streamlit_app.py)."""
    setup_default_session_state_variables()
    language_code = st.session_state.language
    translations = language.load_language_translations(language_code=language_code)
    initialize_streamlit_page_configuration(translations)

    components.display_language_selection_sidebar(translations=translations)
    display_tabbed_application_interface(translations=translations, language_code=language_code)

    st.divider()
    st.markdown(translations.app.footer)


if __name__ == "__main__":
    main()
