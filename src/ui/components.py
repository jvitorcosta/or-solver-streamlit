"""UI components for the OR-Solver interface."""

import streamlit as st

from config import language


def render_sidebar(*, translations: language.TranslationSchema) -> None:
    """Render the sidebar with language selection.

    Args:
        translations: Translation schema for rendering sidebar labels.
    """
    with st.sidebar:
        current_language = st.session_state.language

        st.markdown(f"## :material/settings: **{translations.sidebar.settings}**")
        st.markdown(f":material/language: **{translations.sidebar.language}**")

        selected_language = st.selectbox(
            "",
            options=[lang.value for lang in language.LanguageCode],
            format_func=_format_language_option,
            index=0 if current_language == "en" else 1,
            label_visibility="collapsed",
        )

        # Update session state immediately and trigger rerun on language change
        if current_language != selected_language:
            st.session_state.language = selected_language
            st.toast(
                ":material/language: Language changed!", icon=":material/language:"
            )
            st.rerun()


def _format_language_option(lang_code: str) -> str:
    language_map = {"en": "🇺🇸 English", "pt": "🇵🇹 Português"}
    return language_map[lang_code]
