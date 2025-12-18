"""UI components for the OR-Solver interface."""

import streamlit as st

from config import language

# Language display configuration
LANGUAGE_DISPLAY = {"en": "🇺🇸 English", "pt": "🇵🇹 Português"}


def render_sidebar(*, translations: language.TranslationSchema) -> None:
    """Render the sidebar with language selection.

    Args:
        translations: Translation schema for rendering sidebar labels.
    """
    with st.sidebar:
        current_language = st.session_state.language

        st.markdown(f"## :material/settings: **{translations.sidebar.settings}**")
        st.markdown(f":material/language: **{translations.sidebar.language}**")

        language_options = [lang.value for lang in language.LanguageCode]
        current_index = language_options.index(current_language)

        selected_language = st.selectbox(
            "",
            options=language_options,
            format_func=lambda code: LANGUAGE_DISPLAY[code],
            index=current_index,
            label_visibility="collapsed",
        )

        # Handle language change with immediate feedback
        if selected_language != current_language:
            st.session_state.language = selected_language
            st.toast(
                ":material/language: Language changed!", icon=":material/language:"
            )
            st.rerun()


# Language formatting now handled inline with lambda and constant
