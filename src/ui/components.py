"""UI components for the OR-Solver interface."""

import streamlit as st

from config import language

# Language display configuration
LANGUAGE_DISPLAY = {"en": "🇺🇸 English", "pt": "🇵🇹 Português"}


def display_language_selection_sidebar(
    *, translations: language.TranslationSchema
) -> None:
    """Display sidebar with interactive language selection interface.

    Args:
        translations: Translation schema for rendering sidebar labels.
    """
    with st.sidebar:
        active_language_code = st.session_state.language

        st.markdown(f"## :material/settings: **{translations.sidebar.settings}**")
        st.markdown(f":material/language: **{translations.sidebar.language}**")

        available_language_codes = [lang.value for lang in language.LanguageCode]
        current_language_index = available_language_codes.index(active_language_code)

        user_selected_language = st.selectbox(
            "",
            options=available_language_codes,
            format_func=lambda code: LANGUAGE_DISPLAY[code],
            index=current_language_index,
            label_visibility="collapsed",
        )

        # Handle language change with immediate feedback
        if user_selected_language != active_language_code:
            st.session_state.language = user_selected_language
            st.toast(
                ":material/language: Language changed!", icon=":material/language:"
            )
            st.rerun()
