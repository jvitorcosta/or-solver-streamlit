"""UI components for the OR-Solver interface."""

import streamlit as st

from solvedor.config import language

_LANGUAGE_DISPLAY = {"en": "🇺🇸 English", "pt": "🇧🇷 Português"}


def display_language_selection_sidebar(*, translations: language.TranslationSchema) -> None:
    """Language picker. Its value lives in st.session_state.language, read at the top of a run."""
    with st.sidebar:
        st.markdown(f"## :material/settings: **{translations.sidebar.settings}**")
        st.selectbox(
            f":material/language: **{translations.sidebar.language}**",
            options=[code.value for code in language.LanguageCode],
            format_func=lambda code: _LANGUAGE_DISPLAY[code],
            key="language",
        )
