"""UI components for the OR-Solver interface."""

import streamlit as st

from solvedor.config import language


def display_language_selection_sidebar(*, translations: language.TranslationSchema) -> None:
    """Language picker. Its value lives in st.session_state.language, read at the top of a run."""
    labels = {
        language.LanguageCode.ENGLISH.value: translations.language_labels.english,
        language.LanguageCode.PORTUGUESE.value: translations.language_labels.portuguese,
    }
    with st.sidebar:
        st.markdown(f"## :material/settings: **{translations.sidebar.settings}**")
        st.selectbox(
            f":material/language: **{translations.sidebar.language}**",
            options=[code.value for code in language.LanguageCode],
            format_func=labels.__getitem__,
            key="language",
        )
