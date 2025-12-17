"""UI components for the OR-Solver interface."""

from typing import Tuple

import streamlit as st

from config import language_models
from config import localization

def render_sidebar() -> Tuple[str, float, str]:
    """Render the sidebar with language selection."""
    with st.sidebar:
        # Get current language for initial rendering
        language = st.session_state.get('language', 'en')
        translations = localization.load_language(language)
        
        st.markdown(f"## :material/settings: **{translations.sidebar.settings}**")
        
        # Modern language selection with material icons
        st.markdown(f":material/language: **{translations.sidebar.language}**")
        language = st.selectbox(
            "",
            options=[lang.value for lang in language_models.LanguageCode],
            format_func=lambda x: "🇺🇸 English" if x == "en" else "🇧🇷 Português",
            index=0 if language == "en" else 1,
            label_visibility="collapsed"
        )
        
        # Store language in session state
        st.session_state.language = language
        
    
    return language, 4, "glop"  # Removed variable_type since it's now in the main form