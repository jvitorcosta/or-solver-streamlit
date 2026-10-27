"""The Paper, Guide, Readme and Contributing tabs."""

import logging
from pathlib import Path

import streamlit as st

from solvedor.config import language

logger = logging.getLogger(__name__)

_RESOURCES = Path(__file__).parents[1] / "resources"
_THESIS = _RESOURCES / "solvedor_article.pdf"
_REPO_ROOT = Path(__file__).parents[3]


@st.cache_data(show_spinner=False)
def _read_bytes(path: str) -> bytes:
    """The thesis PDF is a few MB; read it once, not on every rerun."""
    return Path(path).read_bytes()


def display_research_thesis_with_viewer(translations: language.TranslationSchema) -> None:
    if not _THESIS.exists():
        st.warning(translations.errors.thesis_not_found)
        return

    title_column, download_column = st.columns([3, 1])
    with title_column:
        st.markdown(f"**:material/school: {translations.resources.thesis}**")
        st.caption(translations.resources.thesis_caption)
    with download_column:
        try:
            st.download_button(
                f":material/download: {translations.resources.download_pdf}",
                _read_bytes(str(_THESIS)),
                file_name="solvedor_article.pdf",
                mime="application/pdf",
                width="stretch",
            )
        except OSError:
            st.error(f":material/error: {translations.errors.pdf_load_error}")

    st.markdown("")
    try:
        st.pdf(str(_THESIS))
    except Exception:
        # st.pdf needs the optional streamlit[pdf] component; show a message, log the cause.
        logger.exception("could not display the thesis PDF")
        st.warning(translations.errors.pdf_display_error)


def display_guide(language_code: str) -> None:
    """The syntax and plot guide in the page's language (English if there is none)."""
    guide = _RESOURCES / f"guide.{language_code}.md"
    if not guide.exists():
        guide = _RESOURCES / f"guide.{language.LanguageCode.ENGLISH.value}.md"
    st.markdown(guide.read_text(encoding="utf-8"))


def display_markdown_content_from_file(
    file_name: str, translations: language.TranslationSchema
) -> None:
    """A Markdown file from the repository root (README.md, CONTRIBUTING.md)."""
    try:
        st.markdown((_REPO_ROOT / file_name).read_text(encoding="utf-8"))
    except FileNotFoundError:
        st.warning(f"📄 {file_name} {translations.errors.file_not_found}")
    except OSError:
        st.warning(f"⚠️ {translations.errors.file_read_error} {file_name}.")
