from pathlib import Path

import streamlit as st

from config import language


def display_research_thesis_with_viewer(
    translations: language.TranslationSchema = None,
) -> None:
    """Display research thesis PDF with integrated viewer and download functionality."""
    project_resources_directory = Path(__file__).parents[2] / "resources"
    thesis_pdf_file_path = project_resources_directory / "solvedor_article.pdf"

    # Load translations if not provided
    if translations is None:
        import streamlit as st

        from config import language

        language_code = st.session_state.get("language", "en")
        translations = language.load_language_translations(language_code=language_code)

    if thesis_pdf_file_path.exists():
        display_thesis_header_with_download_button(
            thesis_file_path=thesis_pdf_file_path, translations=translations
        )
        st.markdown("")
        display_pdf_in_streamlit_viewer(
            pdf_file_path=thesis_pdf_file_path, translations=translations
        )
    else:
        st.warning(translations.errors.thesis_not_found)


def display_visualization_elements_guide() -> None:
    """Display comprehensive guide explaining visualization elements and their meaning."""
    st.markdown("### :material/help: Visual Elements Guide")
    st.markdown("Understanding the optimization visualization components:")

    st.markdown("""
    #### :material/search: Visual Elements Explained

    ---

    ##### :material/circle: **Feasible Region (Green Area)**
    - All points that satisfy **ALL** constraints simultaneously
    - The optimal solution must lie within or on the boundary
    - Represents the solution space where all requirements are met

    ---

    ##### :material/straighten: **Constraint Lines (Dashed)**
    - Each colored line represents one constraint boundary
    - Different colors help distinguish between constraints
    - Lines separate feasible from infeasible regions
    - Intersections often contain optimal solutions

    ---

    ##### :material/navigation: **Gradient Arrows (Navy Blue)**
    - Show the direction of steepest increase in objective value
    - Point toward higher objective function values
    - Arrow demonstrates optimization direction
    - Help visualize how the algorithm "climbs" toward optimum

    ---

    ##### :material/star: **Optimal Solution (Gold Star)**
    - The **best possible solution** within the feasible region
    - Located at vertex where gradient direction is blocked by constraints
    - Point where you cannot improve without violating constraints
    - Hover to see exact values and objective score

    ---

    #### :material/target: **How They Work Together**

    1. **Follow gradient arrows** from any point in feasible region
    2. **Arrows guide you** toward increasing objective values
    3. **Optimal star marks** where this process terminates
    4. **Mathematical guarantee**: This vertex gives the best possible solution

    ---

    #### :material/lightbulb: **Pro Tips**

    - **Vertices matter**: Optimal solutions for linear programs always occur
      at vertices
    - **Direction matters**: Gradient shows which way to improve the objective
    - **Constraints define limits**: Feasible region boundaries prevent
      further improvement
    - **Interactive exploration**: Hover over points to see exact coordinates and values
    """)


def display_markdown_content_from_file(file_name: str) -> None:
    """Display markdown file content from project root directory.

    Args:
        file_name: Name of the markdown file to render.
    """
    markdown_file_path = Path(__file__).parents[2] / file_name

    try:
        markdown_file_content = markdown_file_path.read_text(encoding="utf-8")
        st.markdown(markdown_file_content)
    except FileNotFoundError:
        st.warning(f"📄 {file_name} not found.")
    except OSError:
        st.warning(f"⚠️ Could not read {file_name}.")


def display_thesis_header_with_download_button(
    *, thesis_file_path: Path, translations: language.TranslationSchema
) -> None:
    """Display thesis header section with interactive download button.

    Args:
        thesis_file_path: Path to the thesis PDF file.
        translations: Translation schema for localized text.
    """
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown(
            (
                "**:material/school: Research Thesis - SolvedOR: "
                "Modern Operations Research Interface**"
            )
        )
        st.caption(
            ("This thesis introduces the OR-Solver application and its methodology.")
        )

    with col2:
        try:
            thesis_pdf_binary_data = thesis_file_path.read_bytes()
            download_button_clicked = st.download_button(
                f":material/download: {translations.resources.download_pdf}",
                thesis_pdf_binary_data,
                file_name="solvedor_article.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

            if download_button_clicked:
                st.toast(
                    ":material/file_download_done: Thesis PDF ready for download!",
                    icon=":material/download:",
                )
        except OSError:
            st.error(f":material/error: {translations.errors.pdf_load_error}")


def display_pdf_in_streamlit_viewer(
    *, pdf_file_path: Path, translations: language.TranslationSchema
) -> None:
    """Display PDF using Streamlit's built-in viewer with error handling.

    Args:
        pdf_file_path: Path to the PDF file to display.
        translations: Translation schema for localized text.
    """
    try:
        st.pdf(str(pdf_file_path))
    except Exception as pdf_display_error:
        st.warning(translations.errors.pdf_display_error)
        st.info(f"Error: {str(pdf_display_error)}")
