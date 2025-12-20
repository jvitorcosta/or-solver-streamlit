from pathlib import Path

import streamlit as st

import language


def display_research_thesis_with_viewer(
    translations: language.TranslationSchema,
) -> None:
    """Display research thesis PDF with integrated viewer and download functionality."""
    project_resources_directory = Path(__file__).parents[2] / "resources"
    thesis_pdf_file_path = project_resources_directory / "paper.pdf"

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


def display_visualization_elements_guide(
    translations: language.TranslationSchema,
) -> None:
    """Display comprehensive guide explaining visualization elements and their meaning."""
    st.markdown(f"### :material/help: {translations.visualization.guide_title}")
    st.markdown(translations.visualization.title)

    st.markdown(f"""
    #### :material/search: {translations.visualization.guide_title}

    ---

    ##### :material/circle: **{translations.visualization.feasible_region}**
    - All points that satisfy **ALL** constraints simultaneously
    - The optimal solution must lie within or on the boundary
    - Represents the solution space where all requirements are met

    ---

    ##### :material/straighten: **{translations.visualization.constraint_lines}**
    - Each colored line represents one constraint boundary
    - Different colors help distinguish between constraints
    - Lines separate feasible from infeasible regions
    - Intersections often contain optimal solutions

    ---

    ##### :material/navigation: **Gradient Arrows (Navy Blue)**
    - Show the direction of steepest increase in objective value
    - Point toward higher objective function values
    - {translations.visualization.gradient_arrow}
    - Help visualize how the algorithm "climbs" toward optimum

    ---

    ##### :material/star: **{translations.visualization.optimal_point}**
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


def display_markdown_content_from_file(
    file_name: str, translations: language.TranslationSchema
) -> None:
    """Display markdown file content from project root directory.

    Args:
        file_name: Name of the markdown file to render.
        translations: Translation schema for localized text.
    """
    markdown_file_path = Path(__file__).parents[2] / file_name

    try:
        markdown_file_content = markdown_file_path.read_text(encoding="utf-8")
        st.markdown(markdown_file_content)
    except FileNotFoundError:
        st.warning(f"📄 {file_name} {translations.errors.file_not_found}")
    except OSError:
        st.warning(f"⚠️ {translations.errors.file_read_error} {file_name}.")


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
        st.markdown(f"**:material/school: {translations.resources.thesis}**")
        st.caption(translations.resources.structure)

    with col2:
        try:
            thesis_pdf_binary_data = thesis_file_path.read_bytes()
            download_button_clicked = st.download_button(
                f":material/download: {translations.resources.download_pdf}",
                thesis_pdf_binary_data,
                file_name="paper.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

            if download_button_clicked:
                st.toast(
                    f":material/file_download_done: {translations.resources.download_pdf}!",
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
        st.pdf(str(pdf_file_path), height=800)
    except Exception as pdf_display_error:
        st.warning(translations.errors.pdf_display_error)
        st.info(f"Error: {str(pdf_display_error)}")
