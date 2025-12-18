from pathlib import Path

import streamlit as st


def render_thesis_tab() -> None:
    """Render the research thesis PDF tab with viewer and download functionality."""
    resources_directory_path = Path(__file__).parents[2] / "resources"
    thesis_file_path = resources_directory_path / "solvedor_article.pdf"

    if thesis_file_path.exists():
        _render_thesis_header_with_download(thesis_file_path=thesis_file_path)
        st.markdown("")
        _render_pdf_viewer(pdf_file_path=thesis_file_path)
    else:
        st.warning("Thesis PDF not found.")


def render_guide_tab() -> None:
    """Render the visual elements guide tab."""
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


def render_markdown_file(file_name: str) -> None:
    """Render a markdown file from the project root directory.

    Args:
        file_name: Name of the markdown file to render.
    """
    file_path = Path(__file__).parents[2] / file_name

    try:
        content = file_path.read_text(encoding="utf-8")
        st.markdown(content)
    except FileNotFoundError:
        st.warning(f"📄 {file_name} not found.")
    except OSError:
        st.warning(f"⚠️ Could not read {file_name}.")


def _render_thesis_header_with_download(*, thesis_file_path: Path) -> None:
    """Render thesis header section with download button.

    Args:
        thesis_file_path: Path to the thesis PDF file.
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
            pdf_data = thesis_file_path.read_bytes()
            download_pressed = st.download_button(
                ":material/download: Download PDF",
                pdf_data,
                file_name="solvedor_article.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

            if download_pressed:
                st.toast(
                    ":material/file_download_done: Thesis PDF ready for download!",
                    icon=":material/download:",
                )
        except OSError:
            st.error("📚 Could not load PDF for download.")


def _render_pdf_viewer(*, pdf_file_path: Path) -> None:
    """Render PDF viewer with error handling.

    Args:
        pdf_file_path: Path to the PDF file to display.
    """
    try:
        st.pdf(str(pdf_file_path))
    except Exception as pdf_error:
        st.warning(
            (
                "PDF viewer not available. You can download the thesis "
                "using the button above."
            )
        )
        st.info(f"Error: {str(pdf_error)}")
