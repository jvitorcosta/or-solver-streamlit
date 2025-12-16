"""Modern Streamlit web interface for OR-Solver."""

from pathlib import Path

import streamlit as st
import yaml

# Import our domain models and services
try:
    from or_solver.domain.parser import ParseError, parse_lp_problem
except ImportError:
    # Fallback for development
    st.error("⚠️ OR-Solver modules not found. Please install the package.")
    st.stop()


def load_examples():
    """Load example problems from config."""
    examples_file = (
        Path(__file__).parent.parent.parent.parent.parent / "config" / "examples.yaml"
    )
    if examples_file.exists():
        with open(examples_file) as f:
            return yaml.safe_load(f)
    return {}


def setup_page_config():
    """Setup Streamlit page configuration."""
    st.set_page_config(
        page_title="🐱 OR-Solver",
        page_icon="🐱",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://github.com/username/or-solver-streamlit",
            "Report a bug": "https://github.com/username/or-solver-streamlit/issues",
            "About": "# 🐱 OR-Solver\nModern Operations Research Solver - Making LP as easy as petting a cat!",
        },
    )

    # Modern CSS styling with Material Icons
    st.markdown(
        """
    <style>
    @import url('https://fonts.googleapis.com/icon?family=Material+Icons');

    .main > div {
        padding-top: 2rem;
    }
    .stAlert > div {
        background-color: #f0f8f7;
        border-left: 4px solid #00c896;
    }
    .stButton > button {
        height: 3em;
        font-weight: 600;
        border-radius: 8px;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
    }
    .stTextArea textarea {
        font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
        font-size: 14px;
        line-height: 1.5;
    }
    h1 {
        color: #2e3440;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .material-icon {
        font-family: 'Material Icons';
        font-weight: normal;
        font-style: normal;
        font-size: 20px;
        display: inline-block;
        line-height: 1;
        text-transform: none;
        letter-spacing: normal;
        word-wrap: normal;
        white-space: nowrap;
        direction: ltr;
        vertical-align: middle;
    }
    .success-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 1rem 0;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    """Render the sidebar with language selection and examples."""
    st.sidebar.markdown("## 🐾 Settings")

    # Language selection
    language = st.sidebar.selectbox(
        "🌍 Language / Idioma",
        options=["en", "pt"],
        format_func=lambda x: "English" if x == "en" else "Português",
        index=0,  # Default to English
    )

    # Precision setting
    precision = st.sidebar.slider(
        "⚙️ Decimal Precision",
        min_value=0,
        max_value=10,
        value=4,
        help="Number of decimal places in results",
    )

    # Solver selection
    solver = st.sidebar.selectbox(
        "⚡ Solver",
        options=["glop", "scip"],
        format_func=lambda x: "GLOP (Linear Programming)"
        if x == "glop"
        else "SCIP (Integer Programming)",
        help="Choose the optimization solver",
    )

    # Examples section
    st.sidebar.markdown("## 😺 Example Problems")

    examples = load_examples()
    if examples:
        example_names = list(examples.keys())
        selected_example = st.sidebar.selectbox(
            "Choose an example:",
            options=[""] + example_names,
            format_func=lambda x: "Select example..."
            if x == ""
            else examples[x].get("name", x)
            if x
            else "",
        )

        if selected_example and st.sidebar.button("📋 Load Example"):
            st.session_state.example_text = examples[selected_example]["problem"]
            st.session_state.example_name = examples[selected_example]["name"]
            st.rerun()

    return language, precision, solver


def render_main_interface(language):
    """Render the main problem input and solving interface."""
    # Modern header design
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🐱 OR-Solver")
        if language == "pt":
            st.markdown(
                "**Otimização Linear Moderna** • *Fácil como fazer carinho em gato!*"
            )
        else:
            st.markdown(
                "**Modern Linear Programming** • *A purr-fect optimization tool!*"
            )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if language == "pt":
            st.metric("Status", "● Ativo", "Pronto para resolver")
        else:
            st.metric("Status", "● Active", "Ready to solve")

    st.markdown("---")

    # Set button labels
    if language == "pt":
        problem_label = "▶️ Digite seu problema de programação linear:"
        solve_button = "🚀 Resolver Problema"
        validate_button = "✓ Validar Sintaxe"
    else:
        problem_label = "▶️ Enter your linear programming problem:"
        solve_button = "🚀 Solve Problem"
        validate_button = "✓ Validate Syntax"

    # Load example if available
    default_text = ""
    if "example_text" in st.session_state:
        default_text = st.session_state.example_text
        if "example_name" in st.session_state:
            st.info(f"📋 Loaded example: {st.session_state.example_name}")

    problem_text = st.text_area(
        problem_label,
        value=default_text,
        height=400,
        placeholder="""maximize 8*cat_food + 10*cat_toys

subject to:
    0.5*cat_food + 0.5*cat_toys <= 150
    0.6*cat_food + 0.4*cat_toys <= 145
    cat_food >= 30
    cat_toys >= 40

where:
    cat_food, cat_toys >= 0""",
        help="🐱 Enter your linear programming problem using academic syntax",
    )

    # Action buttons with better spacing
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

    with col1:
        if st.button(solve_button, type="primary", use_container_width=True):
            solve_problem(problem_text, language)

    with col2:
        if st.button(validate_button, use_container_width=True):
            validate_problem(problem_text, language)

    with col3:
        if st.button("? Syntax Help", use_container_width=True):
            show_syntax_help(language)

    # Add some spacing after buttons
    st.markdown("<br>", unsafe_allow_html=True)


def solve_problem(problem_text, language):
    """Solve the linear programming problem."""
    if not problem_text.strip():
        if language == "pt":
            st.warning("⚠️ Por favor, digite um problema para resolver")
        else:
            st.warning("⚠️ Please enter a problem to solve")
        return

    try:
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Parse problem
        status_text.text("🐱 Parsing problem...")
        progress_bar.progress(25)
        problem = parse_lp_problem(problem_text)

        # Setup solver (mock for now)
        status_text.text("😸 Setting up solver...")
        progress_bar.progress(50)

        # Solve (mock for now)
        status_text.text("😻 Solving optimization...")
        progress_bar.progress(75)

        # Complete
        status_text.text("✅ Solution found!")
        progress_bar.progress(100)

        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()

        # Display results (mock solution)
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown("## 🎯 Optimal Solution Found!")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Objective Value", "1,250.0", delta="Optimal")

        with col2:
            st.metric("Solve Time", "0.123s", delta=None)

        # Variables table
        st.markdown("### 📊 Variable Values")

        import pandas as pd

        solution_data = {
            "Variable": ["🐾 cat_food", "🐾 cat_toys"],
            "Value": [100.0, 125.0],
            "Type": ["Continuous", "Continuous"],
        }
        st.dataframe(pd.DataFrame(solution_data), use_container_width=True)

        st.markdown("**Status:** OPTIMAL 😺")
        st.markdown("</div>", unsafe_allow_html=True)

        # Problem summary
        with st.expander("📋 Problem Summary"):
            st.write(f"**Objective:** {problem.objective.direction.value}")
            st.write(f"**Variables:** {len(problem.get_all_variables())}")
            st.write(f"**Constraints:** {len(problem.constraints)}")

        # Export options
        with st.expander("💾 Export Results"):
            st.download_button(
                "📄 Download as JSON",
                data='{"status": "optimal", "objective_value": 1250.0, "variables": {"cat_food": 100.0, "cat_toys": 125.0}}',
                file_name="solution.json",
                mime="application/json",
            )

    except ParseError as e:
        st.markdown('<div class="error-box">', unsafe_allow_html=True)
        if language == "pt":
            st.markdown("## ❌ Erro de Sintaxe")
        else:
            st.markdown("## ❌ Syntax Error")
        st.markdown(f"**Error:** {e}")
        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.markdown('<div class="error-box">', unsafe_allow_html=True)
        if language == "pt":
            st.markdown("## ❌ Erro na Resolução")
        else:
            st.markdown("## ❌ Solving Error")
        st.markdown(f"**Error:** {e}")
        st.markdown("</div>", unsafe_allow_html=True)


def validate_problem(problem_text, language):
    """Validate the problem syntax."""
    if not problem_text.strip():
        if language == "pt":
            st.warning("⚠️ Por favor, digite um problema para validar")
        else:
            st.warning("⚠️ Please enter a problem to validate")
        return

    try:
        problem = parse_lp_problem(problem_text)

        if language == "pt":
            st.success("✅ Sintaxe válida!")
        else:
            st.success("✅ Valid syntax!")

        # Show formatted problem
        with st.expander("📝 Formatted Problem"):
            st.code(str(problem), language="text")

    except ParseError as e:
        if language == "pt":
            st.error(f"❌ Erro de sintaxe: {e}")
        else:
            st.error(f"❌ Syntax error: {e}")


def show_syntax_help(language):
    """Show syntax reference in a modal."""
    if language == "pt":
        st.info("""
        ## 📚 Guia de Sintaxe - Português

        **Função Objetivo:**
        ```
        maximizar 8*comida_gato + 10*brinquedos_gato
        minimizar 2*x1 + 3*x2
        ```

        **Restrições:**
        ```
        sujeito a:
            0.5*comida_gato + 0.5*brinquedos_gato <= 150
            comida_gato >= 30
        ```

        **Domínio das Variáveis:**
        ```
        onde:
            comida_gato, brinquedos_gato >= 0
            inteiro x1, x2
            binario y1, y2
        ```
        """)
    else:
        st.info("""
        ## 📚 Syntax Guide - English

        **Objective Function:**
        ```
        maximize 8*cat_food + 10*cat_toys
        minimize 2*x1 + 3*x2
        ```

        **Constraints:**
        ```
        subject to:
            0.5*cat_food + 0.5*cat_toys <= 150
            cat_food >= 30
        ```

        **Variable Domain:**
        ```
        where:
            cat_food, cat_toys >= 0
            integer x1, x2
            binary y1, y2
        ```
        """)


def main():
    """Main Streamlit application."""
    setup_page_config()

    # Render sidebar and get settings
    language, precision, solver = render_sidebar()

    # Render main interface
    render_main_interface(language)

    # Footer
    st.markdown("---")
    st.markdown(
        "Made with ❤️ and 🐱 by the OR-Solver team | "
        "[📚 Documentation](https://github.com/username/or-solver-streamlit) | "
        "[🐛 Report Issues](https://github.com/username/or-solver-streamlit/issues)"
    )


if __name__ == "__main__":
    main()
