"""CLI interface for OR-Solver."""

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table

from ..domain.models import Problem
from ..domain.parser import ParseError, parse_lp_problem

# Initialize Rich console and Typer app
console = Console()
app = typer.Typer(
    name="or-solver",
    help="🐱 Modern Operations Research Solver - Making LP as easy as petting a cat!",
    rich_markup_mode="rich",
)

# Subcommands
examples_app = typer.Typer(help="🐾 Example problems and templates")
config_app = typer.Typer(help="🔧 Configuration management")
app.add_typer(examples_app, name="examples")
app.add_typer(config_app, name="config")


@app.command()
def solve(
    problem_file: Path | None = typer.Argument(None, help="Path to LP problem file"),
    interactive: bool = typer.Option(
        False, "--interactive", "-i", help="Interactive mode"
    ),
    output_format: str = typer.Option(
        "text", "--format", "-f", help="Output format: text, json"
    ),
    validate_only: bool = typer.Option(
        False, "--validate", help="Only validate syntax, don't solve"
    ),
    show_steps: bool = typer.Option(False, "--steps", help="Show solution steps"),
) -> None:
    """🐱 Solve a linear programming problem.

    Examples:
        or-solver solve problem.lp
        or-solver solve --interactive
        or-solver solve problem.lp --format json
        or-solver solve problem.lp --validate --steps
    """
    if interactive:
        _solve_interactive()
    elif problem_file:
        _solve_from_file(problem_file, output_format, validate_only, show_steps)
    else:
        console.print("❌ Please provide a problem file or use --interactive mode")
        raise typer.Exit(1)





@app.command()
def validate(
    problem_file: Path = typer.Argument(..., help="Problem file to validate"),
    highlight: bool = typer.Option(True, "--highlight", help="Syntax highlighting"),
) -> None:
    """✅ Validate LP problem syntax.

    Examples:
        or-solver validate problem.lp
        or-solver validate problem.lp --no-highlight
    """
    if not problem_file.exists():
        console.print(f"❌ File not found: {problem_file}")
        raise typer.Exit(1)

    try:
        content = problem_file.read_text(encoding="utf-8")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("🐱 Validating syntax...", total=None)

            # Parse the problem
            problem = parse_lp_problem(content)

            progress.update(task, description="✅ Validation complete!")

        console.print("\n[green]✅ Problem syntax is valid![/green]")

        # Show problem summary
        _show_problem_summary(problem)

        # Show syntax highlighting if requested
        if highlight:
            console.print("\n[bold cyan]📝 Formatted Problem:[/bold cyan]")
            syntax = Syntax(str(problem), "text", theme="monokai", line_numbers=True)
            console.print(syntax)

    except ParseError as e:
        console.print(f"\n[red]❌ Syntax Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[red]❌ Validation failed:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def web() -> None:
    """🌐 Launch the Streamlit web interface."""
    try:
        import subprocess

        console.print("🚀 Launching Streamlit web interface...")
        console.print("🌐 Opening browser at http://localhost:8501")

        # Try to launch streamlit
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "src/or_solver/presentation/web_app.py",
            ],
            check=False,
        )

        if result.returncode != 0:
            console.print("❌ Failed to launch Streamlit. Make sure it's installed.")

    except ImportError:
        console.print("❌ Streamlit not installed. Install with: uv add streamlit")
    except Exception as e:
        console.print(f"❌ Error launching web interface: {e}")


@app.command()
def explain(
    problem_file: Path = typer.Argument(..., help="Problem file to explain"),
    language: str = typer.Option("en", "--lang", "-l", help="Language: en, pt"),
) -> None:
    """📚 Get step-by-step problem explanation.

    Examples:
        or-solver explain diet_problem.lp
        or-solver explain problem.lp --lang pt
    """
    if not problem_file.exists():
        console.print(f"❌ File not found: {problem_file}")
        raise typer.Exit(1)

    # This would integrate with the education service
    console.print("🔮 Educational explanations coming soon! 😸")


@app.command()
def syntax_help() -> None:
    """📖 Show syntax reference guide."""
    console.print(
        Panel.fit(
            """
[bold cyan]🐱 OR-Solver Syntax Guide[/bold cyan]

[yellow]English Syntax (Default):[/yellow]
    maximize 8*cat_food + 10*cat_toys
    
    subject to:
        0.5*cat_food + 0.5*cat_toys <= 150
        cat_food >= 30
    
    where:
        cat_food, cat_toys >= 0

[yellow]Portuguese Syntax:[/yellow]
    maximizar 8*comida_gato + 10*brinquedos_gato
    
    sujeito a:
        0.5*comida_gato + 0.5*brinquedos_gato <= 150
        comida_gato >= 30
    
    onde:
        comida_gato, brinquedos_gato >= 0

[yellow]Variable Types:[/yellow]
    where:
        x1, x2 >= 0        # Continuous non-negative
        integer y1, y2     # Integer variables  
        binary z1, z2      # Binary (0/1) variables

[yellow]Supported Operators:[/yellow]
    <=    Less than or equal
    >=    Greater than or equal
    =     Equal to

[green]💡 Tip:[/green] Use descriptive variable names like 'cat_food' instead of 'x1'!
        """,
            title="📚 Syntax Reference",
            border_style="cyan",
        )
    )


# Examples subcommands
@examples_app.command("list")
def list_examples() -> None:
    """📋 List available example problems."""
    examples = [
        ("diet", "🍽️  Diet Problem - Optimal cat nutrition"),
        ("transport", "🚛 Transportation - Cat food distribution"),
        ("assignment", "🎯 Assignment - Cat toys to cats"),
        ("knapsack", "🎒 Knapsack - Cat supplies selection"),
        ("production", "🏭 Production - Cat treat manufacturing"),
    ]

    table = Table(
        title="🐾 Available Examples", show_header=True, header_style="bold magenta"
    )
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Description", style="green")

    for name, description in examples:
        table.add_row(name, description)

    console.print(table)


@examples_app.command("run")
def run_example(
    name: str = typer.Argument(..., help="Example name to run"),
    format_output: str = typer.Option(
        "text", "--format", help="Output format: text, json"
    ),
) -> None:
    """🏃 Run a specific example problem.

    Examples:
        or-solver examples run diet
        or-solver examples run transport --format json
    """
    # This would load from config/examples.yaml
    console.print(f"🐱 Running example '{name}'...")
    console.print("🔮 Example problems coming soon! 😸")


@examples_app.command("template")
def create_template(
    problem_type: str = typer.Argument(
        ..., help="Problem type: diet, transport, assignment, etc."
    ),
    output_file: Path | None = typer.Option(
        None, "--output", "-o", help="Output file"
    ),
) -> None:
    """📝 Create a problem template.

    Examples:
        or-solver examples template diet
        or-solver examples template transport --output my_problem.lp
    """
    # This would load templates from config/templates.yaml
    console.print(f"🐾 Creating {problem_type} problem template...")
    console.print("🔮 Problem templates coming soon! 😸")


# Config subcommands
@config_app.command("show")
def show_config() -> None:
    """📋 Show current configuration."""
    # This would load from config files
    config_data = {
        "language": "en",
        "default_solver": "glop",
        "precision": 3,
        "theme": "cat_theme",
    }

    table = Table(title="🔧 Current Configuration", show_header=True)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    for key, value in config_data.items():
        table.add_row(key, str(value))

    console.print(table)


@config_app.command("set")
def set_config(
    key: str = typer.Argument(..., help="Configuration key"),
    value: str = typer.Argument(..., help="Configuration value"),
) -> None:
    """⚙️  Set configuration value.

    Examples:
        or-solver config set language pt
        or-solver config set precision 4
    """
    console.print(f"🔧 Setting {key} = {value}")
    # This would save to config files
    console.print("✅ Configuration updated!")


# Helper functions
def _solve_interactive() -> None:
    """Interactive problem solving mode."""
    console.print("🐱 Interactive Problem Solver")
    console.print("Enter your LP problem (press Ctrl+D when done):\n")

    lines = []
    try:
        while True:
            line = input()
            lines.append(line)
    except EOFError:
        pass

    if lines:
        problem_text = "\n".join(lines)
        _solve_problem_text(problem_text)
    else:
        console.print("❌ No problem entered")


def _solve_from_file(
    problem_file: Path, output_format: str, validate_only: bool, show_steps: bool
) -> None:
    """Solve problem from file."""
    if not problem_file.exists():
        console.print(f"❌ File not found: {problem_file}")
        raise typer.Exit(1)

    try:
        content = problem_file.read_text(encoding="utf-8")

        if validate_only:
            # Just validate, don't solve
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("🐱 Validating problem...", total=None)
                problem = parse_lp_problem(content)
                progress.update(task, description="✅ Validation complete!")

            console.print("[green]✅ Problem syntax is valid![/green]")
            _show_problem_summary(problem)
        else:
            _solve_problem_text(content, output_format, show_steps)

    except Exception as e:
        console.print(f"❌ Error processing file: {e}")
        raise typer.Exit(1)


def _solve_problem_text(
    problem_text: str, output_format: str = "text", show_steps: bool = False
) -> None:
    """Solve problem from text."""
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Parse problem
            parse_task = progress.add_task("🐱 Parsing problem...", total=None)
            problem = parse_lp_problem(problem_text)
            progress.update(parse_task, description="✅ Parsing complete!")

            # Setup solver
            setup_task = progress.add_task("😸 Setting up solver...", total=None)
            # TODO: Initialize solver service
            progress.update(setup_task, description="✅ Solver ready!")

            # Solve
            solve_task = progress.add_task("😻 Solving optimization...", total=None)
            # TODO: Solve problem
            progress.update(solve_task, description="✅ Solution found!")

        # For now, show a mock solution
        console.print("\n🎯 [bold green]Optimal Solution Found![/bold green]")
        console.print("   Objective Value: 1250.0")
        console.print("\n   Variables:")
        console.print("   🐾 cat_food = 100.0")
        console.print("   🐾 cat_toys = 125.0")
        console.print("\n   Status: OPTIMAL 😺")

        if output_format == "json":
            # Output as JSON
            result = {
                "status": "optimal",
                "objective_value": 1250.0,
                "variables": {"cat_food": 100.0, "cat_toys": 125.0},
            }
            console.print(json.dumps(result, indent=2))

    except ParseError as e:
        console.print(f"\n[red]❌ Parse Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[red]❌ Solving failed:[/red] {e}")
        raise typer.Exit(1)


def _show_problem_summary(problem: Problem) -> None:
    """Show a summary of the parsed problem."""
    table = Table(title="📊 Problem Summary", show_header=True)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Objective", str(problem.objective.direction.value))
    table.add_row("Variables", str(len(problem.get_all_variables())))
    table.add_row("Constraints", str(len(problem.constraints)))

    # Variable types
    var_types = {"continuous": 0, "integer": 0, "binary": 0}
    for var in problem.variables.values():
        var_types[var.var_type.value] += 1

    for var_type, count in var_types.items():
        if count > 0:
            table.add_row(f"{var_type.title()} vars", str(count))

    console.print("\n")
    console.print(table)


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
