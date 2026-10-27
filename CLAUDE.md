# SolvedOR

Linear and integer programming solver for teaching, with textbook syntax

## Tech stack

- Python 3.12+, managed by **uv** (no pip, no manual venvs)
- src layout: code in `src/solvedor/`, tests in `tests/`
- Ruff (lint + format), mypy strict, pytest + pytest-cov

## Commands

- `make check`: lint + typecheck + tests. **Run before saying a task is done.**
- `make format`: auto-format + safe fixes
- `make test` / `uv run pytest tests/test_x.py::test_name -x`: run tests
- `uv add <pkg>` / `uv add --dev <pkg>`: add dependencies
- `uv run <cmd>`: run anything inside the project env

If `make` is unavailable, run the equivalent `uv run ...` line from the `Makefile`.

## Conventions

- Detailed rules: @.claude/rules/code-style.md, @.claude/rules/testing.md, @.claude/rules/api-conventions.md
- Write a failing test first for bug fixes.
- Prefer small, pure functions; side effects at the edges.
- Conventional Commits (`feat:`, `fix:`, ...). See CONTRIBUTING.md.

## Don'ts

- Don't `pip install` or edit `uv.lock` by hand.
- Don't add `# type: ignore` / `# noqa` without a reason comment.
- Don't commit secrets; config comes from environment variables.

## Architecture

Streamlit app for students: they type a linear or integer program in textbook syntax, it is
parsed, solved with Google OR-Tools, and shown with results and (for 2 variables) a plot.

- `streamlit_app.py`: entry point (Streamlit Community Cloud main file). Calls `solvedor.app.main()`.
- `src/solvedor/solver/`: parsing (`parser.py`, pyparsing grammar), the model (`models.py`,
  pydantic), solving (`backends.py`, GLOP for LP, SCIP for MIP). Must not import streamlit.
- `src/solvedor/ui/`: Streamlit pages and widgets. `src/solvedor/config/`: translations loader.
- `src/solvedor/translations/{en,pt}.yaml`: every user-facing string. Keep both files in step.
- `src/solvedor/resources/`: gallery examples (`examples.yaml`) and the thesis PDF.
- `requirements.txt` is generated from `uv.lock` (`make requirements`) for Streamlit Cloud; CI
  fails if it drifts. Never edit it by hand.
- Refactor in progress: `docs/superpowers/plans/2026-10-27-or-solver-refactor.md`. Temporary
  mypy exemptions and the coverage floor in `pyproject.toml` are removed as its tasks land.

Run the app: `make run`.
