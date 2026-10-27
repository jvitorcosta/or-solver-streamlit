---
paths:
  - "src/**/*.py"
  - "tests/**/*.py"
---

# Code style

- Ruff is the source of truth for formatting and lint (`make format`). Don't hand-format.
- Line length 100. Double quotes. f-strings over `%` / `.format()`.
- Full type hints on every function signature; mypy runs in `strict` mode.
  - Use built-in generics (`list[str]`, `dict[str, int]`) and `X | None`, not `Optional[X]`.
  - Prefer `collections.abc` (`Iterable`, `Sequence`, `Mapping`) for parameters.
- Use `pathlib.Path`, never `os.path`.
- Prefer `dataclasses` (or pydantic models at I/O boundaries) over loose dicts.
- Imports: absolute (`from solvedor.x import y`), no wildcard imports.
- Names: `snake_case` functions/vars, `PascalCase` classes, `UPPER_CASE` constants.
- Docstrings: one-line summary for every public function/class; Google style if more is needed.
- Raise specific exceptions; never bare `except:`. Don't swallow errors silently.
- Logging via `logging.getLogger(__name__)`, not `print`, in library code.
- No `# type: ignore` / `# noqa` without an explanation on the same line.
