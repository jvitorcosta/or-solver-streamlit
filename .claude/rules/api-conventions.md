---
paths:
  - "src/**/*.py"
---

# API conventions

Applies to the package's public Python API, and to any HTTP/CLI surface added later.

## Public Python API

- Public names are re-exported from the package `__init__.py`; everything else is
  private (`_leading_underscore` or not re-exported).
- Keyword-only arguments (`*,`) for anything beyond the first one or two params.
- Return values, don't mutate arguments. Return typed objects, not tuples of 3+.
- Errors: define a base `SolvedorError(Exception)` and subclass it;
  callers should be able to catch everything from this package with one class.
- Breaking changes to public names require a `feat!:` / `BREAKING CHANGE:` commit.

## HTTP APIs (if/when added)

- JSON with `snake_case` fields; ISO-8601 UTC timestamps.
- Resource-oriented paths (`/users/{id}`), plural nouns, no verbs.
- Validate input and serialise output with pydantic models at the edge.
- Errors: `{"error": {"code": "not_found", "message": "..."}}` with the right status code.
- Version in the path (`/v1/...`) once there are external consumers.

## CLI (if present)

- Subcommands are verbs (`greet`, `export`). `--help` text on every command and option.
- Exit code 0 on success, non-zero on failure; errors go to stderr.
