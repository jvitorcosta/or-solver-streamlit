# OR-Solver

[![CI](https://github.com/jvitorcosta/or-solver-streamlit/actions/workflows/ci.yml/badge.svg)](https://github.com/jvitorcosta/or-solver-streamlit/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://solvedor.streamlit.app)

Write a linear or integer program the way a textbook does, and solve it in the browser.

**Live demo: [solvedor.streamlit.app](https://solvedor.streamlit.app)**

OR-Solver is a Streamlit app. It parses the model you type, solves it with
[Google OR-Tools](https://developers.google.com/optimization) (GLOP for linear programs,
SCIP for integer ones), and shows the solution. For two-variable problems it also draws the
feasible region, the constraint lines, the direction the objective improves and the solution.
The interface is in English and Portuguese.

## Quick start

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/jvitorcosta/or-solver-streamlit.git
cd or-solver-streamlit
make install
make run
```

The app opens at `http://localhost:8501`. Pick an example from the gallery to start.

## Writing a problem

```text
maximize 40*product_a + 50*product_b

subject to:
    2*product_a + 3*product_b <= 100    # labor hours
    4*product_a + 2*product_b <= 120    # raw materials

where:
    product_a <= 30
    product_b <= 25
```

- **Objective**: `maximize` / `max` / `minimize` / `min` (Portuguese `maximizar` / `minimizar`
  also work). It may span several lines.
- **Constraints** go after `subject to`, `s.t.`, `st` or `sujeito a` (the colon is optional).
  Operators are `<=`, `>=`, `=`, `≤` and `≥`. Constants and variables can sit on either side.
- **Numbers** can be signed, decimal (`0.5`) or scientific (`1e3`). Write coefficients as
  `3x`, `3*x` or `3 x`. `2e1` is the number 20, so for a variable named `e1` write `2*e1`.
- **A constant in the objective** (`max 3x + 5`) is added to the objective value.
- **Bounds and types** go after `where` or `onde` (optional):
  - `x, y >= 0`, `x <= 30`, `0 <= z <= 10`, `w = 5`
  - `free x` lets a variable be negative
  - `integer a, b` or `binary c` (also `a, b integer`); a binary variable takes no other bounds
- **Every variable is non-negative** unless it is declared `free` or given a lower bound.
- `#` starts a comment. A syntax error message names the line it happened on.

### Results

| Status | Meaning |
| --- | --- |
| Optimal | The best solution, proven. |
| Best found before the time limit | An integer problem ran out of time (10 s) with a solution that is not proven optimal. |
| Infeasible | No point satisfies every constraint. |
| Unbounded | The objective can improve forever. |
| No solution within the time limit | The solver stopped before finding any solution. |

The Guide tab inside the app has the same reference.

## Development

```sh
make install   # dependencies and git hooks
make check     # lint, type check and tests with coverage, as CI runs them
make run       # the app
```

`make help` lists every target. See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

The code lives in `src/solvedor/`: `solver/` turns text into an answer without importing
Streamlit (`solver.api.solve_text`), and `ui/` draws it.

## License

MIT, see [LICENSE](LICENSE).

**Maintainer**: Vitor Pinheiro ([GitHub](https://github.com/jvitorcosta) ·
[LinkedIn](https://linkedin.com/in/jvpro))
