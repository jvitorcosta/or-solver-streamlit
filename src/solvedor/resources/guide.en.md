### :material/help: How to write a problem

A problem has up to three parts. Only the objective is required.

```text
maximize 3x + 2y          # or max, minimize, min

subject to:               # or s.t. / st; the colon is optional
    x + 2y <= 4           # <=, >=, = (≤ and ≥ work too)
    2x + y - z <= 4

where:                    # optional: bounds and types
    x <= 3                # an upper bound
    0 <= y <= 10          # both bounds
    free z                # z can be negative
    integer x             # or binary
```

- **Variables are non-negative** unless you write `free` or give them a lower bound.
- Coefficients can be written `3x`, `3*x` or `3 x`, with decimals (`0.5x`) or exponents (`1e3x`).
  `2e1` is the number 20, so for a variable named `e1` write `2*e1`.
- A constant in the objective (`max 3x + 5`) is added to the objective value.
- Constants and variables may sit on either side of a constraint: `x + 5 >= y` is fine.
- Variable names start with a letter or `_` and may contain digits (`product_a`, `x1`).
- `#` starts a comment.
- If something is wrong, the error message tells you the line.

### :material/function: Results

- **Optimal**: the best solution, proven.
- **Best found before the time limit**: an integer problem stopped after 10 seconds with a
  good solution that is not proven optimal.
- **Infeasible**: no point satisfies every constraint.
- **Unbounded**: the objective can improve forever; some limit is missing.

Linear problems are solved with GLOP and integer problems with SCIP, both from Google OR-Tools.

### :material/show_chart: The plot (two-variable problems)

- **Green area**: the feasible region, computed from your constraints and bounds.
- **Dashed lines**: one per constraint, labelled with the constraint itself.
- **Arrow**: the direction in which the objective improves.
- **Star**: the solution. For a linear problem it sits on a corner of the region.

For integer problems the plot shows the continuous region; the solution is a whole-number
point inside it.
