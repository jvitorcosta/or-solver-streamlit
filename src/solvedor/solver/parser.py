"""Parse linear and integer programs written in textbook syntax.

The input is read line by line, so every error points at the line that caused it:

    maximize 3x + 2y            (or max / minimize / min / maximizar / minimizar)
    subject to:                 (or s.t. / st / sujeito a; the colon is optional)
        x + 2y <= 4             (<=, >=, =, ≤, ≥; constants and variables on either side)
    where:                      (or onde; optional)
        x, y >= 0               (lists share a bound)
        0 <= z <= 10            (both bounds)
        free w                  (no lower bound)
        integer a, b            (or "a, b integer"; same for binary)

Variables are non-negative unless declared otherwise. Comments start with ``#``.
Each line is parsed with a small pyparsing grammar.
"""

from __future__ import annotations

import re
from decimal import Decimal
from functools import cache

import pyparsing as pp

from solvedor.solver.models import (
    Constraint,
    ConstraintOperator,
    LinearExpression,
    ObjectiveDirection,
    ObjectiveFunction,
    Problem,
    Variable,
    VariableType,
)


class ParseError(Exception):
    """A problem text that cannot be parsed, with the 1-based line it failed on."""

    def __init__(self, message: str, line: int = 0, column: int = 0) -> None:
        location = f"line {line}: " if line else ""
        super().__init__(f"{location}{message}")
        self.message = message
        self.line = line
        self.column = column


_DIRECTIONS = {
    "max": ObjectiveDirection.MAXIMIZE,
    "maximize": ObjectiveDirection.MAXIMIZE,
    "maximizar": ObjectiveDirection.MAXIMIZE,
    "min": ObjectiveDirection.MINIMIZE,
    "minimize": ObjectiveDirection.MINIMIZE,
    "minimizar": ObjectiveDirection.MINIMIZE,
}
_OPERATORS = {
    "<=": ConstraintOperator.LESS_EQUAL,
    "≤": ConstraintOperator.LESS_EQUAL,
    ">=": ConstraintOperator.GREATER_EQUAL,
    "≥": ConstraintOperator.GREATER_EQUAL,
    "=": ConstraintOperator.EQUAL,
}
_KEYWORDS = {"integer", "binary", "free"}
_CONSTRAINTS_HEADER = re.compile(r"^(subject\s+to|s\.t\.|st|sujeito\s+a)\s*:?\s*(.*)$", re.I)
_WHERE_HEADER = re.compile(r"^(where|onde)\s*:?\s*(.*)$", re.I)
_OBJECTIVE_START = re.compile(r"^(maximize|maximizar|max|minimize|minimizar|min)\b\s*(.*)$", re.I)

Term = tuple[Decimal, str | None]  # (coefficient, variable name or None for a constant)


@cache
def _grammar() -> dict[str, pp.ParserElement]:
    """Build the per-line grammars once."""
    number = pp.Regex(r"(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?").set_parse_action(
        lambda t: Decimal(t[0])
    )
    name = pp.Regex(r"[A-Za-z_][A-Za-z0-9_]*")
    sign = pp.one_of("+ -")
    term = pp.Group(
        pp.Optional(sign, default="+")("sign")
        + (
            (number("coef") + pp.Optional(pp.Suppress("*")) + pp.Optional(name("var")))
            | name("var")
        )
    )
    # Terms after the first must carry a sign.
    signed_term = pp.Group(
        sign("sign")
        + (
            (number("coef") + pp.Optional(pp.Suppress("*")) + pp.Optional(name("var")))
            | name("var")
        )
    )
    expression = pp.Group(term + pp.ZeroOrMore(signed_term))
    operator = pp.one_of("<= >= ≤ ≥ =")
    constraint = expression("lhs") + operator("op") + expression("rhs")
    names = pp.Group(pp.DelimitedList(name))
    signed_number = pp.Combine(pp.Optional(sign) + pp.Regex(r"(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?"))
    signed_number.set_parse_action(lambda t: Decimal(t[0]))
    keyword = pp.one_of("integer binary free", caseless=True)
    bound = (
        (
            signed_number("low")
            + pp.one_of("<= ≤")
            + name("var")
            + pp.one_of("<= ≤")
            + signed_number("high")
        )
        | (names("vars") + operator("op") + signed_number("value"))
        | (keyword("kind") + names("vars"))
        | (names("vars") + keyword("kind"))
    )
    return {"expression": expression, "constraint": constraint, "bound": bound}


def _parse_line(grammar: str, text: str, line: int) -> pp.ParseResults:
    try:
        return _grammar()[grammar].parse_string(text, parse_all=True)
    except pp.ParseException as e:
        raise ParseError(f"cannot read '{text}' ({e.msg})", line, e.col) from e


def _terms(expression: pp.ParseResults) -> list[Term]:
    terms: list[Term] = []
    for item in expression:
        coef = item.get("coef", Decimal(1))
        if item.get("sign") == "-":
            coef = -coef
        var = item.get("var") or None
        terms.append((coef, var))
    return terms


def _linear(terms: list[Term]) -> tuple[LinearExpression, Decimal]:
    """Split terms into a linear expression and a constant."""
    expression = LinearExpression()
    constant = Decimal(0)
    for coef, var in terms:
        if var is None:
            constant += coef
        else:
            expression.add_term(coef, var)
    return expression, constant


def _split_comment(raw: str) -> str:
    return raw.split("#", 1)[0].strip()


def parse_lp_problem(problem_text: str) -> Problem:
    """Parse a problem text into a Problem. Raises ParseError with the failing line."""
    lines = [(n, _split_comment(raw)) for n, raw in enumerate(problem_text.splitlines(), start=1)]
    lines = [(n, text) for n, text in lines if text]
    if not lines:
        raise ParseError("the problem is empty: start with 'maximize' or 'minimize'")

    first_line, first_text = lines[0]
    match = _OBJECTIVE_START.match(first_text)
    if not match:
        raise ParseError("the problem must start with 'maximize' or 'minimize'", first_line)
    direction = _DIRECTIONS[match.group(1).lower()]
    objective_parts = [(first_line, match.group(2))]

    section = "objective"
    constraint_lines: list[tuple[int, str]] = []
    bound_lines: list[tuple[int, str]] = []
    for line, text in lines[1:]:
        if header := _CONSTRAINTS_HEADER.match(text):
            section = "constraints"
            if header.group(2):
                constraint_lines.append((line, header.group(2)))
        elif header := _WHERE_HEADER.match(text):
            section = "where"
            if header.group(2):
                bound_lines.append((line, header.group(2)))
        elif section == "objective":
            objective_parts.append((line, text))
        elif section == "constraints":
            constraint_lines.append((line, text))
        else:
            bound_lines.append((line, text))

    objective_text = " ".join(part for _, part in objective_parts if part)
    if not objective_text:
        raise ParseError("the objective has no expression", first_line)
    objective_terms = _terms(_parse_line("expression", objective_text, first_line)[0])
    objective_expression, _ = _linear(objective_terms)
    problem = Problem(
        objective=ObjectiveFunction(direction=direction, expression=objective_expression)
    )

    for line, text in constraint_lines:
        problem.add_constraint(_parse_constraint(text, line))

    # Every variable starts non-negative and continuous; the where: section refines it.
    problem.variables = {
        name: Variable(name=name, lower_bound=Decimal(0))
        for name in problem.extract_all_variable_names()
    }

    for line, text in bound_lines:
        _apply_bound(problem, text, line)

    return problem


def _parse_constraint(text: str, line: int) -> Constraint:
    parsed = _parse_line("constraint", text, line)
    lhs_expr, lhs_const = _linear(_terms(parsed["lhs"]))
    rhs_expr, rhs_const = _linear(_terms(parsed["rhs"]))
    # Move variables to the left and constants to the right: lhs - rhs_vars (op) rhs - lhs_const.
    for term in rhs_expr.terms:
        lhs_expr.add_term(-term.coefficient, term.variable)
    lhs_expr.terms = [t for t in lhs_expr.terms if t.coefficient != 0]
    if not lhs_expr.terms:
        raise ParseError(f"'{text}' has no variables", line)
    return Constraint(
        expression=lhs_expr, operator=_OPERATORS[parsed["op"]], rhs=rhs_const - lhs_const
    )


def _apply_bound(problem: Problem, text: str, line: int) -> None:
    parsed = _parse_line("bound", text, line)
    names = [parsed["var"]] if "var" in parsed else list(parsed["vars"])
    for name in names:
        if name.lower() in _KEYWORDS:
            raise ParseError(f"'{name}' is a keyword, not a variable", line)
        if name not in problem.variables:
            raise ParseError(
                f"unknown variable '{name}' (it is not used in the objective or constraints)", line
            )
        variable = problem.variables[name]
        if "kind" in parsed:
            kind = parsed["kind"].lower()
            if kind == "free":
                variable.lower_bound = None
            elif kind == "integer":
                variable.variable_type = VariableType.INTEGER
            else:
                variable.variable_type = VariableType.BINARY
                variable.lower_bound, variable.upper_bound = Decimal(0), Decimal(1)
        elif "low" in parsed:
            variable.lower_bound, variable.upper_bound = parsed["low"], parsed["high"]
        else:
            operator = _OPERATORS[parsed["op"]]
            value = parsed["value"]
            if operator is ConstraintOperator.GREATER_EQUAL:
                variable.lower_bound = value
            elif operator is ConstraintOperator.LESS_EQUAL:
                variable.upper_bound = value
            else:
                variable.lower_bound = variable.upper_bound = value
