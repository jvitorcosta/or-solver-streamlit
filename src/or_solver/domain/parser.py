"""LP syntax parser."""

from __future__ import annotations

import re
from decimal import Decimal

from pyparsing import (
    CaselessKeyword,
    Forward,
    Group,
    Literal,
    OneOrMore,
    ParseException,
    ParserElement,
    Regex,
    Suppress,
    Word,
    ZeroOrMore,
    alphanums,
    alphas,
)
from pyparsing import (
    Optional as OptionalPP,
)

from .models import (
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
    """Exception raised when parsing fails."""



class LPParser:
    """Parser for modern academic LP syntax.

    Supports both English and Portuguese syntax:

    English:
        maximize/minimize <expression>
        subject to: / s.t.:
            <constraints>
        where:
            <variable bounds and types>

    Portuguese:
        maximizar/minimizar <expression>
        sujeito a:
            <constraints>
        onde:
            <variable bounds and types>
    """

    def __init__(self):
        """Initialize the parser with grammar rules."""
        ParserElement.enablePackrat()
        self._setup_grammar()

    def _setup_grammar(self):
        """Setup pyparsing grammar for LP problems."""
        # Basic tokens
        number = Regex(r"[+-]?\d+(\.\d*)?([eE][+-]?\d+)?").setParseAction(
            lambda t: Decimal(t[0])
        )

        variable = Word(alphas, alphanums + "_").setResultsName("variable")

        # Operators
        multiply = Literal("*")
        plus = Literal("+")
        minus = Literal("-")

        # Constraint operators
        le = Literal("<=")
        ge = Literal(">=")
        eq = Literal("=")
        constraint_op = (le | ge | eq).setResultsName("operator")

        # Coefficient (optional, defaults to 1)
        coefficient = OptionalPP(number, default=Decimal("1"))

        # Term: [coefficient] [*] variable
        term = Group(
            OptionalPP(minus, default="+").setResultsName("sign")
            + coefficient.setResultsName("coeff")
            + OptionalPP(multiply)
            + variable
        )

        # Expression: term + term + ... (handling signs)
        expression = Forward()
        first_term = Group(
            OptionalPP(plus | minus, default="+").setResultsName("sign")
            + coefficient.setResultsName("coeff")
            + OptionalPP(multiply)
            + variable
        )
        additional_term = Group(
            (plus | minus).setResultsName("sign")
            + coefficient.setResultsName("coeff")
            + OptionalPP(multiply)
            + variable
        )
        expression <<= first_term + ZeroOrMore(additional_term)

        # Objective direction keywords (English and Portuguese)
        maximize_en = CaselessKeyword("maximize") | CaselessKeyword("max")
        minimize_en = CaselessKeyword("minimize") | CaselessKeyword("min")
        maximize_pt = CaselessKeyword("maximizar")
        minimize_pt = CaselessKeyword("minimizar")

        maximize_kw = (maximize_en | maximize_pt).setParseAction(lambda: "maximize")
        minimize_kw = (minimize_en | minimize_pt).setParseAction(lambda: "minimize")

        objective_direction = (maximize_kw | minimize_kw).setResultsName("direction")

        # Objective function
        objective = Group(
            objective_direction + expression.setResultsName("expression")
        ).setResultsName("objective")

        # Constraint keywords
        subject_to_en = (
            CaselessKeyword("subject") + CaselessKeyword("to") + Suppress(":")
            | CaselessKeyword("s.t.") + Suppress(":")
            | CaselessKeyword("st") + Suppress(":")
        )
        subject_to_pt = (
            CaselessKeyword("sujeito") + CaselessKeyword("a") + Suppress(":")
        )
        subject_to = subject_to_en | subject_to_pt

        # Single constraint
        constraint = Group(
            expression.setResultsName("lhs")
            + constraint_op
            + number.setResultsName("rhs")
        )

        # Constraints section
        constraints_section = Group(subject_to + OneOrMore(constraint)).setResultsName(
            "constraints"
        )

        # Variable bounds and types keywords
        where_en = CaselessKeyword("where") + Suppress(":")
        where_pt = CaselessKeyword("onde") + Suppress(":")
        where_kw = where_en | where_pt

        # Variable lists
        var_list = Group(OneOrMore(variable + OptionalPP(Suppress(","))))

        # Bound specifications
        non_negative = Group(
            var_list.setResultsName("variables") + Suppress(">=") + Literal("0")
        ).setResultsName("non_negative")

        # Variable type declarations
        integer_decl = Group(
            CaselessKeyword("integer") + var_list.setResultsName("variables")
        ).setResultsName("integer")

        binary_decl = Group(
            CaselessKeyword("binary") + var_list.setResultsName("variables")
        ).setResultsName("binary")

        # Bounds section
        bounds_section = Group(
            where_kw + ZeroOrMore(non_negative | integer_decl | binary_decl)
        ).setResultsName("bounds")

        # Complete LP problem
        self.lp_problem = (
            objective + OptionalPP(constraints_section) + OptionalPP(bounds_section)
        )

    def parse(self, problem_text: str) -> Problem:
        """Parse a complete LP problem from text.

        Args:
            problem_text: The LP problem text to parse.

        Returns:
            A Problem object representing the parsed problem.

        Raises:
            ParseError: If parsing fails.
        """
        try:
            # Clean up the text
            text = self._preprocess_text(problem_text)

            # Parse with pyparsing
            parsed = self.lp_problem.parseString(text, parseAll=True)

            # Convert to Problem object
            return self._build_problem(parsed)

        except ParseException as e:
            raise ParseError(f"Failed to parse LP problem: {e}")

    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess the input text."""
        # Remove comments
        lines = []
        for line in text.split("\n"):
            # Remove comments (anything after #)
            line = re.sub(r"#.*$", "", line)
            # Remove extra whitespace
            line = line.strip()
            if line:  # Skip empty lines
                lines.append(line)

        return "\n".join(lines)

    def _build_problem(self, parsed_data) -> Problem:
        """Build a Problem object from parsed data."""
        problem = Problem(
            objective=self._build_objective(parsed_data.objective),
            constraints=[],
            variables={},
        )

        # Add constraints
        if hasattr(parsed_data, "constraints"):
            for constraint_data in parsed_data.constraints[
                1:
            ]:  # Skip the "subject to" part
                constraint = self._build_constraint(constraint_data)
                problem.add_constraint(constraint)

        # Add variable bounds and types
        if hasattr(parsed_data, "bounds"):
            self._process_bounds(problem, parsed_data.bounds)

        return problem

    def _build_objective(self, obj_data) -> ObjectiveFunction:
        """Build an ObjectiveFunction from parsed data."""
        direction_str = obj_data[0]  # "maximize" or "minimize"
        direction = ObjectiveDirection(direction_str)

        expression = self._build_expression(obj_data.expression)

        return ObjectiveFunction(direction=direction, expression=expression)

    def _build_constraint(self, constraint_data) -> Constraint:
        """Build a Constraint from parsed data."""
        lhs_expression = self._build_expression(constraint_data.lhs)
        operator = ConstraintOperator(constraint_data.operator)
        rhs = constraint_data.rhs

        return Constraint(expression=lhs_expression, operator=operator, rhs=rhs)

    def _build_expression(self, expr_data) -> LinearExpression:
        """Build a LinearExpression from parsed data."""
        expression = LinearExpression()

        for term_data in expr_data:
            # Handle sign
            sign = 1 if term_data.sign == "+" else -1

            # Get coefficient
            coeff = term_data.coeff * sign

            # Get variable name
            var_name = term_data.variable

            # Add term to expression
            expression.add_term(coeff, var_name)

        return expression

    def _process_bounds(self, problem: Problem, bounds_data):
        """Process variable bounds and type declarations."""
        for bound_spec in bounds_data[1:]:  # Skip the "where" part
            if hasattr(bound_spec, "non_negative"):
                # Non-negative variables
                for var_name in bound_spec.non_negative.variables:
                    if var_name not in problem.variables:
                        problem.variables[var_name] = Variable(name=var_name)
                    problem.variables[var_name].lower_bound = Decimal("0")

            elif hasattr(bound_spec, "integer"):
                # Integer variables
                for var_name in bound_spec.integer.variables:
                    if var_name not in problem.variables:
                        problem.variables[var_name] = Variable(name=var_name)
                    problem.variables[var_name].var_type = VariableType.INTEGER

            elif hasattr(bound_spec, "binary"):
                # Binary variables
                for var_name in bound_spec.binary.variables:
                    if var_name not in problem.variables:
                        problem.variables[var_name] = Variable(name=var_name)
                    problem.variables[var_name].var_type = VariableType.BINARY
                    problem.variables[var_name].lower_bound = Decimal("0")
                    problem.variables[var_name].upper_bound = Decimal("1")


def parse_lp_problem(problem_text: str) -> Problem:
    """Parse a linear programming problem from text.

    This is a convenience function that creates a parser instance
    and parses the problem.

    Args:
        problem_text: The LP problem text to parse.

    Returns:
        A Problem object representing the parsed problem.

    Raises:
        ParseError: If parsing fails.
    """
    parser = LPParser()
    return parser.parse(problem_text)


# Legacy syntax converter for backward compatibility
def convert_old_syntax(old_problem_text: str) -> str:
    """Convert old Portuguese colon-based syntax to new academic syntax.

    Args:
        old_problem_text: Problem text in old format.

    Returns:
        Problem text in new academic format.
    """
    lines = old_problem_text.strip().split("\n")
    converted_lines = []

    objective_found = False
    constraints_section = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Convert problem type declaration (ignore for now)
        if line.startswith("problema:") or line.startswith("p:"):
            continue

        # Convert objective
        if line.startswith("max:") or line.startswith("min:"):
            if line.startswith("max:"):
                objective_expr = line[4:].strip()
                converted_lines.append(f"maximize {objective_expr}")
            else:
                objective_expr = line[4:].strip()
                converted_lines.append(f"minimize {objective_expr}")
            objective_found = True
            continue

        # Convert constraints
        if line.startswith("restricao:") or line.startswith("r:"):
            if not constraints_section and objective_found:
                converted_lines.append("")
                converted_lines.append("subject to:")
                constraints_section = True

            if line.startswith("restricao:"):
                constraint_expr = line[10:].strip()
            else:
                constraint_expr = line[2:].strip()

            converted_lines.append(f"    {constraint_expr}")
            continue

        # Convert rounding specification (ignore for now)
        if line.startswith("arredondamento:") or line.startswith("arred:"):
            continue

        # Keep other lines as-is
        converted_lines.append(line)

    # Add default non-negative constraints if none specified
    if not any("where:" in line or "onde:" in line for line in converted_lines):
        # Extract variables from the problem
        variables = set()
        for line in converted_lines:
            # Simple regex to find variable patterns like x1, x2, etc.
            vars_in_line = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", line)
            for var in vars_in_line:
                if var.lower() not in [
                    "maximize",
                    "minimize",
                    "subject",
                    "to",
                    "where",
                    "integer",
                    "binary",
                ]:
                    variables.add(var)

        if variables:
            converted_lines.append("")
            converted_lines.append("where:")
            sorted_vars = sorted(variables)
            converted_lines.append(f"    {', '.join(sorted_vars)} >= 0")

    return "\n".join(converted_lines)
