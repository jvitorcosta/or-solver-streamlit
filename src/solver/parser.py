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

from solver.models import (
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
        constraints_section = Group(
            Suppress(subject_to) + OneOrMore(constraint)
        ).setResultsName("constraints")

        # Variable bounds and types keywords (optional)
        where_en = CaselessKeyword("where") + Suppress(":")
        where_pt = CaselessKeyword("onde") + Suppress(":")
        where_kw = where_en | where_pt

        # Variable lists
        var_list = Group(variable + ZeroOrMore(Suppress(",") + variable))

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

        # Bounds can appear with or without "where:" keyword
        bounds_content = ZeroOrMore(non_negative | integer_decl | binary_decl)
        bounds_section = Group(
            OptionalPP(Suppress(where_kw)) + bounds_content
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

            # Parse with pyparsing (parseAll=True ensures complete parsing)
            parsed = self.lp_problem.parseString(text, parseAll=True)

            # Convert to Problem object
            return self._build_problem(parsed)

        except ParseException as e:
            # Provide more detailed error information
            error_line = e.lineno if hasattr(e, "lineno") else "unknown"
            error_col = e.col if hasattr(e, "col") else "unknown"
            error_context = e.line if hasattr(e, "line") else ""

            error_msg = f"Parsing failed at line {error_line}, column {error_col}"
            if error_context:
                error_msg += f" in: '{error_context}'"
            error_msg += f" - {str(e)}"

            raise ParseError(error_msg) from e

    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess the input text."""
        # Remove comments and clean up
        lines = []
        for line in text.split("\n"):
            # Remove comments (anything after #)
            line = re.sub(r"#.*$", "", line)
            # Remove extra whitespace and normalize
            line = line.strip()
            # Skip empty lines or lines with only whitespace
            if line and not line.isspace():
                # Normalize multiple spaces to single spaces
                line = re.sub(r"\s+", " ", line)
                # Ensure proper spacing around operators
                line = re.sub(r"([<>=]+)", r" \1 ", line)
                line = re.sub(r"([+-])", r" \1 ", line)
                line = re.sub(r"\s+", " ", line).strip()
                lines.append(line)

        result = "\n".join(lines)
        # Remove any trailing whitespace or newlines
        return result.strip()

    def _build_problem(self, parsed_data) -> Problem:
        """Build a Problem object from parsed data."""
        problem = Problem(
            objective=self._build_objective(parsed_data.objective),
            constraints=[],
            variables={},
        )

        # Add constraints
        if hasattr(parsed_data, "constraints"):
            for constraint_data in parsed_data.constraints:
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
        for bound_spec in bounds_data:
            if len(bound_spec) != 2:
                continue

            first_element, second_element = bound_spec[0], bound_spec[1]

            if str(first_element) == "integer":
                self._process_integer_variables(problem, second_element)
            elif str(first_element) == "binary":
                self._process_binary_variables(problem, second_element)
            elif isinstance(first_element, list) and str(second_element) == "0":
                self._process_non_negative_variables(problem, first_element)

    def _process_integer_variables(self, problem: Problem, var_list):
        """Process integer variable declarations."""
        for var_name in var_list:
            if var_name not in problem.variables:
                problem.variables[var_name] = Variable(name=var_name)
            problem.variables[var_name].var_type = VariableType.INTEGER

    def _process_binary_variables(self, problem: Problem, var_list):
        """Process binary variable declarations."""
        for var_name in var_list:
            if var_name not in problem.variables:
                problem.variables[var_name] = Variable(name=var_name)
            problem.variables[var_name].var_type = VariableType.BINARY
            problem.variables[var_name].lower_bound = Decimal("0")
            problem.variables[var_name].upper_bound = Decimal("1")

    def _process_non_negative_variables(self, problem: Problem, var_list):
        """Process non-negative variable bounds."""
        for var_name in var_list:
            if var_name not in problem.variables:
                problem.variables[var_name] = Variable(name=var_name)
            problem.variables[var_name].lower_bound = Decimal("0")


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
