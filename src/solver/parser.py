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
from pyparsing import Optional as OptionalPP

from solver.models import (
    Constraint,
    ConstraintOperator,
    LinearExpression,
    ObjectiveDirection,
    ObjectiveFunction,
    Problem,
    Variable,
    VariableType,
    add_term_to_linear_expression,
)


class ParseError(Exception):
    """Exception raised when parsing fails."""


# Configuration for LP Parser
_PARSER_GRAMMAR = None


def _get_or_initialize_parser_grammar():
    """Get or initialize the parser grammar."""
    global _PARSER_GRAMMAR
    if _PARSER_GRAMMAR is None:
        ParserElement.enablePackrat()
        _PARSER_GRAMMAR = _setup_complete_parser_grammar()
    return _PARSER_GRAMMAR


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
        self.grammar = _get_or_initialize_parser_grammar()

    def _setup_grammar(self):
        """Setup pyparsing grammar for LP problems (deprecated - use standalone function)."""
        return _setup_complete_parser_grammar()


def _setup_complete_parser_grammar():
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
    subject_to_pt = CaselessKeyword("sujeito") + CaselessKeyword("a") + Suppress(":")
    subject_to = subject_to_en | subject_to_pt

    # Single constraint
    constraint = Group(
        expression.setResultsName("lhs") + constraint_op + number.setResultsName("rhs")
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

    # Variable type declarations (support both "integer vars" and "vars integer")
    integer_decl = Group(
        (
            (CaselessKeyword("integer") + var_list.setResultsName("variables"))
            | (var_list.setResultsName("variables") + CaselessKeyword("integer"))
        )
    ).setResultsName("integer")

    binary_decl = Group(
        (
            (CaselessKeyword("binary") + var_list.setResultsName("variables"))
            | (var_list.setResultsName("variables") + CaselessKeyword("binary"))
        )
    ).setResultsName("binary")

    # Bounds can appear with or without "where:" keyword
    bounds_content = ZeroOrMore(non_negative | integer_decl | binary_decl)
    bounds_section = Group(
        OptionalPP(Suppress(where_kw)) + bounds_content
    ).setResultsName("bounds")

    # Complete LP problem
    lp_problem = (
        objective + OptionalPP(constraints_section) + OptionalPP(bounds_section)
    )

    return lp_problem

    def parse(self, problem_text: str) -> Problem:
        """Parse a complete LP problem from text."""
        return parse_lp_text_with_grammar(problem_text, self.grammar)

    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess the input text (deprecated - use standalone function)."""
        return clean_and_normalize_problem_text(text)

    def _build_problem(self, parsed_data) -> Problem:
        """Build a Problem object from parsed data (deprecated - use standalone function)."""
        return construct_problem_from_parsed_structure(parsed_data)

    def _build_objective(self, obj_data) -> ObjectiveFunction:
        """Build an ObjectiveFunction from parsed data (deprecated - use standalone function)."""
        return construct_objective_from_parsed_data(obj_data)

    def _build_constraint(self, constraint_data) -> Constraint:
        """Build a Constraint from parsed data (deprecated - use standalone function)."""
        return construct_constraint_from_parsed_data(constraint_data)

    def _build_expression(self, expr_data) -> LinearExpression:
        """Build a LinearExpression from parsed data (deprecated - use standalone function)."""
        return construct_linear_expression_from_parsed_data(expr_data)

    def _process_bounds(self, problem: Problem, bounds_data):
        """Process variable bounds and type declarations (deprecated - use standalone function)."""
        return apply_variable_bounds_and_types(problem, bounds_data)

    def _process_integer_variables(self, problem: Problem, var_list):
        """Process integer variable declarations (deprecated - use standalone function)."""
        return set_variables_as_integer_type(problem, var_list)

    def _process_binary_variables(self, problem: Problem, var_list):
        """Process binary variable declarations (deprecated - use standalone function)."""
        return set_variables_as_binary_type(problem, var_list)

    def _process_non_negative_variables(self, problem: Problem, var_list):
        """Process non-negative variable bounds (deprecated - use standalone function)."""
        return set_variables_non_negative_bounds(problem, var_list)


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
    complete_grammar = _get_or_initialize_parser_grammar()
    return parse_lp_text_with_grammar(problem_text, complete_grammar)


# Standalone parsing functions (functional programming approach)


def parse_lp_text_with_grammar(problem_text: str, parsing_grammar) -> Problem:
    """Parse LP text using the provided grammar."""
    try:
        preprocessed_text = clean_and_normalize_problem_text(problem_text)

        parsed_structure = parsing_grammar.parseString(preprocessed_text, parseAll=True)
        return construct_problem_from_parsed_structure(parsed_structure)

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


def clean_and_normalize_problem_text(text: str) -> str:
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


def construct_problem_from_parsed_structure(parsed_structure) -> Problem:
    """Build a Problem object from parsed data."""
    optimization_problem = Problem(
        objective=construct_objective_from_parsed_data(parsed_structure.objective),
        constraints=[],
        variables={},
    )

    if hasattr(parsed_structure, "constraints"):
        for constraint_data in parsed_structure.constraints:
            new_constraint = construct_constraint_from_parsed_data(constraint_data)
            optimization_problem.add_constraint(new_constraint)

    if hasattr(parsed_structure, "bounds"):
        apply_variable_bounds_and_types(optimization_problem, parsed_structure.bounds)

    return optimization_problem


def construct_objective_from_parsed_data(objective_data) -> ObjectiveFunction:
    """Build an ObjectiveFunction from parsed data."""
    optimization_direction = objective_data[0]  # "maximize" or "minimize"
    direction_enum = ObjectiveDirection(optimization_direction)

    objective_expression = construct_linear_expression_from_parsed_data(
        objective_data.expression
    )

    return ObjectiveFunction(direction=direction_enum, expression=objective_expression)


def construct_constraint_from_parsed_data(constraint_data) -> Constraint:
    """Build a Constraint from parsed data."""
    left_hand_side_expression = construct_linear_expression_from_parsed_data(
        constraint_data.lhs
    )
    comparison_operator = ConstraintOperator(constraint_data.operator)
    right_hand_side_value = constraint_data.rhs

    return Constraint(
        expression=left_hand_side_expression,
        operator=comparison_operator,
        rhs=right_hand_side_value,
    )


def construct_linear_expression_from_parsed_data(expression_data) -> LinearExpression:
    """Build a LinearExpression from parsed data."""
    linear_expression = LinearExpression()

    for term_data in expression_data:
        sign_multiplier = 1 if term_data.sign == "+" else -1
        final_coefficient = term_data.coeff * sign_multiplier
        variable_name = term_data.variable

        add_term_to_linear_expression(
            linear_expression, final_coefficient, variable_name
        )

    return linear_expression


def apply_variable_bounds_and_types(
    optimization_problem: Problem, bounds_specifications
):
    """Process variable bounds and type declarations."""
    for bound_specification in bounds_specifications:
        if len(bound_specification) < 2:
            continue

        if len(bound_specification) == 2:
            first_element, second_element = (
                bound_specification[0],
                bound_specification[1],
            )

            if str(first_element).lower() == "integer":
                set_variables_as_integer_type(optimization_problem, second_element)
            elif str(second_element).lower() == "integer":
                set_variables_as_integer_type(optimization_problem, first_element)
            elif str(first_element).lower() == "binary":
                set_variables_as_binary_type(optimization_problem, second_element)
            elif str(second_element).lower() == "binary":
                set_variables_as_binary_type(optimization_problem, first_element)
            elif (
                hasattr(first_element, "__iter__")
                and not isinstance(first_element, str)
                and str(second_element) == "0"
            ):
                set_variables_non_negative_bounds(optimization_problem, first_element)
            elif (
                hasattr(second_element, "__iter__")
                and not isinstance(second_element, str)
                and str(first_element) == "0"
            ):
                set_variables_non_negative_bounds(optimization_problem, second_element)


def set_variables_as_integer_type(optimization_problem: Problem, variable_list):
    """Process integer variable declarations."""
    if isinstance(variable_list, str):
        variable_list = [variable_list]
    elif not isinstance(variable_list, list):
        variable_list = list(variable_list) if variable_list else []

    for variable_name in variable_list:
        clean_variable_name = str(variable_name).strip()
        if (
            clean_variable_name
            and clean_variable_name not in optimization_problem.variables
        ):
            optimization_problem.variables[clean_variable_name] = Variable(
                name=clean_variable_name
            )
        if clean_variable_name in optimization_problem.variables:
            optimization_problem.variables[
                clean_variable_name
            ].variable_type = VariableType.INTEGER


def set_variables_as_binary_type(optimization_problem: Problem, variable_list):
    """Process binary variable declarations."""
    if isinstance(variable_list, str):
        variable_list = [variable_list]
    elif not isinstance(variable_list, list):
        variable_list = list(variable_list) if variable_list else []

    for variable_name in variable_list:
        clean_variable_name = str(variable_name).strip()
        if (
            clean_variable_name
            and clean_variable_name not in optimization_problem.variables
        ):
            optimization_problem.variables[clean_variable_name] = Variable(
                name=clean_variable_name
            )
        if clean_variable_name in optimization_problem.variables:
            optimization_problem.variables[
                clean_variable_name
            ].variable_type = VariableType.BINARY
            optimization_problem.variables[clean_variable_name].lower_bound = Decimal(
                "0"
            )
            optimization_problem.variables[clean_variable_name].upper_bound = Decimal(
                "1"
            )


def set_variables_non_negative_bounds(optimization_problem: Problem, variable_list):
    """Process non-negative variable bounds."""
    if isinstance(variable_list, str):
        variable_list = [variable_list]
    elif hasattr(variable_list, "__iter__") and not isinstance(variable_list, str):
        variable_list = list(variable_list)
        if (
            len(variable_list) == 1
            and hasattr(variable_list[0], "__iter__")
            and not isinstance(variable_list[0], str)
        ):
            variable_list = list(variable_list[0])
    else:
        variable_list = [variable_list] if variable_list else []

    for variable_name in variable_list:
        clean_variable_name = str(variable_name).strip()
        if (
            clean_variable_name
            and clean_variable_name not in optimization_problem.variables
        ):
            optimization_problem.variables[clean_variable_name] = Variable(
                name=clean_variable_name
            )
        if clean_variable_name in optimization_problem.variables:
            optimization_problem.variables[clean_variable_name].lower_bound = Decimal(
                "0"
            )
