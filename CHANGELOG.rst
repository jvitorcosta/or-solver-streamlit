=========
Changelog
=========

All notable changes to SolvedOR will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[Unreleased]
============

[2.1.0] - 2026-10-27
====================

Fixed
-----
* Unbounded linear programs were reported as infeasible (GLOP's presolve); they are now
  re-solved without presolve to tell the two apart.
* Variables that only appear in the objective were dropped.
* Variables were free by default; they are now non-negative unless declared ``free``.
* The 2D plot was hardcoded to one example; it now draws the real feasible region.
* An objective value of 0 was shown as "N/A".
* Results disappeared when the language changed or the page reran.
* Clearing the workspace showed "Using example: None".
* Integer problems could run forever; there is now a 10 second time limit.

Added
-----
* Syntax: ``≤`` / ``≥``, signed and scientific numbers, constants on either side of a
  constraint, ``s.t.`` / ``st`` / ``sujeito a``, ``free`` variables, ``low <= x <= high`` bounds.
* Parse errors name the line they happened on.
* Statuses for "best solution found before the time limit" and "no solution within the time
  limit".
* Guide tab in English and Portuguese; every UI string is translated, with tests.
* Headless app tests (Streamlit AppTest), type checking (mypy strict), CI on three operating
  systems.

Changed
-------
* Code moved to a ``src/solvedor`` package based on the python-uv-claude-template.
* ``solver.api.solve_text`` is a pure API with no Streamlit import.
* Requires Python 3.12.

Removed
-------
* The command line interface and unused dependencies.
* The unused ``settings.yaml``.

[1.0.0] - 2024-12-16
====================

Added
-----
* Operations Research solver with a textbook-syntax parser
* Clean layered architecture (domain/presentation)
* Academic-standard LP syntax parser with pyparsing
* Multilingual support (English/Portuguese)
* Rich CLI interface with typer
* Modern Streamlit web interface
* Pre-commit hooks with comprehensive linting
* Material design integration
* White theme as default
* Comprehensive testing framework

Changed
-------
* Replaced legacy colon-based syntax with academic notation
* Simplified project structure (removed unused layers)
* Updated to modern Python tooling (uv, ruff, pre-commit)
* Streamlined Makefile with ``make run`` instead of ``make web``
* Replaced emojis with material icons

Removed
-------
* File upload feature (simplified interface)
* Legacy syntax converter
* PyPI release workflow
* Windows testing (focused on macOS/Ubuntu)
* Redundant documentation and comments

[0.0.0] - Previous Version
==========================
* Legacy Portuguese colon-based syntax
* Basic Streamlit interface
* Old project structure
