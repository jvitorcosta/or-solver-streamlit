=========
Changelog
=========

All notable changes to OR-Solver will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[Unreleased]
============

[2.0.0] - 2024-12-16
====================

Added
-----
* 🐱 Modern cat-themed Operations Research solver
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
* Replaced emojis with material icons (kept cats!)

Removed
-------
* File upload feature (simplified interface)
* Legacy syntax converter
* PyPI release workflow
* Windows testing (focused on macOS/Ubuntu)
* Redundant documentation and comments

[1.0.0] - Previous Version
==========================
* Legacy Portuguese colon-based syntax
* Basic Streamlit interface
* Old project structure
