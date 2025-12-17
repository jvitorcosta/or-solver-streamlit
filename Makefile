.PHONY: help install dev test lint format clean run

# 🐱 OR-Solver Commands
help:
	@echo "🐾 Available commands:"
	@echo "  install     Install project"
	@echo "  dev         Install with dev dependencies"
	@echo "  test        Run all tests"
	@echo "  lint        Check code style"
	@echo "  format      Format code"
	@echo "  pre-commit  Run pre-commit hooks"
	@echo "  run         Start web interface"
	@echo "  clean       Clean build artifacts"

# Setup
install:
	uv sync

dev:
	uv sync --extra dev --extra test
	uv run pre-commit install

# Testing & Quality
test:
	uv run pytest -v

lint:
	uv run ruff check src tests
	uv run mypy src

format:
	uv run ruff format src tests
	uv run ruff check --fix src tests

pre-commit:
	uv run pre-commit run --all-files

# Running
run:
	uv run streamlit run app.py

# Maintenance
clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .coverage htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete
