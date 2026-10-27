.DEFAULT_GOAL := help
.PHONY: help run requirements install sync lock upgrade format lint typecheck test cov check pre-commit build clean

# Recipes use only `uv ...` and Python one-liners, so they behave the same in
# bash/zsh (macOS, Linux) and cmd/PowerShell (Windows): no awk, rm or find.
PY := uv run --no-project --quiet python -c

help: ## Show this help
	@$(PY) "import re; [print(f'  {t:<12} {d}') for t, d in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open('Makefile').read(), re.M)]"

run: ## Run the app locally
	uv run streamlit run streamlit_app.py

requirements: ## Regenerate requirements.txt (Streamlit Community Cloud) from uv.lock
	uv export --no-hashes --no-dev -q -o requirements.txt

install: ## Install deps and git hooks (first-time setup)
	uv sync --all-groups
	uv run pre-commit install

sync: ## Sync the environment with uv.lock
	uv sync --all-groups

lock: ## Re-resolve uv.lock
	uv lock

upgrade: ## Upgrade all dependencies to their latest allowed versions
	uv lock --upgrade
	uv sync --all-groups

format: ## Auto-format and apply safe lint fixes
	uv run ruff format .
	uv run ruff check --fix .

lint: ## Lint and check formatting (no changes)
	uv run ruff check .
	uv run ruff format --check .

typecheck: ## Static type checking
	uv run mypy

test: ## Run tests
	uv run pytest

cov: ## Run tests with coverage
	uv run pytest --cov --cov-report=term-missing

check: lint typecheck cov ## Everything CI runs

pre-commit: ## Run all pre-commit hooks on all files
	uv run pre-commit run --all-files

build: ## Build sdist + wheel into dist/
	uv build

clean: ## Remove caches and build artifacts
	@$(PY) "import shutil, pathlib as p; [shutil.rmtree(d, ignore_errors=True) for d in ['dist', 'build', '.pytest_cache', '.mypy_cache', '.ruff_cache', 'htmlcov', *(x for x in p.Path('.').rglob('__pycache__') if '.venv' not in x.parts)]]; p.Path('.coverage').unlink(missing_ok=True)"
