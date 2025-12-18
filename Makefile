.PHONY: help
help: ## Show this help
	@awk 'BEGIN {FS = ":.*##"; printf "usage: make \033[36m\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Setup
.PHONY: install
install: ## Install project dependencies
	uv sync

.PHONY: dev
dev: ## Complete development setup with pre-commit hooks
	uv sync --frozen --extra dev --extra test
	uv run pre-commit install

.PHONY: setup
setup: dev ## Alias for dev (backward compatibility)

##@ Development
.PHONY: run
run: ## Start the web interface
	uv run streamlit run src/app.py

.PHONY: format
format: ## Format code with ruff
	uv run ruff format src tests
	uv run ruff check --fix src tests

##@ Quality & Testing
.PHONY: lint
lint: ## Run pre-commit hooks
	uv run pre-commit run --all-files

.PHONY: test
test: ## Run all tests
	uv run pytest -v

##@ Maintenance
.PHONY: clean
clean: ## Remove build and cache files
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pytest_cache|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	find . | grep -E "dist" | xargs rm -rf
	find . | grep -E ".egg-info" | xargs rm -rf
	find . | grep -E "htmlcov" | xargs rm -rf
	rm -rf .coverage*
	rm -rf build/
