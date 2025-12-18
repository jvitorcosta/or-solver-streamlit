# Contributing

## Requirements

1. Ensure you have **Python 3.11+** installed:

   ```sh
   python --version  # Should be 3.11+
   ```

2. Ensure that you have `uv` installed.
   We recommend installing it using the official installer:

   ```sh
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   If you already have `uv` installed, please update it to the latest version:

   ```sh
   uv self update
   ```

3. Fork the repository on GitHub and clone your fork:

   ```sh
   git clone https://github.com/YOUR_USERNAME/or-solver-streamlit.git
   cd or-solver-streamlit
   ```

## Setup

Run the following command in the root directory of the repository:

> [!NOTE]
> Pre-commit hooks will be automatically installed during setup
> to ensure code quality on every commit.

```sh
make dev
```

This command will:

- Install all dependencies (including dev and test dependencies)
- Set up pre-commit hooks automatically

The environment is created at `.venv` directory.

## Development

[`Makefile`](./Makefile) has useful commands for development:

```console
$ make help
Available commands:
  dev              Setup development environment
  run              Run Streamlit web interface
  test             Run test suite
  lint             Check code quality
  format           Format code with ruff
  clean            Remove build and cache files
```

Use [`uv run`](https://docs.astral.sh/uv/reference/cli/#uv-run) to run commands
in the project's environment:

```sh
uv run streamlit run src/app.py
uv run pytest tests/ -v
uv run or-solver --help
```

## Tests

Run tests to ensure everything works:

```sh
make test
```

1. Commit with [semantic commit](https://www.conventionalcommits.org/)
