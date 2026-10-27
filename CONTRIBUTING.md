# Contributing

## Requirements

1. Ensure you have **Python 3.12+** installed:

   ```sh
   python --version  # Should be 3.12+
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
make install
```

This command will:

- Install all dependencies (including dev and test dependencies)
- Set up pre-commit hooks automatically

The environment is created at `.venv` directory.

## Development

[`Makefile`](./Makefile) has useful commands for development:

```sh
make help
```

`make check` runs everything CI runs: lint, type check and the tests with coverage.

Use [`uv run`](https://docs.astral.sh/uv/reference/cli/#uv-run) to run commands
in the project's environment:

```sh
uv run streamlit run streamlit_app.py
uv run pytest -v
```

## Tests

Run tests to ensure everything works:

```sh
make test
```

Commit with [conventional commit](https://www.conventionalcommits.org/) messages.

If you change dependencies, run `make requirements` so `requirements.txt` (used by
Streamlit Community Cloud) matches `uv.lock`; CI checks this.
