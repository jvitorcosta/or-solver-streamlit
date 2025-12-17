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

For example:

```sh
make run    # Start web interface at http://localhost:8501
make test   # Run all tests
make lint   # Check code quality
```

Use [`uv run`](https://docs.astral.sh/uv/reference/cli/#uv-run) to run commands in the project's environment:

```sh
uv run streamlit run app.py
uv run pytest tests/ -v
uv run or-solver --help
```

## Tests
Run tests to ensure everything works:

```sh
make test
```

4. Commit with [semantic commit](https://www.conventionalcommits.org/)

## Testing

Manual testing checklist:
- [ ] Test English and Portuguese interfaces
- [ ] Try all problem templates in the gallery
- [ ] Verify responsive design on different screen sizes
- [ ] Test invalid input handling

## Getting Help

Check out [good first issues](https://github.com/jvitorcosta/or-solver-streamlit/labels/good%20first%20issue) to get started!

For questions, open an issue on GitHub.
