---
paths:
  - "tests/**"
---

# Testing

- Framework: **pytest**. Plain `assert`; no `unittest.TestCase`.
- Layout mirrors `src/`: `src/pkg/foo.py` → `tests/test_foo.py`.
- Name tests `test_<unit>_<behaviour>`, e.g. `test_parse_rejects_empty_input`.
- One behaviour per test. Arrange / Act / Assert, separated by blank lines.
- Use fixtures (in `conftest.py` when shared) over setup helpers.
  Use `tmp_path` for files and `monkeypatch` for env vars; never touch the real FS/home.
- Use `pytest.mark.parametrize` instead of loops over cases.
- Test behaviour through the public API, not private helpers.
- Mock only at boundaries (network, clock, filesystem); don't mock the code under test.
- Bug fix ⇒ add a regression test that fails before the fix.
- Coverage gate is 80% (`make cov`). Don't write tests just to hit lines.
- Run a single test: `uv run pytest tests/test_x.py::test_name -x`.
