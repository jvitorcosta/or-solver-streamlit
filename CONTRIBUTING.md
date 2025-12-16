# 🐱 Contributing to OR-Solver

Welcome! We're excited you want to help make linear programming as easy as petting a cat! 😻

## 😸 Quick Start

### Setup
```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/or-solver-streamlit.git
cd or-solver-streamlit

# Install dependencies
make dev

# Run tests
make test
```

## 🧪 Testing

```bash
# Run all tests
make test

# Run specific test
uv run pytest tests/test_parser.py -v
```

### Writing Tests
- Add tests for new features
- Test both English and Portuguese syntax
- Use descriptive names with cat emojis! 🐾

Example:
```python
def test_parse_cat_food_problem():
    """Test parsing a cat food optimization problem."""
    problem_text = """
    maximize cat_food + cat_toys
    subject to:
        cat_food <= 100
    where:
        cat_food, cat_toys >= 0
    """
    
    problem = parse_lp_problem(problem_text)
    assert problem.objective.direction == ObjectiveDirection.MAXIMIZE
```

## 🎨 Code Style

We use automated formatting:

```bash
# Format code
make format

# Check style
make lint
```

**Guidelines:**
- Use type hints
- Add docstrings
- Use descriptive variable names
- Cat emojis are welcome! 😺

## 🚀 Pull Requests

1. **Fork** and create a feature branch
2. **Make changes** with tests
3. **Run** `make test` and `make lint`
4. **Submit** PR with clear description

## 🏗️ Architecture

Simple layered structure:
- `presentation/` - CLI and web interfaces
- `application/` - Business logic
- `domain/` - Core models and parsing
- `infrastructure/` - External tools (OR-Tools)

## 🐾 Issues

**Bug Reports:**
- Include Python version and OS
- Provide minimal example
- Rate bug severity on cat scale (1-10 cats) 😿

**Feature Requests:**
- Describe the use case
- Explain educational value
- How does it make the project more cat-like? 🐱

## 🤝 Community

Be kind, helpful, and have fun! We're all learning together.

---

*Happy coding! 😸*