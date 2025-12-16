# 🐱 OR-Solver: Modern Operations Research Solver

> _A purr-fect tool for linear programming with academic-standard syntax!_

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Powered by Streamlit](https://img.shields.io/badge/powered%20by-Streamlit-red.svg)](https://streamlit.io/)

**OR-Solver** is a modern, multilingual Operations Research solver that makes linear programming as easy as petting a cat! 🐾 Built with clean architecture, academic-standard syntax, and beautiful interfaces for both CLI and web.

## 😻 Features

🐱 **Clean Academic Syntax**: Write LP problems like in textbooks
🌍 **Multilingual Support**: English and Portuguese interfaces
⚡ **Modern Architecture**: Clean layered design with type safety
🖥️ **Dual Interface**: Rich CLI with typer + Beautiful Streamlit web UI
🧪 **Robust Testing**: Property-based testing with hypothesis
📚 **Educational**: Step-by-step solution explanations
🎨 **Beautiful**: Rich terminal output and modern web design
🚀 **Fast**: Built with latest Python ecosystem (uv, ruff, pydantic)

## 😸 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/username/or-solver-streamlit
cd or-solver-streamlit

# Install with uv (recommended)
make install

# Or with pip
pip install -e .
```

### 🐾 Basic Usage

**Web Interface** (Streamlit):

```bash
make web
# or
or-solver web
```

**CLI Interface**:

```bash
# Solve a problem from file
or-solver solve problem.lp

# Interactive mode
or-solver solve --interactive

# Get help
or-solver --help
```

## 🙀 Syntax Guide

### Modern Academic Notation

Write your linear programming problems using clean, textbook-style syntax:

#### English Syntax (Default)

```
maximize 8*cat_food + 10*cat_toys

subject to:
    0.5*cat_food + 0.5*cat_toys <= 150
    0.6*cat_food + 0.4*cat_toys <= 145
    cat_food >= 30
    cat_toys >= 40

where:
    cat_food, cat_toys >= 0
```

#### Portuguese Syntax

```
maximizar 8*comida_gato + 10*brinquedos_gato

sujeito a:
    0.5*comida_gato + 0.5*brinquedos_gato <= 150
    0.6*comida_gato + 0.4*brinquedos_gato <= 145
    comida_gato >= 30
    brinquedos_gato >= 40

onde:
    comida_gato, brinquedos_gato >= 0
```

### Migration from Old Syntax

Convert old Portuguese colon-based syntax to modern notation:

```bash
# Automatic conversion
or-solver migrate old_problem.txt new_problem.lp
```

**Before** (old syntax):

```
problema: linear
max: 8x1 + 10x2
restricao: 0.5x1 + 0.5x2 <= 150
```

**After** (new syntax):

```
maximize 8*x1 + 10*x2
subject to:
    0.5*x1 + 0.5*x2 <= 150
where:
    x1, x2 >= 0
```

## 🎯 CLI Reference

### Core Commands

```bash
# 🐱 Solve optimization problems
or-solver solve <file>              # Solve from file
or-solver solve --interactive       # Interactive mode
or-solver solve --format json       # Output as JSON

# 😺 Problem validation and migration
or-solver validate <file>           # Validate syntax
or-solver migrate <old> <new>       # Convert old syntax

# 🐾 Examples and templates
or-solver examples list             # List available examples
or-solver examples run diet         # Run diet problem example
or-solver examples template knapsack # Generate knapsack template

# 🔧 Configuration
or-solver config set language en    # Set language (en/pt)
or-solver config set solver glop    # Set default solver
or-solver config show              # Show current config

# 📚 Educational features
or-solver explain <file>            # Step-by-step solution
or-solver syntax-help              # Syntax reference
```

### Rich Output Examples

The CLI provides beautiful, colored output with progress indicators and emoji:

```
🐱 Parsing problem... ✅ Done
😸 Setting up solver... ✅ Done
😻 Solving optimization... ✅ Done

🎯 Optimal Solution Found!
   Objective Value: 1250.0

   Variables:
   🐾 cat_food = 100.0
   🐾 cat_toys = 125.0

   Status: OPTIMAL 😺
```

## 🐾 Web Interface

Launch the beautiful Streamlit interface:

```bash
make web
```

### Features

- 🌍 **Language Toggle**: Switch between English/Portuguese
- 🎨 **Syntax Highlighting**: Real-time syntax coloring
- 🔄 **Live Conversion**: Old syntax → New syntax converter
- 📋 **Problem Templates**: Pre-built examples with cat themes
- 📊 **Visual Results**: Charts and tables for solutions
- 📚 **Educational Mode**: Step-by-step explanations
- 💾 **Export Options**: Save problems and solutions

## 😺 Examples Gallery

### Transportation Problem (Cat Food Distribution)

```
minimize 3*x11 + 2*x12 + 4*x21 + 3*x22

subject to:
    # Supply constraints (cat food factories)
    x11 + x12 <= 100  # Factory 1
    x21 + x22 <= 150  # Factory 2

    # Demand constraints (pet stores)
    x11 + x21 >= 80   # Store 1
    x12 + x22 >= 70   # Store 2

where:
    x11, x12, x21, x22 >= 0
```

### Diet Problem (Optimal Cat Nutrition)

```
minimize 2*kibble + 3*wet_food + 1*treats

subject to:
    # Nutritional requirements
    20*kibble + 15*wet_food + 5*treats >= 100   # Protein
    10*kibble + 25*wet_food + 2*treats >= 80    # Fat
    5*kibble + 10*wet_food + 15*treats >= 50    # Fiber

where:
    kibble, wet_food, treats >= 0
```

### Assignment Problem (Cat Toys to Cats)

```
minimize sum(cost[i,j] * assign[i,j] for i,j)

subject to:
    # Each cat gets exactly one toy
    sum(assign[i,j] for j) = 1  for all cats i

    # Each toy assigned to exactly one cat
    sum(assign[i,j] for i) = 1  for all toys j

where:
    assign[i,j] in {0, 1}
```

## 🔧 Development

### Setup Development Environment

```bash
# Install development dependencies
make dev

# Run tests
make test

# Lint and format
make lint
make format

# Build documentation
make docs
```

### Architecture

The project follows a clean layered architecture:

```
src/or_solver/
├── presentation/     # CLI and Web interfaces
├── application/      # Business logic services
├── domain/          # Core models and rules
└── infrastructure/  # External adapters (OR-Tools, I/O)
```

### Contributing

We welcome contributions! 🐱 Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📚 Documentation

- **Full Documentation**: [GitHub Pages](https://username.github.io/or-solver-streamlit)
- **API Reference**: Auto-generated from code
- **Tutorials**: Step-by-step guides
- **Examples**: Problem templates and solutions

## 🐾 Educational Use

OR-Solver is designed with education in mind:

- 📖 **Textbook Syntax**: Matches academic linear programming notation
- 🎓 **Step-by-Step**: Detailed solution explanations
- 🌍 **Multilingual**: Support for Portuguese and English
- 🎯 **Templates**: Common problem types with educational examples
- 📊 **Visualization**: Charts showing feasible regions and solutions

Perfect for:

- Operations Research courses
- Optimization workshops
- Self-study and learning
- Research and prototyping

## 🤝 Support

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/username/or-solver-streamlit/issues)
- 💡 **Feature Requests**: [GitHub Discussions](https://github.com/username/or-solver-streamlit/discussions)
- 📧 **Email**: <joao.pinheiro@example.com>
- 📚 **Documentation**: [Read the Docs](https://username.github.io/or-solver-streamlit)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- 🐱 **Google OR-Tools** - Powerful optimization library
- 🌟 **Streamlit** - Beautiful web applications
- ⚡ **Typer** - Modern CLI framework
- 🎨 **Rich** - Beautiful terminal output
- 🔬 **Hypothesis** - Property-based testing

---

Made with ❤️ and 🐱 by the OR-Solver team

_"Making optimization as easy as herding cats... wait, that doesn't sound right!"_ 😸

## Contributors

- **João Pinheiro (JV)** - Original creator and maintainer
