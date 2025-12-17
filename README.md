# OR-Solver

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Modern Operations Research solver with academic-standard syntax for linear programming problems.

## Overview

OR-Solver is a web-based Linear Programming solver that accepts textbook-style mathematical syntax. The application provides an intuitive interface for solving optimization problems with detailed step-by-step explanations.

## Quick Start

1. **Clone the repository**:

   ```sh
   git clone https://github.com/jvitorcosta/or-solver-streamlit.git
   cd or-solver-streamlit
   ```

2. **Setup development environment**:

   > [!NOTE]
   > Requires Python 3.11+ and `uv` package manager

   ```sh
   make dev
   ```

3. **Run the application**:

   ```sh
   make run
   ```

   Visit `http://localhost:8501` to access the web interface.

## Usage

Write optimization problems using academic-standard mathematical syntax:

### Linear Programming Example
```text
# Production Planning Problem
maximize 40*product_a + 50*product_b

subject to:
    # Resource constraints
    2*product_a + 3*product_b <= 100    # Labor hours
    4*product_a + 2*product_b <= 120    # Raw materials
    product_a <= 30                     # Capacity A
    product_b <= 25                     # Capacity B

where:
    product_a, product_b >= 0           # Non-negativity
```

### Integer Programming Example
```text
# Facility Location Problem
minimize 50000*location_1 + 60000*location_2 + 45000*location_3

subject to:
    location_1 + location_2 >= 1       # Coverage requirement
    location_2 + location_3 >= 1       # Service constraint

where:
    location_1, location_2, location_3 >= 0
    location_1, location_2, location_3 integer  # Binary variables
```

### Supported Features
- **Objective Types**: `maximize` / `minimize` (or Portuguese `maximizar` / `minimizar`)
- **Constraints**: `<=`, `>=`, `=` with mathematical expressions
- **Variables**: Named variables with coefficients (explicit or implicit)
- **Comments**: Use `#` for inline documentation
- **Variable Types**: Continuous (default) or `integer` for discrete optimization
- **Languages**: English and Portuguese syntax support

The solver provides:
- Optimal solution values and objective function value
- Step-by-step solution explanations with mathematical reasoning
- Graphical visualization for 2D problems
- Detailed constraint analysis and sensitivity information

## Features

- **Interactive Web Interface**: Built with Streamlit for ease of use
- **Multilingual Support**: Available in English and Portuguese
- **Problem Templates**: Built-in gallery with common optimization problems
- **Detailed Solutions**: Step-by-step explanations of the solving process
- **Responsive Design**: Works on desktop and mobile devices
- **Academic Syntax**: Accepts standard mathematical optimization notation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Maintainer**: João Vitor Pinheiro da Costa - [GitHub](https://github.com/jvitorcosta) | [LinkedIn](https://linkedin.com/in/jvpro)
