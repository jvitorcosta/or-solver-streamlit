from pathlib import Path
from typing import Any

import yaml


def load_yaml_configuration_file(*, file_name: str) -> dict[str, Any]:
    """Load YAML configuration file from config directory.

    Args:
        file_name: Name of the YAML file to load (e.g., 'examples.yaml').

    Returns:
        Dictionary containing the YAML file contents.
        Returns empty dict if file doesn't exist or is invalid.

    Raises:
        ValueError: If file_name is empty or None.
    """
    if not (file_name and file_name.strip()):
        raise ValueError("File name cannot be empty or None")

    file_path = Path(__file__).parent / file_name.strip()

    try:
        content = file_path.read_text(encoding="utf-8")
        return yaml.safe_load(content) or {}
    except (FileNotFoundError, yaml.YAMLError, OSError):
        return {}


def load_optimization_examples() -> dict[str, Any]:
    """Load optimization problem examples from resources directory.

    Returns:
        Dictionary containing example optimization problems.
        Returns empty dict if examples file is not found or invalid.
    """
    examples_path = Path(__file__).parents[2] / "resources" / "examples.yaml"

    try:
        content = examples_path.read_text(encoding="utf-8")
        return yaml.safe_load(content) or {}
    except (FileNotFoundError, yaml.YAMLError, OSError):
        return {}
