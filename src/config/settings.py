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
    if not file_name or not file_name.strip():
        raise ValueError("File name cannot be empty or None")

    config_directory = Path(__file__).parent
    file_path = config_directory / file_name.strip()

    if not file_path.exists():
        return {}

    try:
        with open(file_path, encoding="utf-8") as yaml_file:
            content = yaml.safe_load(yaml_file)
            return content if content is not None else {}
    except (yaml.YAMLError, OSError):
        # Return empty dict for invalid YAML or file access errors
        return {}


def load_optimization_examples() -> dict[str, Any]:
    """Load optimization problem examples from resources directory.

    Returns:
        Dictionary containing example optimization problems.
        Returns empty dict if examples file is not found or invalid.
    """
    resources_directory = Path(__file__).parents[2] / "resources"
    examples_file_path = resources_directory / "examples.yaml"

    if not examples_file_path.exists():
        return {}

    try:
        with open(examples_file_path, encoding="utf-8") as yaml_file:
            content = yaml.safe_load(yaml_file)
            return content if content is not None else {}
    except (yaml.YAMLError, OSError):
        return {}
