from pathlib import Path
from typing import Any

import yaml


def parse_yaml_config_from_file_system(*, file_name: str) -> dict[str, Any]:
    """Parse YAML configuration file from config directory into dictionary.

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

    yaml_file_path = Path(__file__).parent / file_name.strip()

    try:
        yaml_file_content = yaml_file_path.read_text(encoding="utf-8")
        return yaml.safe_load(yaml_file_content) or {}
    except (FileNotFoundError, yaml.YAMLError, OSError):
        return {}


def extract_examples_from_resources_directory() -> dict[str, Any]:
    """Extract optimization problem examples from YAML file in resources directory.

    Returns:
        Dictionary containing example optimization problems.
        Returns empty dict if examples file is not found or invalid.
    """
    examples_yaml_file_path = Path(__file__).parents[2] / "resources" / "examples.yaml"

    try:
        examples_file_content = examples_yaml_file_path.read_text(encoding="utf-8")
        return yaml.safe_load(examples_file_content) or {}
    except (FileNotFoundError, yaml.YAMLError, OSError):
        return {}
