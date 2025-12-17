"""Configuration and examples loading utilities."""

from pathlib import Path
from typing import Dict, Any
import yaml


def _get_config_directory_path() -> Path:
    """Get the path to the configuration directory.
    
    Returns:
        Path to the config directory (same as current file location).
    """
    return Path(__file__).parent


def load_yaml_configuration_file(*, file_name: str) -> Dict[str, Any]:
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
    
    config_directory = _get_config_directory_path()
    file_path = config_directory / file_name.strip()
    
    if not file_path.exists():
        return {}
    
    try:
        with open(file_path, "r", encoding="utf-8") as yaml_file:
            content = yaml.safe_load(yaml_file)
            return content if content is not None else {}
    except (yaml.YAMLError, OSError):
        # Return empty dict for invalid YAML or file access errors
        return {}


def load_optimization_examples() -> Dict[str, Any]:
    """Load optimization problem examples from configuration.
    
    Returns:
        Dictionary containing example optimization problems.
        Returns empty dict if examples file is not found or invalid.
    """
    return load_yaml_configuration_file(file_name="examples.yaml")


# Backward compatibility alias - prefer load_optimization_examples
def load_examples() -> Dict[str, Any]:
    """Load examples (backward compatibility alias).
    
    DEPRECATED: Use load_optimization_examples instead.
    
    Returns:
        Dictionary containing example problems.
    """
    return load_optimization_examples()