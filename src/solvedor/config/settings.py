"""The gallery examples shipped in resources/examples.yaml."""

from functools import cache
from pathlib import Path
from typing import Any

import yaml

_EXAMPLES_FILE = Path(__file__).parents[1] / "resources" / "examples.yaml"


@cache
def extract_examples_from_resources_directory() -> dict[str, Any]:
    """The gallery examples by key. Loaded once; they ship inside the package."""
    return yaml.safe_load(_EXAMPLES_FILE.read_text(encoding="utf-8")) or {}
