"""PostToolUse hook: ruff-format and autofix the Python files Claude edits in this project.

Files outside the project are left alone: they belong to other repos, with their own style.
Never blocks: formatting problems are reported but the exit code is 0.
"""

import json
import os
import subprocess
import sys
from pathlib import Path


def should_format(file_path: str, project_dir: str) -> bool:
    """True for an existing .py/.pyi file inside the project directory."""
    if not file_path.endswith((".py", ".pyi")):
        return False
    path = Path(file_path).resolve()
    return path.is_file() and path.is_relative_to(Path(project_dir).resolve())


def main() -> int:
    payload = json.load(sys.stdin)
    file_path = payload.get("tool_input", {}).get("file_path", "")
    project_dir = os.environ.get("CLAUDE_PROJECT_DIR", ".")
    if not should_format(file_path, project_dir):
        return 0
    for args in (["ruff", "check", "--fix", "--quiet"], ["ruff", "format", "--quiet"]):
        subprocess.run(["uv", "run", "--quiet", *args, file_path], cwd=project_dir, check=False)  # noqa: S603, S607 - fixed argv, uv on PATH
    return 0


if __name__ == "__main__":
    sys.exit(main())
