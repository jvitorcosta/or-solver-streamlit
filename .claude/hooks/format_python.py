"""PostToolUse hook: ruff-format and autofix any Python file Claude edits.

Never blocks: formatting problems are reported but the exit code is 0.
"""

import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    payload = json.load(sys.stdin)
    file_path = payload.get("tool_input", {}).get("file_path", "")
    if not file_path.endswith((".py", ".pyi")) or not Path(file_path).exists():
        return 0

    cwd = os.environ.get("CLAUDE_PROJECT_DIR", ".")
    for args in (["ruff", "check", "--fix", "--quiet"], ["ruff", "format", "--quiet"]):
        subprocess.run(["uv", "run", "--quiet", *args, file_path], cwd=cwd, check=False)  # noqa: S603, S607 - fixed argv, uv on PATH
    return 0


if __name__ == "__main__":
    sys.exit(main())
