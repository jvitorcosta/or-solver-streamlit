"""PreToolUse hook: block unsafe or off-convention Bash commands.

Claude Code sends the tool call as JSON on stdin. Exit code 2 blocks the
command and shows stderr to Claude so it can correct course.
"""

import json
import re
import sys

RULES: list[tuple[str, str]] = [
    (
        r"\brm\s+-[a-z]*r[a-z]*f?[a-z]*\s+(/|~|\$HOME|\*)(\s|$)",
        "Refusing recursive delete of /, ~ or *.",
    ),
    (r"\bpip3?\s+install\b", "Use `uv add <pkg>` (or `uv add --dev`) instead of pip."),
    (r"\bpython3?\s+-m\s+pip\b", "Use `uv add` / `uv sync` instead of pip."),
    (r"\bgit\s+push\b.*(--force\b|-f\b)(?!-with-lease)", "Use --force-with-lease, never --force."),
    (r"\bgit\s+push\b.*\s(main|master)(\s|$)", "Don't push to main/master directly; open a PR."),
    (r"\bgit\s+commit\b.*--no-verify", "Don't skip pre-commit hooks; fix the failures."),
    (r"\b(curl|wget)\b[^|]*\|\s*(ba|z)?sh\b", "Don't pipe downloads into a shell."),
    (
        r"(^|[\s;&|])(cat|less|more|head|tail)\s+[^|;&]*\.env(\s|$)",
        "Don't read .env files; secrets stay out of context.",
    ),
]


def main() -> int:
    payload = json.load(sys.stdin)
    command: str = payload.get("tool_input", {}).get("command", "")
    for pattern, reason in RULES:
        if re.search(pattern, command):
            print(f"Blocked by .claude/hooks/validate_bash.py: {reason}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
