"""PreToolUse hook: block unsafe or off-convention Bash commands.

Claude Code sends the tool call as JSON on stdin. Exit code 2 blocks the
command and shows stderr to Claude so it can correct course.

A command line is split into its parts (`;`, `&&`, `||`, `|`, newlines) and each
part is checked on its own, so `git push origin fix && gh pr create --base main`
is not mistaken for a push to main.
"""

import json
import re
import shlex
import sys

PROTECTED = {"main", "master"}

# Checked against each part of the command line.
RULES: list[tuple[str, str]] = [
    (
        r"\brm\s+-[a-z]*r[a-z]*f?[a-z]*\s+(/|~|\$HOME|\*)(\s|$)",
        "Refusing recursive delete of /, ~ or *.",
    ),
    (r"\bpip3?\s+install\b", "Use `uv add <pkg>` (or `uv add --dev`) instead of pip."),
    (r"\bpython3?\s+-m\s+pip\b", "Use `uv add` / `uv sync` instead of pip."),
    (r"\bgit\s+commit\b.*\s--no-verify\b", "Don't skip pre-commit hooks; fix the failures."),
    (
        r"(^|\s)(cat|less|more|head|tail)\s+[^|;&]*\.env(\s|$)",
        "Don't read .env files; secrets stay out of context.",
    ),
]

# Checked against the whole line, since a pipe joins the two halves.
PIPE_RULES: list[tuple[str, str]] = [
    (r"\b(curl|wget)\b[^|;&]*\|\s*(ba|z)?sh\b", "Don't pipe downloads into a shell."),
]


def split_parts(command: str) -> list[str]:
    """The simple commands in a command line, split on ;, &&, ||, | and newlines."""
    return [part.strip() for part in re.split(r"&&|\|\||[;|\n]", command) if part.strip()]


def git_push_problem(part: str) -> str | None:
    """Why a `git push ...` part is not allowed, or None."""
    try:
        words = shlex.split(part, posix=True)
    except ValueError:
        words = part.split()
    for i in range(len(words) - 1):
        if words[i] == "git" and words[i + 1] == "push":
            args = words[i + 2 :]
            break
    else:
        return None

    for arg in args:
        if arg in ("--force", "-f") or (
            arg.startswith("-") and not arg.startswith("--") and "f" in arg[1:]
        ):
            return "Use --force-with-lease, never --force."
    refspecs = [arg for arg in args if not arg.startswith("-")][1:]  # the first one is the remote
    for spec in refspecs:
        if spec.startswith("+"):
            return "Use --force-with-lease, never a forced (+) refspec."
        target = spec.split(":")[-1].removeprefix("refs/heads/")
        if target in PROTECTED:
            return "Don't push to main/master directly; open a PR."
    return None


def check(command: str) -> str | None:
    """Why the command is blocked, or None when it may run."""
    for pattern, reason in PIPE_RULES:
        if re.search(pattern, command):
            return reason
    for part in split_parts(command):
        if reason := git_push_problem(part):
            return reason
        for pattern, reason in RULES:
            if re.search(pattern, part):
                return reason
    return None


def main() -> int:
    payload = json.load(sys.stdin)
    command: str = payload.get("tool_input", {}).get("command", "")
    if reason := check(command):
        print(f"Blocked by .claude/hooks/validate_bash.py: {reason}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
