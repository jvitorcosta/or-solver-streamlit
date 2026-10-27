---
description: Review the current branch's changes against main
allowed-tools: Bash(git diff:*), Bash(git log:*), Bash(make:*), Read, Grep, Glob
---

## Context

- Commits on this branch: !`git log --oneline main..HEAD`
- Changed files: !`git diff --stat main...HEAD`

## Task

Use the **code-reviewer** agent to review `git diff main...HEAD` (plus any
uncommitted changes) against `CLAUDE.md` and `.claude/rules/`. If the diff
touches input handling, auth, subprocess, file paths or secrets, also run
the **security-auditor** agent.

Then run `make check` and report:

1. **Blocking**: bugs, missing tests, failing checks
2. **Should fix**: convention violations, unclear code
3. **Nits**: optional polish

Cite `file:line` for each point. Don't edit files unless I ask. $ARGUMENTS
