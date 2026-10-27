---
description: Fix a GitHub issue end-to-end (usage: /fix-issue <number>)
argument-hint: <issue-number>
allowed-tools: Bash(gh issue view:*), Bash(git switch:*), Bash(git status:*), Bash(git diff:*), Bash(make:*), Bash(uv run:*), Read, Edit, Write, Grep, Glob
---

Fix GitHub issue #$ARGUMENTS.

1. Read it: `gh issue view $ARGUMENTS --comments`. Restate the problem in one sentence.
   If it's ambiguous, stop and ask me before coding.
2. Create a branch: `git switch -c fix/$ARGUMENTS-<short-slug>`.
3. Find the relevant code (Grep/Glob). Explain the root cause briefly.
4. Write a **failing** regression test first; run it to confirm it fails.
5. Implement the smallest fix that makes it pass.
6. Run `make format && make check` until green.
7. Summarise the change and propose a Conventional Commit message ending in
   `Closes #$ARGUMENTS`. Don't commit or push unless I say so.
