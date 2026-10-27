---
name: release
description: Cut a new release of this package — bump the version with uv, update CHANGELOG.md, tag, and build. Use when the user asks to release, publish, bump the version, or tag a version.
---

# Release

Follow these steps in order. Stop and ask if any check fails.

1. **Preflight**
   - `git status` must be clean and on `main`, up to date with origin.
   - `make check` must pass.
2. **Pick the bump** from commits since the last tag
   (`git log $(git describe --tags --abbrev=0 2>/dev/null || git rev-list --max-parents=0 HEAD)..HEAD --oneline`):
   - any `feat!:` / `BREAKING CHANGE` → `major`
   - any `feat:` → `minor`
   - otherwise → `patch`
   Confirm the choice with the user.
3. **Bump**: `uv version --bump <major|minor|patch>` (updates `pyproject.toml` and `uv.lock`).
4. **Changelog**: prepend a section to `CHANGELOG.md` using the format in
   [changelog-format.md](changelog-format.md).
5. **Build**: `uv build` and check `dist/` has a wheel and sdist.
6. **Commit & tag** (ask first):
   `git commit -am "chore(release): vX.Y.Z"` then `git tag vX.Y.Z`.
7. Pushing (`git push --follow-tags`) and publishing (`uv publish`) are **always**
   done only after explicit user confirmation.
