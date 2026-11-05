# Security

## Reporting a problem

Please report security problems privately, with [the private report form](https://github.com/jvitorcosta/or-solver-streamlit/security/advisories/new)
(under the Security tab), not in a public issue. Only the maintainer can read it.

Include what you found, how to reproduce it, and what it could let someone do. You'll get an answer
within a week. Fixes land on `main`; this project doesn't maintain older versions.

## How this repository is protected

- `main` can't be force-pushed or deleted, and it only changes through pull requests.
- Workflows run with a read-only token, and actions are pinned to full commit SHAs.
- Workflows from forks wait for the maintainer's approval before they run.
- Dependabot, CodeQL and secret scanning watch this repository.

