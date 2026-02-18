# Contributing to Felix Agent SDK

Thank you for contributing! This document covers setting up your development environment.

## Development Setup

```bash
git clone https://github.com/AppSprout-dev/felix-agent-sdk.git
cd felix-agent-sdk
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/
pytest tests/ -v --cov=src
```

## Code Style

We use `ruff` for linting and formatting:

```bash
ruff check src/
ruff format src/
mypy src/
```

## Submitting Changes

1. Fork the repo and create a feature branch from `main`
2. Write tests for new functionality
3. Ensure `pytest` and `ruff` pass
4. Open a pull request — `@CalebisGross` will review core modules

## Branch Protection

The `main` branch requires at least one approving review before merge.
Direct pushes to `main` are not permitted after initial setup.

## Git Attribution

When porting code from the original Felix framework:

- **Unchanged algorithms**: Use `git commit --author="Caleb Gross <209704970+CalebisGross@users.noreply.github.com>"`
- **Refactored code**: Add `Co-authored-by: Caleb Gross <209704970+CalebisGross@users.noreply.github.com>` trailer
- **New code**: Normal commits
