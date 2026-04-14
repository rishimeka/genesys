# Contributing to Genesys

Thanks for your interest in contributing! Genesys is open source under the Apache 2.0 license.

## Getting started

1. Fork the repo and clone it
2. Copy `.env.example` to `.env` and fill in your API keys
3. Install dependencies: `pip install -e ".[dev]"`
4. Run tests: `pytest`

## Development

```bash
# In-memory mode (no Docker needed)
uvicorn genesys.api:app --reload --port 8000

# With Postgres
docker compose up -d postgres
GENESYS_BACKEND=postgres uvicorn genesys.api:app --reload --port 8000
```

## Pull requests

- Keep PRs focused — one feature or fix per PR
- Add tests for new functionality
- Make sure `pytest` passes before submitting

## Reporting issues

Open an issue on GitHub with steps to reproduce. Include your Python version and backend (`memory` or `postgres`).

## Code of conduct

Be kind. We're all here to build something useful.
