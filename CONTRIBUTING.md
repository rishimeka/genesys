# Contributing to Genesys

Thanks for your interest in contributing! Genesys is open source under the [GNU Affero General Public License v3.0](LICENSE).

## Contributor License Agreement

All contributors must agree to our [Contributor License Agreement](CLA.md) before their code can be merged. By opening a pull request, you acknowledge that you have read the CLA and agree to its terms. Your git commit metadata (name and email) serves as your electronic signature.

The CLA grants Astrix Labs the right to include your contributions under the project's license (AGPLv3) and, where necessary, under commercial license terms for the hosted Genesys service. You retain copyright over your contributions.

## Architecture overview

Genesys Memory is a scoring engine, causal graph, and lifecycle manager for AI memory. The codebase is organized as:

```
src/genesys_memory/
├── engine/             # Scoring, transitions, forgetting, reactivation
│   ├── config.py       # All tunable thresholds (env-configurable)
│   ├── scoring.py      # Three-force multiplicative decay scoring
│   ├── transitions.py  # Status FSM (TAGGED -> ACTIVE -> EPISODIC -> DORMANT)
│   ├── forgetting.py   # Conjunctive active forgetting
│   └── reactivation.py # BFS cascade reactivation
├── core_memory/        # Core promotion logic (graph-derived)
├── storage/            # Storage provider abstractions + in-memory implementation
│   ├── base.py         # Abstract interfaces (GraphStorageProvider, etc.)
│   └── memory.py       # In-memory (zero deps)
├── retrieval/          # Embedding providers (OpenAI, local sentence-transformers)
├── mcp/                # MCP tool definitions
├── server.py           # Lightweight stdio MCP server
└── providers.py        # Provider wiring (in-memory backend, optional embeddings)
```

Key design principles:
- **Storage abstraction is mandatory.** All data access goes through provider interfaces in `storage/base.py`. Never import a database client directly in business logic.
- **Background processing for heavy operations.** Embedding generation, entity extraction, LLM-based inference, and consolidation are all async background tasks.
- **The scoring formula is sacred.** `decay_score = relevance x connectivity x reactivation`. Multiplicative — zero on any axis means zero total.

## Getting started

1. Fork the repo and clone it
2. Install dependencies: `pip install -e ".[dev]"`
3. Run tests: `pytest`

```bash
# Run the lightweight MCP server (stdio transport, no infrastructure needed)
python -m genesys_memory
```

## Code style

- **Python 3.11+**, async-first
- **Linting**: `ruff check src/` (config in `pyproject.toml`, line length 120)
- **Type checking**: `mypy src/genesys_memory --ignore-missing-imports` (strict mode configured)
- **Formatting**: follow existing patterns — type hints throughout, minimal comments
- Engine thresholds live in `engine/config.py` and are env-configurable. Don't hardcode magic numbers in engine files.

## Testing

```bash
# Run all unit tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src/genesys_memory --cov-report=term-missing
```

Tests are in `tests/` and use `pytest-asyncio` (auto mode). When adding new engine logic, add corresponding test cases in the relevant test file.

## Pull requests

- Keep PRs focused — one feature or fix per PR
- Add tests for new functionality
- Make sure `pytest` and `ruff check src/` pass before submitting
- If your change touches engine logic (scoring, transitions, forgetting, promotion), run the full test suite — these modules are tightly coupled

## Reporting issues

Open an issue on GitHub with:
- Steps to reproduce
- Python version and backend
- Relevant error output or logs

## Good first issues

Look for issues labeled [`good first issue`](https://github.com/rishimeka/genesys/labels/good%20first%20issue). These are scoped, well-defined tasks suitable for new contributors.

## Code of conduct

Be kind. We're all here to build something useful.
