# Contributing to Genesys

Thanks for your interest in contributing! Genesys is open source under the Apache 2.0 license.

## Architecture overview

Genesys is a scoring engine, causal graph, and lifecycle manager for AI memory. The codebase is organized as:

```
src/genesys/
���── api.py              # Unified server — REST API + MCP over HTTP
├── engine/             # Scoring, transitions, forgetting, reactivation
│   ├── config.py       # All tunable thresholds (env-configurable)
│   ├── scoring.py      # Three-force multiplicative decay scoring
│   ├── transitions.py  # Status FSM (TAGGED → ACTIVE → EPISODIC → DORMANT)
│   ├── forgetting.py   # Conjunctive active forgetting
│   └── reactivation.py # BFS cascade reactivation
├── core_memory/        # Core promotion logic (graph-derived)
├── storage/            # Storage provider abstractions + implementations
│   ├── base.py         # Abstract interfaces (GraphStorageProvider, etc.)
│   ├── postgres.py     # Postgres + pgvector
│   ├── memory.py       # In-memory (zero deps)
│   └── ...             # FalkorDB, MongoDB, Obsidian
├── retrieval/          # Hybrid search (vector + keyword + graph)
├── mcp/                # MCP tool definitions
├── ingestion/          # Conversation history parsers (Claude, ChatGPT)
└── background/         # Async workers (entity extraction, causal inference)
```

Key design principles:
- **Storage abstraction is mandatory.** All data access goes through provider interfaces in `storage/base.py`. Never import a database client directly in business logic.
- **Background processing for heavy operations.** Embedding generation, entity extraction, LLM-based inference, and consolidation are all async background tasks.
- **The scoring formula is sacred.** `decay_score = relevance × connectivity × reactivation`. Multiplicative — zero on any axis means zero total.

## Getting started

1. Fork the repo and clone it
2. Copy `.env.example` to `.env` and fill in your API keys
3. Install dependencies: `pip install -e ".[dev]"`
4. Run tests: `pytest`

```bash
# In-memory mode (no Docker needed)
uvicorn genesys.api:app --reload --port 8000

# With Postgres
docker compose up -d postgres
GENESYS_BACKEND=postgres uvicorn genesys.api:app --reload --port 8000
```

## Code style

- **Python 3.11+**, async-first
- **Linting**: `ruff check src/` (config in `pyproject.toml`, line length 120)
- **Type checking**: `mypy src/genesys --ignore-missing-imports` (strict mode configured)
- **Formatting**: follow existing patterns — type hints throughout, minimal comments
- Engine thresholds live in `engine/config.py` and are env-configurable. Don't hardcode magic numbers in engine files.

## Testing

```bash
# Run all unit tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src/genesys --cov-report=term-missing

# Skip integration tests (require running Postgres)
pytest tests/ -v --ignore=tests/test_integration.py --ignore=tests/test_retrieval.py
```

Tests are in `tests/` and use `pytest-asyncio` (auto mode). When adding new engine logic, add corresponding test cases in the relevant test file.

## Pull requests

- Keep PRs focused — one feature or fix per PR
- Add tests for new functionality
- Make sure `pytest` and `ruff check src/` pass before submitting
- If your change touches engine logic (scoring, transitions, forgetting, promotion), run the full test suite — these modules are tightly coupled
- If your change touches a storage provider, test with both `memory` and `postgres` backends if possible

## Reporting issues

Open an issue on GitHub with:
- Steps to reproduce
- Python version and backend (`memory`, `postgres`, `obsidian`, or `falkordb`)
- Relevant error output or logs

## Good first issues

Look for issues labeled [`good first issue`](https://github.com/rishimeka/genesys/labels/good%20first%20issue). These are scoped, well-defined tasks suitable for new contributors.

## Code of conduct

Be kind. We're all here to build something useful.
