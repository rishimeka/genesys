# Genesys

**The intelligence layer for AI memory.**

> Scoring engine + causal graph + lifecycle manager for AI agent memory. Speaks MCP natively.

## What is this

Genesys is not another vector database. It's a scoring engine + causal graph + lifecycle manager that makes AI memory actually *work*. Memories are scored by a multiplicative formula (relevance × connectivity × reactivation), connected in a causal graph, and actively forgotten when they become irrelevant. It plugs into any storage backend and speaks MCP natively.

## Why

- **Flat memory doesn't scale.** Dumping everything into a vector store gives you recall with zero understanding. The 500th memory buries the 5 that matter.
- **No forgetting = no intelligence.** Real memory systems forget. Without active pruning, your AI drowns in stale context.
- **No causal reasoning.** Vector similarity can't answer "why did I choose X?" — you need a graph.

Your AI remembers everything but understands nothing. Genesys fixes that.

## Quick Start

### Local mode (in-memory, no Docker)

```bash
git clone https://github.com/rishimeka/genesys.git
cd genesys
pip install -e .  # publishes as genesys-memory on PyPI
cp .env.example .env
# Set OPENAI_API_KEY in .env

uvicorn genesys.api:app --port 8000
```

### With Postgres + pgvector

```bash
git clone https://github.com/rishimeka/genesys.git
cd genesys
pip install -e ".[postgres]"  # or: pip install genesys-memory[postgres]
cp .env.example .env
# Set OPENAI_API_KEY and DATABASE_URL in .env

docker compose up -d postgres
alembic upgrade head
GENESYS_BACKEND=postgres uvicorn genesys.api:app --port 8000
```

## Connect to your AI

### Claude Code

```bash
claude mcp add --transport http genesys http://localhost:8000/mcp
```

### Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "genesys": {
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

### Any MCP client

Point your client at the MCP endpoint:

```
http://localhost:8000/mcp
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `memory_store` | Store a new memory, optionally linking to related memories |
| `memory_recall` | Recall memories by natural language query (vector + graph) |
| `memory_search` | Search memories with filters (status, date range, keyword) |
| `memory_traverse` | Walk the causal graph from a given memory node |
| `memory_explain` | Explain why a memory exists and its causal chain |
| `memory_stats` | Get memory system statistics |
| `pin_memory` | Pin a memory so it's never forgotten |
| `unpin_memory` | Unpin a previously pinned memory |
| `delete_memory` | Permanently delete a memory |
| `list_core_memories` | List core memories, optionally filtered by category |
| `set_core_preferences` | Set user preferences for core memory categories |

## How it works

Every memory is scored by three forces multiplied together:

```
decay_score = relevance × connectivity × reactivation
```

- **Relevance** decays over time. Old memories fade unless reinforced.
- **Connectivity** rewards memories with many causal links. Hub memories survive.
- **Reactivation** boosts memories that keep getting recalled. Frequency matters.

Because the formula is multiplicative, a memory must score on *all three* axes to survive. A highly connected but never-accessed memory still decays. A frequently recalled but causally orphaned memory still fades.

```
                    ┌─────────┐
                    │  STORE  │
                    └────┬────┘
                         │
                    ┌────▼────┐
                    │ ACTIVE  │◄──── reactivation
                    └────┬────┘
                         │ decay
                    ┌────▼────┐
                    │ DORMANT │
                    └────┬────┘
                         │ continued decay
                    ┌────▼────┐
           ┌───────│ FADING  │
           │        └─────────┘
           │ score=0, orphan,
           │ not pinned
      ┌────▼────┐
      │ PRUNED  │
      └─────────┘
```

Memories can also be promoted to **core** status — structurally important memories that are auto-pinned and never pruned.

## Benchmark Results

Tested on the [LoCoMo](https://arxiv.org/abs/2402.06397) long-conversation memory benchmark (1,540 questions across 10 conversations, category 5 excluded):

| Category | J-Score |
|----------|---------|
| Single-hop | 94.3% |
| Temporal | 87.5% |
| Multi-hop | 69.8% |
| Open-domain | 91.7% |
| **Overall** | **89.9%** |

Answer model: `gpt-4o-mini` | Judge model: `gpt-4o-mini` | Retrieval k=20

Full results and reproduction steps in [`benchmarks/`](benchmarks/).

## Storage backends

| Backend | Status | Use case |
|---------|--------|----------|
| `memory` | Built-in | Zero deps, try it out |
| `postgres` + pgvector | Production | Persistent, scalable |
| Obsidian | Coming soon | Local-first knowledge base |
| Custom | Bring your own | Implement `GraphStorageProvider` |

## Configuration

Copy `.env.example` to `.env` and set:

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | Embeddings |
| `ANTHROPIC_API_KEY` | No | LLM memory processing (consolidation, contradiction detection) |
| `GENESYS_BACKEND` | No | `memory` (default) or `postgres` |
| `DATABASE_URL` | If postgres | Postgres connection string |
| `GENESYS_USER_ID` | No | Default user ID for single-tenant mode |

See [`.env.example`](.env.example) for all options.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[Apache 2.0](LICENSE)
