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

### Option 1: In-Memory (zero dependencies)

The fastest way to try Genesys. No database required — state is kept in memory and optionally persisted to a JSON file.

```bash
pip install genesys-memory
cp .env.example .env
# Set OPENAI_API_KEY in .env

uvicorn genesys.api:app --port 8000
```

To persist across restarts, set `GENESYS_PERSIST_PATH` in `.env`:

```env
GENESYS_PERSIST_PATH=.genesys_state.json
```

> **Give this to Claude to set it up for you:**
> *"Install genesys-memory, create a .env with my OpenAI key, start the server on port 8000 with the in-memory backend, and connect it as an MCP server."*

### Option 2: Postgres + pgvector (production)

Persistent, scalable storage with vector search via pgvector.

```bash
pip install 'genesys-memory[postgres]'
cp .env.example .env
```

Edit `.env`:

```env
OPENAI_API_KEY=sk-...
GENESYS_BACKEND=postgres
DATABASE_URL=postgresql://genesys:genesys@localhost:5432/genesys
```

Start Postgres and run migrations:

```bash
docker compose up -d postgres
alembic upgrade head
GENESYS_BACKEND=postgres uvicorn genesys.api:app --port 8000
```

> **Give this to Claude to set it up for you:**
> *"Install genesys-memory[postgres], start a Postgres container with pgvector using docker compose, run alembic migrations, create a .env with my OpenAI key and DATABASE_URL, start the server with GENESYS_BACKEND=postgres, and connect it as an MCP server."*

### Option 3: Obsidian Vault (local-first)

Turns your Obsidian vault into a Genesys memory store. Markdown files become memory nodes, `[[wikilinks]]` become causal edges. A SQLite sidecar (`.genesys/index.db`) handles indexing.

```bash
pip install 'genesys-memory[obsidian]'
cp .env.example .env
```

Edit `.env`:

```env
OPENAI_API_KEY=sk-...
GENESYS_BACKEND=obsidian
OBSIDIAN_VAULT_PATH=/path/to/your/vault
```

Start the server:

```bash
uvicorn genesys.api:app --port 8000
```

On first start, Genesys indexes all `.md` files in the vault and generates embeddings. A file watcher re-indexes incrementally when you edit notes.

> If `OBSIDIAN_VAULT_PATH` is not set, Genesys auto-detects by looking for `.obsidian/` in `~/Documents/personal`, `~/Documents/Obsidian`, and `~/obsidian`.

> **Give this to Claude to set it up for you:**
> *"Install genesys-memory[obsidian], create a .env with my OpenAI key, set GENESYS_BACKEND=obsidian and OBSIDIAN_VAULT_PATH to my vault at [YOUR_VAULT_PATH], start the server on port 8000, and connect it as an MCP server."*

### Option 4: FalkorDB (graph-native)

Uses FalkorDB (Redis-based graph database) for native graph traversal.

```bash
pip install 'genesys-memory[falkordb]'
cp .env.example .env
```

Edit `.env`:

```env
OPENAI_API_KEY=sk-...
GENESYS_BACKEND=falkordb
FALKORDB_HOST=localhost
```

Start FalkorDB and the server:

```bash
docker compose up -d falkordb
uvicorn genesys.api:app --port 8000
```

> **Give this to Claude to set it up for you:**
> *"Install genesys-memory[falkordb], start a FalkorDB container using docker compose, create a .env with my OpenAI key and GENESYS_BACKEND=falkordb, start the server on port 8000, and connect it as an MCP server."*

### From source

```bash
git clone https://github.com/rishimeka/genesys.git
cd genesys
pip install -e '.[dev]'
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

| Backend | Install | Use case |
|---------|---------|----------|
| `memory` | Built-in | Zero deps, try it out |
| `postgres` + pgvector | `pip install 'genesys-memory[postgres]'` | Persistent, scalable |
| Obsidian vault | `pip install 'genesys-memory[obsidian]'` | Local-first knowledge base |
| FalkorDB | `pip install 'genesys-memory[falkordb]'` | Graph-native traversal |
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
