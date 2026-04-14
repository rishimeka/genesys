# Genesys Benchmarks

## LoCoMo Evaluation

Evaluated on the [LoCoMo](https://arxiv.org/abs/2402.06397) long-conversation memory benchmark: 10 conversations, 1,540 questions (category 5 adversarial excluded), judged by LLM-as-Judge.

### Results (Run 9)

| Category | Correct | Total | J-Score |
|----------|---------|-------|---------|
| Single-hop | 266 | 282 | 94.3% |
| Temporal | 281 | 321 | 87.5% |
| Multi-hop | 67 | 96 | 69.8% |
| Open-domain | 771 | 841 | 91.7% |
| **Overall** | **1385** | **1540** | **89.9%** |

Answer model: `gpt-4o-mini` | Judge model: `gpt-4o-mini` | Retrieval k=20

### How to reproduce

**Prerequisites:**
- Genesys running with Postgres backend (`GENESYS_BACKEND=postgres`)
- OpenAI API key set in `.env`
- LoCoMo dataset at `benchmarks/locomo10.json` (10 conversations from the LoCoMo corpus)

**Run:**

```bash
# Start Genesys
docker compose up -d postgres
alembic upgrade head
GENESYS_BACKEND=postgres uvicorn genesys.api:app --port 8000 &

# Run full pipeline (ingest + eval + judge for all 10 conversations)
PYTHONPATH=src python benchmarks/locomo_run.py --k 20

# Results saved to benchmarks/locomo_judged.json
```

The pipeline runs each conversation independently: clears the database, ingests that conversation's memories (one memory per session), evaluates all questions, then moves to the next conversation. Scores are aggregated and judged at the end.

### Pipeline files

| File | Description |
|------|-------------|
| `locomo_run.py` | Full pipeline: ingest + eval + judge per conversation |
| `locomo_ingest.py` | Standalone ingestion script |
| `locomo_eval.py` | Standalone evaluation script |
| `locomo_judge.py` | Standalone judge script |
| `locomo_judged.json` | Full judged results (1,540 entries) |

### Memory granularity

Memories are stored at **session granularity** (one memory per conversation session). Each memory concatenates all turns within a session with speaker labels. This gives ~19-32 memories per conversation, allowing k=20 retrieval to cover most or all of the conversation context.
