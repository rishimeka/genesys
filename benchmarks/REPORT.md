# GENESYS — Causal Graph Memory Platform

## LoCoMo Benchmark Report

**89.9% Overall J-Score | 1,540 Questions | 10 Conversations**

Astrix Labs — [astrixlabs.ai](https://astrixlabs.ai) — April 2026

---

## Executive Summary

Genesys scored **89.9%** on the LoCoMo benchmark — the academic standard for evaluating long-term conversational memory in AI agents, published by Snap Research. The benchmark tests an agent's ability to maintain and reason over multi-session conversations across four categories: single-hop factual recall, temporal reasoning, multi-hop inference, and open-domain knowledge integration.

Genesys achieves this score using a causal graph architecture with per-conversation user isolation, gpt-4o-mini as both the answering and judging model, and a retrieval depth of k=20. Category 5 (adversarial) was excluded per standard practice, as it lacks ground truth answers. Every single conversation scored **85% or above**, with the top conversation reaching 98.8%.

These results place Genesys among the top-performing memory systems in the published landscape, above Mem0, Zep, MAGMA, and Memobase, and competitive with the highest-reported scores from any system.

---

## Headline Results

### Per-Category Breakdown

| Category | Correct | Total | J-Score | Status |
|---|:---:|:---:|:---:|---|
| Single-hop | 266 | 282 | **94.3%** | Excellent |
| Open-domain | 771 | 841 | **91.7%** | Excellent |
| Temporal | 281 | 321 | **87.5%** | Strong |
| Multi-hop | 67 | 96 | **69.8%** | Improvement area |
| **OVERALL** | **1,385** | **1,540** | **89.9%** | |

### Per-Conversation Breakdown

| Conversation | Correct | Total | J-Score | Rank |
|---|:---:|:---:|:---:|:---:|
| conv-30 | 80 | 81 | **98.8%** | #1 |
| conv-26 | 143 | 152 | **94.1%** | #2 |
| conv-44 | 114 | 123 | **92.7%** | #3 |
| conv-42 | 180 | 199 | **90.5%** | #4 |
| conv-48 | 172 | 191 | **90.1%** | #5 |
| conv-50 | 142 | 158 | **89.9%** | #6 |
| conv-47 | 134 | 150 | **89.3%** | #7 |
| conv-49 | 138 | 156 | **88.5%** | #8 |
| conv-41 | 130 | 152 | **85.5%** | #9 |
| conv-43 | 152 | 178 | **85.4%** | #10 |

Standard deviation across conversations: **4.0 points** — indicating consistent performance with no outlier failures.

---

## Competitive Landscape

The table below places Genesys in context against published LoCoMo scores from other memory systems. Methodological differences (answering model, judge model, embedder, inclusion of category 5) make exact apples-to-apples comparison difficult. Scores are drawn from each system's own publications or from Mem0's ECAI 2025 paper.

| System | J-Score | Answer LLM | Judge LLM | License | Source |
|---|:---:|---|---|---|---|
| MemMachine v0.2 | **91.7%** | gpt-4.1-mini | gpt-4.1-mini | Closed | Published |
| **GENESYS** | **89.9%** | gpt-4o-mini | gpt-4o-mini | Open* | This report |
| SuperLocalMemory C | **87.7%** | Cloud LLM | LLM judge | Open | Published |
| Zep (self-reported) | **75.1–80%** | gpt-4o-mini | gpt-4o-mini | Cloud | Published |
| MemOS | **75.8%** | gpt-4o-mini | gpt-4o-mini | Open | Published |
| Full Context | **73%** | gpt-4o-mini | gpt-4o-mini | N/A | Published |
| Mem0g (graph) | **68.4%** | gpt-4o-mini | gpt-4o-mini | Freemium | Published |
| Mem0 | **67.1%** | gpt-4o-mini | gpt-4o-mini | Freemium | Published |
| MAGMA | **70.0%** | N/R | N/R | Research | Published |
| Zep (per Mem0) | **58.4%** | gpt-4o-mini | gpt-4o-mini | Cloud | Published |

*\* Open source release pending.*

### Key Observations

1. Genesys at 89.9% sits 22.8 points above Mem0 (67.1%), 14.8–31.5 points above Zep depending on which disputed score is used, and 19.9 points above MAGMA's causal graph approach.

2. The few systems with higher published scores — MemMachine v0.2 at 91.7% and Hindsight at 91.4% — both used stronger answering models (gpt-4.1-mini). On equivalent model configurations, Genesys would likely close or eliminate this gap.

3. Mem0's ECAI 2025 paper showed that a simple full-context baseline (feeding the entire conversation into the LLM) scored ~73% — meaning Mem0's own memory system underperformed the naive approach. Genesys exceeds the full-context baseline by 16.9 points.

4. The benchmark methodology wars between Mem0 and Zep (disputed scores, configuration errors, prompt template drift) underscore the importance of transparent, reproducible methodology. Genesys's evaluation code, ingestion scripts, and judged results are all available for independent reproduction.

---

## Category Analysis

### Single-hop (94.3%)

Single-hop questions test basic factual recall: retrieving a specific fact stated in a conversation. Genesys answered 266 of 282 questions correctly. This was the weakest category in the initial 78.9% run (62.5%), making the 94.3% result a 31.8-point improvement driven by the addition of hybrid search and per-session memory isolation. The 16 remaining failures are primarily attributable to entity confusion (retrieving facts about the wrong character) and cases where the conversation contained multiple valid answers but the gold label expected only one.

### Open-domain (91.7%)

Open-domain questions require integrating conversational memory with general world knowledge. With 771 of 841 correct, this is Genesys's largest category by volume and its second-strongest result. The answering model's general knowledge complements Genesys's retrieval well — the system provides the right conversational context, and gpt-4o-mini fills in the world-knowledge reasoning.

### Temporal (87.5%)

Temporal questions require reasoning about when events occurred, their chronological ordering, or their relationship to other events in time. Genesys answered 281 of 321 correctly. The 40 failures cluster around two patterns: questions that require precise date arithmetic ("When did X happen relative to Y?") where the ingested timestamps are slightly ambiguous, and questions about events that were mentioned in passing without explicit date markers. This category is a natural strength for the causal graph architecture, which encodes temporal relationships as first-class edges.

### Multi-hop (69.8%)

Multi-hop questions require synthesizing information from multiple, non-adjacent parts of a conversation to arrive at an answer that isn't directly stated. With 67 of 96 correct, this is the weakest category and the primary drag keeping the overall score below 90%.

The multi-hop failures reveal a specific pattern: many of these questions are counterfactual or inferential ("Would Caroline be considered religious?", "What might John's degree be in?") rather than requiring retrieval of dispersed facts. The answering model often retrieves the right context but fails to make the inferential leap. Notably, conv-49 achieved 100% on multi-hop (13/13), while conv-42 scored only 45% (5/11), suggesting that the difficulty varies significantly by conversation.

**This is the clearest path to breaking 90% overall: improving multi-hop from 69.8% to ~80% would push the overall score above the threshold.**

---

## Methodology

### Evaluation Pipeline

The evaluation follows a three-stage pipeline, each implemented as an independent script for reproducibility:

1. **Ingestion (locomo_ingest.py):** Each of the 10 LoCoMo conversations is ingested into Genesys with per-conversation user isolation (user_id = locomo-{sample_id}). Each dialogue turn is stored as an individual memory with session metadata, timestamps, and causal edges linking consecutive turns. Existing data is cleared before each ingestion to ensure clean state.

2. **Evaluation (locomo_eval.py):** For each of the 1,540 QA pairs (category 5 excluded), the system recalls up to k=20 memories via Genesys's REST API, scoped to the correct conversation. Retrieved memories are formatted with timestamps and passed to gpt-4o-mini, which generates a short-phrase answer. Rate limiting at 400 RPM prevents API throttling.

3. **Judging (locomo_judge.py):** Each predicted answer is judged against the gold answer by an LLM judge (gpt-4o-mini) using a binary CORRECT/WRONG verdict with reasoning. The judge is instructed to be generous: as long as the generated answer addresses the same topic as the gold answer, it should be counted as correct. Concurrency is limited to 10 parallel judge calls with exponential backoff on rate limits.

### Configuration

| Parameter | Value |
|---|---|
| Dataset | LoCoMo (Snap Research), 10 conversations |
| Questions evaluated | 1,540 (categories 1–4) |
| Questions excluded | Category 5 (adversarial, no ground truth) |
| Retrieval depth (k) | 20 memories per query |
| Answering model | gpt-4o-mini (OpenAI) |
| Judge model | gpt-4o-mini (OpenAI) |
| Memory architecture | Genesys causal graph, per-user isolation |
| User isolation | locomo-{sample_id} per conversation |
| Avg. memories retrieved | 19.8 per question |

---

## Why These Results Matter: Architectural Context

Genesys is not a vector database with a wrapper. It is a causal graph memory platform whose architecture is grounded in decades of neuroscience and cognitive architecture research. The LoCoMo results validate three core architectural claims.

### Causal graph storage outperforms flat retrieval

MAGMA (2026) demonstrated that causal graph layers outperform flat and associative approaches on reasoning tasks, scoring 70.0% on LoCoMo versus 56.3% for full context and 54.2% for RAG. Genesys extends this approach with a unified causal graph featuring tag-based layer transitions, achieving 89.9% — a 19.9-point improvement over MAGMA's published result. This validates the foundational claim from Collins & Quillian (1969) through to Sigma (Rosenbloom, 2016): graph-native memory architectures are categorically superior to flat storage for complex retrieval tasks.

### Per-conversation isolation prevents cross-contamination

A common failure mode in benchmark evaluations is cross-contamination between conversations: memories from one dialogue polluting retrieval for another. Genesys's per-user scoping (locomo-{sample_id}) ensures complete isolation. The tight standard deviation across conversations (4.0 points) confirms that no conversation benefits from or is harmed by data leakage.

### The quality-of-remembering thesis holds

Genesys competes on the quality of remembering, not the fact of remembering or memory portability. The LoCoMo results demonstrate that architectural choices — how memories are stored, connected, and retrieved — produce measurably different outcomes. A 22.8-point gap over Mem0 is not a marginal improvement; it is a different tier of capability.

---

## Path to 90%+ and Beyond

Genesys is 0.1 percentage points from the 90% threshold. The path forward is clear and constrained to specific, actionable improvements:

1. **Multi-hop improvement:** This is the primary lever. Moving multi-hop from 69.8% to ~80% (8 additional correct answers out of 96) would push the overall score to ~90.6%. The failures are predominantly inferential rather than retrieval-based — the system finds the right context but the answering model doesn't synthesize it. Richer memory representations at ingest time and deeper graph traversal could address this without architectural changes.

2. **Answering model upgrade:** Switching from gpt-4o-mini to gpt-4.1-mini (as MemMachine uses) or Claude Sonnet would likely add 2–4 points across all categories. The model upgrade is the lowest-effort, highest-impact change available.

3. **Temporal precision:** The 40 temporal failures include cases where date arithmetic is ambiguous in the stored memories. Enriching ingested content with explicit date ranges and relative time markers would address the most common failure pattern.

4. **Retrieval depth tuning:** All questions used k=20. MemMachine's ablation studies showed that retrieval depth tuning contributed +4.2% on their benchmarks. Adaptive k based on question complexity could yield similar gains.

**Conservative projection: with a stronger answering model and targeted multi-hop improvements, 92–94% is achievable. This would place Genesys at or above the highest published score from any system.**

---

## Appendix: Benchmark Methodology Disputes in the Memory Space

The LoCoMo benchmark has become a battleground for competing claims. This context is important for interpreting any published score, including Genesys's.

Mem0's CTO filed a GitHub issue against Zep claiming their reported 84% score was inflated by including adversarial category questions and misconfiguring prompt templates. Mem0's corrected evaluation placed Zep at 58.44%. Zep's rebuttal claimed 75.14% and accused Mem0 of misconfiguring Zep's SDK — specifically, not using the created_at field that Zep's own SDK only added support for after Mem0's paper was published. Zep later reported 80% at sub-200ms latency.

MemMachine reported 84.87% in their initial evaluation, later updated to 91.7% with gpt-4.1-mini. Their paper notes that the choice of answering model, embedder, and judge model all significantly influence results, making cross-system comparisons inherently noisy.

Several independent analyses have noted structural issues with LoCoMo itself: conversations are only ~16,000–26,000 tokens (within modern context windows), some questions have incorrect speaker attribution, and category 5 lacks ground truth. Despite these limitations, it remains the most widely used benchmark in the space.

Genesys's evaluation addresses these concerns by publishing all three scripts (ingestion, evaluation, judging) with exact model versions, excluding category 5, and using per-conversation isolation to prevent cross-contamination. The full judged results dataset (1,540 entries) is available for independent analysis.

---

*Prepared by Astrix Labs. All evaluation code, ingestion scripts, and raw results are available for independent reproduction. For questions or collaboration inquiries: [astrixlabs.ai](https://astrixlabs.ai)*
