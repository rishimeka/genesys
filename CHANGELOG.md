# Changelog

## [0.4.2] - 2026-07-17

### Fixed

- **`memory_traverse` induced subgraph now includes the start node** and its incident edges. The in-memory provider returned the start node from `traverse()`; the Postgres provider excludes it (`WHERE id != start`), so on Postgres backends every edge incident to the start node was dropped from `edges` — including the sole match when `edge_types` filtered to a start-incident edge (e.g. traversing from an amended node with `edge_types=["supersedes"]` returned the right neighbor but zero edges). Normalized in the tool layer so all backends behave identically: the start node leads the node set and its edges appear.

## [0.4.1] - 2026-07-17

### Fixed (0.4.0 field-retest round 2)

- **Conflict-hint precision**: `possible_conflicts` numeric check now anchors numbers to the nearest *non-stopword* word and carries the number's unit (currency prefix, `%`, or following unit word). "latency is 200ms" no longer flags "budget is $50,000", and "6 weeks" vs "8 months" no longer collide; "budget $50,000" vs "budget $80,000" still fires.
- **Recall superseder down-ranking (latent)**: when the superseded memory fell outside the result set, recall marked the *superseding* memory as superseded and decayed it. SUPERSEDES is directed new→old; only targets are tagged now.
- **`memory_explain` edge direction**: edge entries gain a `direction` ("outgoing"/"incoming") field — an incoming `supersedes` edge previously read as if this node superseded the other. Legacy `target` (= other end) unchanged for compatibility.
- **Search/enumeration supersede visibility**: `memory_search` hits (both modes) now carry `superseded_by`, so change-cursor consumers can distinguish current from superseded memories without traversing edges.

## [0.4.0] - 2026-07-17

### Added

- **`memory_amend` MCP tool** (tool count 12 → 13). Records a correction as a new memory that `SUPERSEDES` an existing one, reusing the `memory_store` hot path (embedding + auto-link + eventing). The old memory is intentionally kept — recall already decays superseded hits and tags them `superseded_by`, and the transitions engine demotes them naturally. Ownership-checked. Returns `{node_id, supersedes, status: "amended"}`.
- **Writer-specified typed edges** in `memory_store` via a new `related` argument: `[{"id", "type"}]`, directed `new_node --type--> target`, validated fail-fast (invalid edge types rejected before the node is created). Writers can now express `supersedes` / `contradicts` / `part_of` etc. instead of everything collapsing to `caused_by`. The legacy `related_to` argument is unchanged (still `caused_by`).
- **`category` argument** on `memory_store`, closing the half-wired category path (nodes carried a `category` field and `list_core_memories` filtered on it, but there was no way to set it). A caller-set category is never overwritten by the background LLM classifier.
- **Concise recall**: `memory_recall(verbosity="concise")` returns only `id/summary/status/score/activation/is_core` (plus `superseded_by` when set) and skips the expensive causal-chain enrichment. `verbosity="full"` remains the default and is unchanged. Reactivation writes still occur in both modes.
- **`activation` alias** for `decay_score` on every recall hit and in `memory_explain`, plus a `score_model` block in `memory_explain` (formula, live per-force `connectivity_factor`/`activation_factor`, and a `staleness_note`). Makes the score legible: it is an activation/retention weight that *rises* on retrieval, not a countdown to deletion. See `docs/scoring.md`.
- **`memory_traverse` now returns edges**: `edges` (`source/target/type/weight/created_by`) and `edge_count` for the induced subgraph among reachable nodes — a superset of the BFS tree, so paths can be reconstructed. Backed by a new `get_connecting_edges()` storage method (edge-type + org-visibility filtering done in storage; de-duped by edge id). The tool degrades gracefully when the provider does not implement `get_connecting_edges` (returns `edges: []`, `edge_count: 0` — same guard pattern as `get_causal_chains_batch`). **Parity note for genesys-server:** the postgres, falkordb, mongo, and obsidian providers do NOT yet implement `get_connecting_edges` and need it added to return traverse edges in production; until then production `memory_traverse` returns empty `edges`.
- **`active_since` filter** on `memory_search` (filters on `last_reactivated_at`), a sibling of the existing `since` (which filters `created_at`). Both accept tz-naive ISO dates (treated as UTC).
- **Enumeration mode on `memory_search`** ("what's new since I last looked"): an empty `query` skips vector search entirely and lists nodes by `last_reactivated_at` descending, honoring the same filters — works with no embedder configured, no seed query needed. Search hits (both modes) now also carry `last_reactivated_at` and `source_session` (additive fields) for provenance.
- **Embedder-aware auto-link tuning**: `GENESYS_AUTOLINK_MIN_SIMILARITY` (env override), `GENESYS_AUTOLINK_MAX_EDGES` (default 3, per-store fan-out cap), and `GENESYS_AUTOLINK_MAX_NODE_DEGREE` (default 10, per-node *accumulation* cap — a node that already carries that many `auto_link` edges accretes no more, so hubs stop growing one edge per store). The similarity floor resolves via `config.resolve_autolink_min_similarity()` — explicit env > embedder recommendation (OpenAI 0.6 / local 0.45) > 0.45 fallback — replacing the hardcoded 0.3. The non-OpenAI floor sits *above* the local genuine-match band top (~0.4) because MiniLM noise pairs were observed at ~0.44: under local embeddings only near-duplicate content auto-links. `recommended_autolink_min_similarity` added to the embedding providers.
- **`possible_conflicts` hint** on `memory_store`: a pure-stdlib `heuristic_conflict_signal` flags lexical numeric/negation divergence against vector-similar candidates. Advisory only — never materialized as `CONTRADICTS` edges. The conflict scan has its **own floor**, decoupled from the auto-link floor (`GENESYS_CONFLICT_MIN_SIMILARITY`, defaulting to the recall floor via `config.resolve_conflict_min_similarity()`) and a wider candidate window (`GENESYS_CONFLICT_SCAN_K`, default 8) — so raising the auto-link floor doesn't shrink conflict detection. `numeric_mismatch` only fires when differing numbers appear in a *comparable position* (same nearest-preceding context word, e.g. "costs 50" vs "costs 75"), not on any two texts that merely both contain numbers (dates vs IDs no longer trigger it).
- **Structured tool errors in the stdio server**: `call_tool` no longer propagates tool exceptions as protocol-level MCP failures. Missing required arguments and tool exceptions return a `{"error": ..., "retryable": bool}` payload; `retryable` is true only for read tools (`memory_recall`/`memory_search`/`memory_traverse`/`memory_explain`/`memory_stats`/`list_core_memories`), matching the README retry guidance (never blind-retry writes).
- `docs/scoring.md` — the scoring legibility doc (three forces, stability, status/pinning override, conjunctive forgetting, worked numbers).
- `SUPPORTIVE_EDGE_TYPES` and `NEGATIVE_EDGE_TYPES` enum sets in `models/enums.py` for explicit edge classification.
- `get_supportive_degree()` method on all storage providers (base protocol, in-memory, Postgres, FalkorDB, MongoDB, Obsidian).
- LLM reasoning is now captured in contradiction detection and causal inference edge metadata. The `reason` field from `detect_contradiction` and `infer_causal_edges` is persisted in the edge's `metadata` dict.

### Changed

- **Auto-link is less permissive (fixes the "hairball")**: higher, embedder-aware similarity floor plus a per-store fan-out cap (`AUTOLINK_MAX_EDGES`) keep traversal neighborhoods scoped. Auto-linking now de-dupes against *any* existing edge between a pair (any type, either direction), so a `user_explicit` `caused_by` is never shadowed by a parallel `auto_link related_to`.
- `memory_store`'s `summary` is generated by word-boundary truncation (`≤200` chars incl. ellipsis) so words are never split mid-token. Still truncation, not an LLM summary.

- **Edge semantics correctness (breaking behavior change):** Nodes with only `CONTRADICTS` or `SUPERSEDES` edges are now considered orphans for forgetting purposes. Previously, any edge — including contradiction edges — prevented a node from being classified as an orphan, which meant contradicted memories were immune to pruning and could be incorrectly promoted to core status.

- **Core promotion now uses supportive degree:** The hub score component of `consolidation_score()` now counts only supportive edges (`CAUSED_BY`, `SUPPORTS`, `DERIVED_FROM`, `RELATED_TO`), excluding `CONTRADICTS` and `SUPERSEDES`. Nodes whose connectivity comes entirely from contradiction edges will no longer be promoted to core. **Note for existing instances:** This change is forward-only. Nodes already promoted to CORE status are not re-evaluated automatically (`evaluate_core_promotion` skips CORE nodes). Existing CORE nodes promoted under the old rules stay CORE until manually demoted via `unpin_memory`. Production instances upgrading to this version should consider running a one-time backfill to re-evaluate CORE nodes under the stricter rules — specifically, any CORE node whose only edges are CONTRADICTS/SUPERSEDES may have been mis-promoted.

- **Superseded nodes deprioritized in retrieval:** Nodes with incoming `SUPERSEDES` edges now receive a 0.3x multiplier on their retrieval rank score. They still appear in results (for audit trail purposes) but are ranked lower than their replacements.

### Fixed

- **Positional backward compatibility of `memory_store`**: the new `related` and `category` parameters are appended *after* the original positional tail (`created_at`, `visibility`, `org_id`), not inserted mid-signature — existing positional callers of the published API keep working.
- `memory_search`'s `since`/`active_since` filters no longer raise `TypeError` on tz-naive ISO input (e.g. `"2050-01-01"`); naive timestamps are normalized to UTC, same as `memory_store`'s `created_at`.
- `is_orphan()` across all storage providers now uses supportive degree instead of raw degree. A node with only negative edges (contradictions, supersessions) is correctly identified as an orphan.
- `get_orphans()` updated to match new orphan semantics across all providers.
- Contradiction reasoning was being requested from the LLM but discarded at parse time. Now captured and stored.
- Causal inference prompt now requests reasoning; previously only asked for target, type, and confidence.
