# Changelog

## [Unreleased]

### Changed

- **Edge semantics correctness (breaking behavior change):** Nodes with only `CONTRADICTS` or `SUPERSEDES` edges are now considered orphans for forgetting purposes. Previously, any edge — including contradiction edges — prevented a node from being classified as an orphan, which meant contradicted memories were immune to pruning and could be incorrectly promoted to core status.

- **Core promotion now uses supportive degree:** The hub score component of `consolidation_score()` now counts only supportive edges (`CAUSED_BY`, `SUPPORTS`, `DERIVED_FROM`, `RELATED_TO`), excluding `CONTRADICTS` and `SUPERSEDES`. Nodes whose connectivity comes entirely from contradiction edges will no longer be promoted to core. **Note for existing instances:** This change is forward-only. Nodes already promoted to CORE status are not re-evaluated automatically (`evaluate_core_promotion` skips CORE nodes). Existing CORE nodes promoted under the old rules stay CORE until manually demoted via `unpin_memory`. Production instances upgrading to this version should consider running a one-time backfill to re-evaluate CORE nodes under the stricter rules — specifically, any CORE node whose only edges are CONTRADICTS/SUPERSEDES may have been mis-promoted.

- **Superseded nodes deprioritized in retrieval:** Nodes with incoming `SUPERSEDES` edges now receive a 0.3x multiplier on their retrieval rank score. They still appear in results (for audit trail purposes) but are ranked lower than their replacements.

### Added

- `SUPPORTIVE_EDGE_TYPES` and `NEGATIVE_EDGE_TYPES` enum sets in `models/enums.py` for explicit edge classification.
- `get_supportive_degree()` method on all storage providers (base protocol, in-memory, Postgres, FalkorDB, MongoDB, Obsidian).
- LLM reasoning is now captured in contradiction detection and causal inference edge metadata. The `reason` field from `detect_contradiction` and `infer_causal_edges` is persisted in the edge's `metadata` dict.

### Fixed

- `is_orphan()` across all storage providers now uses supportive degree instead of raw degree. A node with only negative edges (contradictions, supersessions) is correctly identified as an orphan.
- `get_orphans()` updated to match new orphan semantics across all providers.
- Contradiction reasoning was being requested from the LLM but discarded at parse time. Now captured and stored.
- Causal inference prompt now requests reasoning; previously only asked for target, type, and confidence.
