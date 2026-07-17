from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from datetime import datetime, timezone
from typing import Any

from genesys_memory.core_memory.preferences import CoreMemoryPreferences
from genesys_memory.core_memory.promoter import evaluate_core_promotion
from genesys_memory.models.edge import MemoryEdge
from genesys_memory.context import current_org_ids, current_user_id, current_user_role
from genesys_memory.models.enums import EdgeType, MemoryStatus, Visibility
from genesys_memory.models.node import MemoryNode
from genesys_memory.storage.base import CacheProvider, EmbeddingProvider, EventBusProvider, GraphStorageProvider

logger = logging.getLogger(__name__)


def _caller_uid() -> str | None:
    """Return the current user ID or None if not set."""
    return current_user_id.get(None)


def _caller_owns_node(node: MemoryNode) -> bool:
    """Check if the current caller is the original owner of a node.

    Raises PermissionError if current_user_id is not set. In multi-user
    mode, the node must have been created by the current user or the
    caller must be an admin within the node's org.
    """
    uid = _caller_uid()
    if uid is None:
        raise PermissionError("current_user_id not set — cannot verify ownership")
    role = current_user_role.get(None)
    if role == "admin" and node.org_id and node.org_id in current_org_ids.get([]):
        return True
    if node.original_user_id:
        return node.original_user_id == uid
    return True


def _is_edge_stale(edge: MemoryEdge) -> bool:
    from genesys_memory.engine import config
    if not edge.last_validated_at:
        return False
    days = (datetime.now(timezone.utc) - edge.last_validated_at).days
    return days > config.EDGE_STALE_DAYS


def _parse_iso_utc(value: str) -> datetime:
    """Parse an ISO 8601 string, assuming UTC when no timezone is given.

    Node timestamps are always tz-aware, and comparing a tz-aware datetime
    against a naive one raises TypeError — so every ISO timestamp accepted
    from a tool argument must go through this normalization.
    """
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _truncate_summary(content: str, limit: int = 200) -> str:
    """Truncate content to a word boundary. This is truncation, not an LLM summary.

    Returns short content verbatim; otherwise cuts at the last whole word that
    fits within ``limit`` (including the trailing ellipsis) so words are never
    split mid-token.
    """
    if len(content) <= limit:
        return content
    cut = content[: limit - 1]
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    return cut + "…"


def _live_connectivity_factor(causal_weight: int, max_causal_weight: int, is_orphan: bool) -> float:
    """Force-2 (connectivity) computed live for legibility.

    Mirrors the Force-2 block of ``scoring.calculate_decay_score`` exactly — the
    scoring math is not changed here, only surfaced so callers can read the live
    contribution instead of the stored (possibly stale) ``decay_score``.
    """
    import math

    from genesys_memory.engine import config

    if max_causal_weight > 0:
        raw = math.log2(1 + causal_weight) / math.log2(1 + max_causal_weight)
        cf = raw ** 2
    else:
        cf = 0.0
    if is_orphan:
        return 0.0
    if cf < config.MIN_CONNECTIVITY:
        cf = config.MIN_CONNECTIVITY
    return cf


class MCPToolHandler:
    def __init__(
        self,
        graph: GraphStorageProvider,
        embeddings: EmbeddingProvider | None,
        cache: CacheProvider,
        event_bus: EventBusProvider | None = None,
        on_change: Callable[..., Any] | None = None,
        llm: object | None = None,
    ):
        self.graph = graph
        self.embeddings = embeddings
        self.cache = cache
        self.event_bus = event_bus
        self.on_change = on_change
        self.llm = llm
        self.preferences = CoreMemoryPreferences(cache)

    def _defer_saves(self) -> AbstractContextManager[None]:
        """Return a context manager that batches saves if the graph supports it."""
        for cls in type(self.graph).__mro__:
            if 'defer_saves' in cls.__dict__:
                return self.graph.defer_saves()  # type: ignore[attr-defined,no-any-return]
        return nullcontext()

    async def _notify(self, event_type: str, data: dict[str, Any]) -> None:
        if self.on_change:
            await self.on_change(event_type, data)

    async def memory_store(
        self,
        content: str,
        source_session: str = "",
        related_to: list[str] | None = None,
        created_at: str | None = None,
        visibility: str = "private",
        org_id: str | None = None,
        related: list[dict[str, str]] | None = None,
        category: str | None = None,
    ) -> dict[str, Any]:
        """Store a new memory. Returns the node ID.

        Parameter order note: this is a published API — new parameters are
        appended AFTER the original positional tail (``created_at``,
        ``visibility``, ``org_id``) so existing positional callers keep
        working. Never insert parameters mid-signature.

        Args:
            related_to: Legacy; creates caused_by edges — prefer ``related`` for
                typed edges. Each entry is the id of an existing node.
            related: Typed explicit edges, each ``{"id": <node-id>, "type":
                <edge-type>}``. Direction convention is source = the new node,
                i.e. ``new_node --type--> target`` reads "new node <type>
                target" (e.g. ``supersedes`` → the new node supersedes the
                target). Invalid types are rejected before the node is created.
            category: Free-form classification string. Suggested vocabulary is
                the auto-promote categories (professional, educational, family,
                location) since those interact with core promotion, but any
                string is accepted.
            created_at: Optional ISO 8601 timestamp. Defaults to now.
            visibility: "private" or "org". Defaults to "private".
            org_id: Required when visibility is "org".

        The result may include ``possible_conflicts`` — heuristic hints (lexical
        numeric/negation divergence against auto-link candidates), not verified
        contradictions, and never materialized as edges.
        """
        vis = Visibility(visibility)
        if vis == Visibility.ORG and not org_id:
            return {"error": "org_id required when visibility is 'org'"}
        if vis == Visibility.ORG and org_id not in current_org_ids.get([]):
            return {"error": "org_id not in caller's org memberships"}

        # Validate typed relations up front — explicit writes must not
        # half-succeed (fail fast before the node is ever created).
        parsed_related: list[tuple[str, EdgeType]] = []
        if related:
            for item in related:
                raw_type = item.get("type")
                try:
                    etype = EdgeType(raw_type)
                except (ValueError, KeyError):
                    return {
                        "error": f"invalid edge type: {raw_type}",
                        "valid_types": [e.value for e in EdgeType],
                    }
                parsed_related.append((item["id"], etype))

        embedding = await self.embeddings.embed(content) if self.embeddings else []
        summary = _truncate_summary(content)

        ts = _parse_iso_utc(created_at) if created_at else datetime.now(timezone.utc)
        node = MemoryNode(
            status=MemoryStatus.ACTIVE,
            content_summary=summary,
            content_full=content,
            embedding=embedding,
            created_at=ts,
            last_accessed_at=ts,
            last_reactivated_at=ts,
            decay_score=1.0,
            causal_weight=0,
            source_agent="claude",
            source_session=source_session,
            category=category,
            visibility=vis,
            org_id=org_id,
            original_user_id=_caller_uid(),
        )

        node_id = await self.graph.create_node(node)

        # Explicit edges from related_to (legacy caused_by, with visibility check)
        if related_to:
            for target_id in related_to:
                target_node = await self.graph.get_node(target_id)
                if target_node is None:
                    logger.warning(
                        "related_to target %s not visible to caller %s; skipping edge",
                        target_id, _caller_uid(),
                    )
                    continue
                edge = MemoryEdge(
                    source_id=node.id,
                    target_id=uuid.UUID(target_id),
                    type=EdgeType.CAUSED_BY,
                    weight=0.7,
                    created_by="user_explicit",
                )
                await self.graph.create_edge(edge)

        # Typed explicit edges from `related` (writer-specified semantics)
        if parsed_related:
            for target_id, etype in parsed_related:
                target_node = await self.graph.get_node(target_id)
                if target_node is None:
                    logger.warning(
                        "related target %s not visible to caller %s; skipping edge",
                        target_id, _caller_uid(),
                    )
                    continue
                edge = MemoryEdge(
                    source_id=node.id,
                    target_id=uuid.UUID(target_id),
                    type=etype,
                    weight=0.7,
                    created_by="user_explicit",
                )
                await self.graph.create_edge(edge)

        # Auto-link to semantically similar existing memories
        possible_conflicts: list[dict[str, Any]] = []
        if embedding:
            try:
                from genesys_memory.engine import config
                from genesys_memory.engine.contradiction import heuristic_conflict_signal

                org_ids = current_org_ids.get([])
                min_sim = config.resolve_autolink_min_similarity(self.embeddings)
                # The conflict-hint scan uses its OWN (lower) floor: a changed
                # figure between two versions of a fact often lands below the
                # strict "clearly the same topic" auto-link band, so gating the
                # scan on the auto-link floor would silently shrink conflict
                # detection every time that floor is raised.
                conflict_min_sim = config.resolve_conflict_min_similarity(self.embeddings)
                # F6 dedupe: fetch this node's existing neighbors once. ANY edge
                # (any type, either direction) suppresses a duplicate auto-link,
                # so a user_explicit caused_by/supersedes never gets shadowed by
                # a parallel auto_link related_to.
                existing = await self.graph.get_edges(node_id, "both")
                linked_ids = {
                    str(e.target_id) if str(e.source_id) == node_id else str(e.source_id)
                    for e in existing
                }
                # +1 because the query's own top hit is usually the new node.
                # The window is the max of the auto-link fan-out and the
                # (wider) conflict-scan window.
                scan_k = max(config.AUTOLINK_MAX_EDGES + 1, config.CONFLICT_SCAN_K)
                similar = await self.graph.vector_search(
                    embedding, k=scan_k, org_ids=org_ids
                )
                links_created = 0
                with self._defer_saves():
                    for other_node, score in similar:
                        if str(other_node.id) == node_id:
                            continue
                        if score < min_sim and score < conflict_min_sim:
                            continue
                        # Org boundary rule: org nodes only link to same-org nodes
                        if vis == Visibility.ORG:
                            if other_node.visibility != Visibility.ORG or other_node.org_id != org_id:
                                continue
                        # Heuristic conflict hint — never creates structure.
                        if score >= conflict_min_sim:
                            signal = heuristic_conflict_signal(
                                content, other_node.content_full or other_node.content_summary
                            )
                            if signal:
                                possible_conflicts.append({
                                    "id": str(other_node.id),
                                    "summary": other_node.content_summary,
                                    "signal": signal,
                                })
                        if score < min_sim:
                            continue
                        # F6: skip if this pair is already linked by any edge.
                        if str(other_node.id) in linked_ids:
                            continue
                        if links_created >= config.AUTOLINK_MAX_EDGES:
                            # Fan-out cap reached; keep scanning for conflicts only.
                            continue
                        # F2 incoming-degree cap: fan-out alone doesn't stop a
                        # hub from *accreting* one link per store forever, so
                        # also bound the target's accumulated auto-link degree.
                        target_edges = await self.graph.get_edges(str(other_node.id), "both")
                        target_auto_degree = sum(
                            1 for e in target_edges if e.created_by == "auto_link"
                        )
                        if target_auto_degree >= config.AUTOLINK_MAX_NODE_DEGREE:
                            continue
                        edge = MemoryEdge(
                            source_id=node.id,
                            target_id=other_node.id,
                            type=EdgeType.RELATED_TO,
                            weight=round(score, 4),
                            reason=f"cosine similarity {score:.3f}",
                            created_by="auto_link",
                        )
                        await self.graph.create_edge(edge)
                        linked_ids.add(str(other_node.id))
                        links_created += 1
            except Exception:
                logger.warning("Auto-linking failed for node %s", node_id, exc_info=True)

        # Promote tagged → active if edges were formed (consolidation signal)
        has_edges = related_to or parsed_related or not await self.graph.is_orphan(node_id)
        if has_edges:
            await self.graph.update_node(node_id, {"status": MemoryStatus.ACTIVE})

        if self.event_bus:
            await self.event_bus.publish("memory.created", {
                "node_id": node_id,
                "content_full": content,
            })

        await self._notify("memory.created", {"node_id": node_id, "content": content[:200]})
        result: dict[str, Any] = {"node_id": node_id, "status": "stored", "visibility": vis.value}
        if possible_conflicts:
            result["possible_conflicts"] = possible_conflicts
        return result

    async def memory_amend(
        self, node_id: str, content: str, reason: str | None = None,
    ) -> dict[str, Any]:
        """Record a correction: create a new memory that supersedes an old one.

        The old memory is kept (its status is unchanged — recall already decays
        superseded hits by SUPERSEDED_DECAY and tags them ``superseded_by``, and
        the transitions engine demotes it naturally as it stops being recalled).
        Mutating status here would bypass that engine and be non-reversible.
        """
        old = await self.graph.get_node(node_id)
        if not old:
            return {"error": "Node not found", "node_id": node_id}
        if not _caller_owns_node(old):
            return {"error": "Only the node owner can amend this memory"}

        # Reuse the store hot path so the new node gets embedding + auto-link +
        # eventing. Passing `related` also lets F6 dedupe suppress any parallel
        # auto_link edge back to the old node.
        store_result = await self.memory_store(
            content,
            source_session=old.source_session,
            category=old.category,
            related=[{"id": node_id, "type": EdgeType.SUPERSEDES.value}],
            visibility=old.visibility.value,
            org_id=old.org_id,
        )
        if "error" in store_result:
            return store_result
        new_id = store_result["node_id"]

        # Upgrade the supersedes edge to a full-strength, reasoned correction.
        # No field-level edge mutator exists on the Protocol, so recreate it.
        for e in await self.graph.get_edges(new_id, "out", EdgeType.SUPERSEDES):
            if str(e.target_id) == node_id:
                await self.graph.delete_edge(str(e.id))
        await self.graph.create_edge(MemoryEdge(
            source_id=uuid.UUID(new_id),
            target_id=uuid.UUID(node_id),
            type=EdgeType.SUPERSEDES,
            weight=1.0,
            reason=reason or "amended",
            created_by="user_explicit",
        ))

        await self._notify("memory.amended", {"node_id": new_id, "supersedes": node_id})
        return {"node_id": new_id, "supersedes": node_id, "status": "amended"}

    async def memory_recall(
        self,
        query: str,
        k: int = 10,
        max_results: int | None = None,
        read_only: bool = False,
        verbosity: str = "full",
    ) -> dict[str, Any]:
        """Recall memories by hybrid search: vector + keyword, ranked by vector similarity.

        verbosity: "full" (default; unchanged payload) or "concise". Concise
        hits carry only id/summary/status/score/activation/is_core (plus
        superseded_by when set); the expensive causal-chain queries are skipped
        entirely. Reactivation writes still happen in both modes (governed by
        read_only, not verbosity).
        """
        import asyncio

        if verbosity not in ("full", "concise"):
            logger.warning("memory_recall verbosity=%r unrecognized; using 'full'", verbosity)
            verbosity = "full"

        MAX_K = 100
        if k > MAX_K:
            logger.info("memory_recall k=%d capped to %d", k, MAX_K)
            k = MAX_K

        # Extract keyword terms while embedding runs
        _base_stopwords = {
            "what", "where", "who", "how", "why", "which", "does", "did",
            "have", "are", "is", "the", "a", "an",
            "in", "on", "at", "to", "for", "of", "and", "or", "do",
            "from", "that", "this", "with", "about", "some", "any", "many",
            "much", "her", "his", "its", "their", "she", "he", "it", "they",
        }
        _temporal_stopwords = {"when", "has", "had", "was", "were", "been"}
        _temporal_signals = {
            "before", "after", "first", "last", "recently", "ago",
            "year", "month", "date", "time", "during", "until", "since",
        }
        query_lower = query.lower()
        query_words = query_lower.split()
        has_temporal = any(w.strip("?.,!'\"") in _temporal_signals for w in query_words)
        _stopwords = _base_stopwords if has_temporal else _base_stopwords | _temporal_stopwords
        terms = [w for w in query_words if w.strip("?.,!'\"") not in _stopwords and len(w) > 2]

        org_ids = current_org_ids.get([])
        # Run embedding + all keyword searches concurrently. Keyword search
        # coroutines are built once and MUST be awaited on every branch below
        # (no-embedder recall still relies on the keyword path).
        kw_coros = [self.graph.keyword_search(t.strip("?.,!'\""), k=k, org_ids=org_ids) for t in terms[:5]]
        if self.embeddings:
            embed_and_kw: list[Any] = await asyncio.gather(self.embeddings.embed(query), *kw_coros)
            embedding: list[float] = embed_and_kw[0]
            kw_results_per_term: list[list[MemoryNode]] = embed_and_kw[1:]
        else:
            embedding = []
            kw_results_per_term = await asyncio.gather(*kw_coros) if kw_coros else []

        # 1. Vector search (needs embedding)
        from genesys_memory.engine import config as engine_config
        min_sim = engine_config.resolve_recall_min_similarity(self.embeddings)

        vector_results = await self.graph.vector_search(embedding, k=k, org_ids=org_ids) if embedding else []

        # 2. Collect keyword hits
        kw_node_ids: set[str] = set()
        kw_nodes_map: dict[str, MemoryNode] = {}
        for kw_nodes in kw_results_per_term:
            for node in kw_nodes:
                nid = str(node.id)
                kw_node_ids.add(nid)
                kw_nodes_map[nid] = node

        # 3. Merge by union, track vector scores and keyword membership
        from genesys_memory.engine.scoring import cosine_similarity
        merged: dict[str, dict[str, Any]] = {}

        vector_ids: set[str] = set()
        for node, score in vector_results:
            nid = str(node.id)
            is_kw_hit = nid in kw_node_ids
            # A lexical keyword match is its own independent relevance
            # signal (stemmed term overlap with content), so it isn't
            # gated by the vector-similarity floor — only pure vector
            # matches are, to filter embedding noise.
            if score < min_sim and not is_kw_hit:
                continue
            vector_ids.add(nid)
            merged[nid] = {"node": node, "vec_score": score, "in_both": is_kw_hit}

        # Add keyword-only results — compute their vector similarity for
        # ranking only (not gated by min_sim; see comment above).
        for nid, node in kw_nodes_map.items():
            if nid not in merged:
                if node.embedding and embedding:
                    vec_score = cosine_similarity(node.embedding, embedding)
                else:
                    vec_score = 0.0
                merged[nid] = {"node": node, "vec_score": vec_score, "in_both": False}

        # 4. Format results without causal chains first (defer expensive graph queries)
        memories = []
        node_by_id: dict[str, MemoryNode] = {}
        for nid, info in merged.items():
            node = info["node"]
            node_by_id[nid] = node
            mem = self._format_memory_light(node, info["vec_score"])
            rank_score = info["vec_score"] + (0.1 if info["in_both"] else 0.0)
            mem["_rank_score"] = rank_score
            memories.append(mem)

            if self.event_bus and nid in vector_ids:
                await self.event_bus.publish("memory.accessed", {"node_id": nid})

        # Inject core memories not already in results. Auto-promoted core
        # memories are only injected when relevant to the query (avoids
        # noise); explicitly *pinned* memories are always injected — that's
        # the contract of pin_memory (always available regardless of query).
        core_min_sim = engine_config.resolve_core_inject_min_similarity(self.embeddings)
        core_nodes = await self.graph.get_nodes_by_status(MemoryStatus.CORE, limit=50)
        seen_ids = set(merged.keys())
        core_candidates: list[tuple[MemoryNode, float]] = []
        for cnode in core_nodes:
            cid = str(cnode.id)
            if cid not in seen_ids:
                core_sim = 0.0
                if cnode.embedding and embedding:
                    core_sim = cosine_similarity(cnode.embedding, embedding)
                if cnode.pinned or core_sim >= core_min_sim:
                    core_candidates.append((cnode, core_sim))
        core_candidates.sort(key=lambda x: (x[0].pinned, x[1]), reverse=True)
        for cnode, core_sim in core_candidates[:10]:
            cid = str(cnode.id)
            node_by_id[cid] = cnode
            mem = self._format_memory_light(cnode, core_sim)
            mem["is_core"] = True
            mem["_rank_score"] = core_sim
            memories.append(mem)

        # Batch superseded check + spreading activation: one get_all_edges call
        SUPERSEDED_DECAY = 0.3
        SPREAD_BONUS = 0.05
        all_mem_ids = [m["id"] for m in memories if m.get("id")]
        if all_mem_ids:
            try:
                all_mem_edges = await self.graph.get_all_edges(all_mem_ids)
                superseded_map: dict[str, str] = {}
                mem_id_set = set(all_mem_ids)
                neighbor_counts: dict[str, int] = {}
                for edge in all_mem_edges:
                    tgt = str(edge.target_id)
                    src = str(edge.source_id)
                    # A SUPERSEDES edge is directed new(source) -> old(target):
                    # only the TARGET is superseded. (A previous elif here
                    # marked the superseder itself as superseded whenever the
                    # old node fell out of the result set — down-ranking the
                    # correction it was supposed to prefer.)
                    if edge.type == EdgeType.SUPERSEDES and tgt in node_by_id:
                        superseded_map[tgt] = src
                    # Count edges between result set members for spreading activation
                    if src in mem_id_set and tgt in mem_id_set:
                        neighbor_counts[src] = neighbor_counts.get(src, 0) + 1
                        neighbor_counts[tgt] = neighbor_counts.get(tgt, 0) + 1
                for mem in memories:
                    mem_id = mem.get("id")
                    if not mem_id:
                        continue
                    if mem_id in superseded_map:
                        mem["_rank_score"] *= SUPERSEDED_DECAY
                        mem["superseded_by"] = superseded_map[mem_id]
                    # Spreading activation: boost memories connected to other results
                    spread = neighbor_counts.get(mem_id, 0)
                    if spread > 0:
                        mem["_rank_score"] += SPREAD_BONUS * spread
            except Exception:
                logger.warning("Superseded/spreading check failed", exc_info=True)

        memories.sort(key=lambda m: m["_rank_score"], reverse=True)
        for mem in memories:
            mem.pop("_rank_score", None)

        # Cap final results
        cap = max_results if max_results is not None else k
        memories = memories[:cap]

        # Enrich top results with causal chains (batch fetch). Skipped entirely
        # in concise mode — the chain queries are the main latency cost.
        org_ids_for_chain = current_org_ids.get([])
        mem_ids_for_chain = [m["id"] for m in memories if m.get("id")]
        if verbosity != "concise" and mem_ids_for_chain:
            try:
                has_batch = hasattr(self.graph, "get_causal_chains_batch")
                if has_batch:
                    upstream_map, downstream_map = await asyncio.gather(
                        self.graph.get_causal_chains_batch(mem_ids_for_chain, "upstream", org_ids=org_ids_for_chain),
                        self.graph.get_causal_chains_batch(mem_ids_for_chain, "downstream", org_ids=org_ids_for_chain),
                    )
                else:
                    chain_coros = []
                    for mid in mem_ids_for_chain:
                        chain_coros.append(self.graph.get_causal_chain(mid, "upstream", org_ids=org_ids_for_chain))
                        chain_coros.append(self.graph.get_causal_chain(mid, "downstream", org_ids=org_ids_for_chain))
                    chain_results = await asyncio.gather(*chain_coros, return_exceptions=True)
                    upstream_map = {}
                    downstream_map = {}
                    for i, mid in enumerate(mem_ids_for_chain):
                        up = chain_results[i * 2]
                        down = chain_results[i * 2 + 1]
                        upstream_map[mid] = [] if isinstance(up, BaseException) else up
                        downstream_map[mid] = [] if isinstance(down, BaseException) else down

                for mid in mem_ids_for_chain:
                    upstream = upstream_map.get(mid, [])
                    downstream = downstream_map.get(mid, [])
                    mem = next(m for m in memories if m.get("id") == mid)
                    causal_basis = []
                    causal_chain = []
                    seen_causal: set[str] = set()
                    for n in upstream[:10]:
                        nid_str = str(n.id)
                        if nid_str not in seen_causal:
                            causal_basis.append({"id": nid_str, "summary": n.content_summary, "direction": "upstream"})
                            seen_causal.add(nid_str)
                    for n in downstream[:10]:
                        nid_str = str(n.id)
                        if nid_str not in seen_causal:
                            causal_basis.append({"id": nid_str, "summary": n.content_summary, "direction": "downstream"})
                            seen_causal.add(nid_str)
                    if upstream:
                        for n in reversed(upstream[:10]):
                            causal_chain.append({"id": str(n.id), "summary": n.content_summary})
                        origin = node_by_id.get(mid)
                        if origin:
                            causal_chain.append({"id": mid, "summary": origin.content_summary})
                    mem["causal_basis"] = causal_basis
                    if causal_chain:
                        mem["causal_chain"] = causal_chain
            except Exception:
                logger.warning("Causal chain batch fetch failed", exc_info=True)

        # Update reactivation state + validate co-retrieval edges (skip in read_only mode)
        if not read_only:
            with self._defer_saves():
                now = datetime.now(timezone.utc)
                reactivation_coros = []
                reactivation_mems = []
                for mem in memories:
                    mem_id = mem.get("id")
                    if not mem_id:
                        continue
                    recalled = node_by_id.get(mem_id)
                    if recalled:
                        stability_delta = 0.1 / recalled.stability
                        reactivation_coros.append(
                            self.graph.atomic_reactivation_update(mem_id, now, stability_delta)
                        )
                        reactivation_mems.append((mem, recalled.reactivation_count + 1))
                if reactivation_coros:
                    results = await asyncio.gather(*reactivation_coros, return_exceptions=True)
                    for j, (mem, new_count) in enumerate(reactivation_mems):
                        if not isinstance(results[j], BaseException):
                            mem["reactivation_count"] = new_count

                if len(memories) > 1:
                    recalled_ids = {m["id"] for m in memories if m.get("id")}
                    try:
                        recall_edges = await self.graph.get_all_edges(list(recalled_ids))
                        validate_coros = [
                            self.graph.validate_edge(str(edge.id))
                            for edge in recall_edges
                            if str(edge.source_id) in recalled_ids and str(edge.target_id) in recalled_ids
                        ]
                        if validate_coros:
                            await asyncio.gather(*validate_coros, return_exceptions=True)
                    except Exception:
                        logger.warning("Co-retrieval edge validation failed", exc_info=True)

        if verbosity == "concise":
            concise: list[dict[str, Any]] = []
            for mem in memories:
                c: dict[str, Any] = {
                    "id": mem.get("id"),
                    "summary": mem.get("summary"),
                    "status": mem.get("status"),
                    "score": mem.get("score"),
                    "activation": mem.get("activation"),
                    "is_core": mem.get("is_core", False),
                }
                if "superseded_by" in mem:
                    c["superseded_by"] = mem["superseded_by"]
                concise.append(c)
            return {"query": query, "results": concise, "count": len(concise)}

        return {"query": query, "results": memories, "count": len(memories)}

    def _format_memory_light(self, node: MemoryNode, score: float) -> dict[str, Any]:
        """Format a memory node without causal chain queries (for ranking phase)."""
        return {
            "id": str(node.id),
            "content": node.content_full or node.content_summary,
            "summary": node.content_summary,
            "status": node.status.value,
            "decay_score": round(node.decay_score, 4),
            # `activation` is an alias for `decay_score`: same value, clearer
            # name. The score is an activation/retention weight (retrieval
            # RAISES it), not a countdown to deletion.
            "activation": round(node.decay_score, 4),
            "score": round(score, 4),
            "created_at": node.created_at.isoformat(),
            "causal_basis": [],
            "is_core": node.status == MemoryStatus.CORE,
        }

    async def _format_memory(self, node: MemoryNode, score: float) -> dict[str, Any]:
        """Format a memory node with causal chain info."""
        result = self._format_memory_light(node, score)
        org_ids = current_org_ids.get([])
        try:
            upstream = await self.graph.get_causal_chain(str(node.id), "upstream", org_ids=org_ids)
            downstream = await self.graph.get_causal_chain(str(node.id), "downstream", org_ids=org_ids)
            causal_basis = []
            seen = set()
            for n in upstream[:5]:
                if str(n.id) not in seen:
                    causal_basis.append({"id": str(n.id), "summary": n.content_summary, "direction": "upstream"})
                    seen.add(str(n.id))
            for n in downstream[:5]:
                if str(n.id) not in seen:
                    causal_basis.append({"id": str(n.id), "summary": n.content_summary, "direction": "downstream"})
                    seen.add(str(n.id))
            result["causal_basis"] = causal_basis
            if upstream:
                causal_chain = []
                for n in reversed(upstream[:5]):
                    causal_chain.append({"id": str(n.id), "summary": n.content_summary})
                causal_chain.append({"id": str(node.id), "summary": node.content_summary})
                result["causal_chain"] = causal_chain
        except Exception:
            logger.warning("Causal chain formatting failed for node %s", node.id, exc_info=True)
        return result

    @staticmethod
    def _node_passes_filters(node: MemoryNode, filters: dict[str, Any] | None) -> bool:
        """Apply memory_search's post-query filters to a node.

        ISO timestamps are normalized to UTC when tz-naive (matching
        memory_store's created_at handling) — node timestamps are always
        tz-aware, so comparing against a naive datetime would raise TypeError.
        """
        if not filters:
            return True
        if "category" in filters and node.category != filters["category"]:
            return False
        if "entity" in filters and filters["entity"] not in node.entity_refs:
            return False
        if "since" in filters:
            since_dt = _parse_iso_utc(filters["since"])
            if node.created_at < since_dt:
                return False
        if "active_since" in filters:
            active_dt = _parse_iso_utc(filters["active_since"])
            if node.last_reactivated_at < active_dt:
                return False
        return True

    @staticmethod
    def _format_search_hit(node: MemoryNode, score: float) -> dict[str, Any]:
        return {
            "id": str(node.id),
            "summary": node.content_summary,
            "status": node.status.value,
            "decay_score": round(node.decay_score, 4),
            "score": round(score, 4),
            "category": node.category,
            "created_at": node.created_at.isoformat(),
            # Provenance / recency fields (additive) so "what changed" answers
            # carry where and when a memory last moved.
            "last_reactivated_at": node.last_reactivated_at.isoformat(),
            "source_session": node.source_session,
        }

    async def memory_search(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        k: int = 10,
    ) -> dict[str, Any]:
        """Filtered vector search by status/category/date/entity.

        Enumeration mode (F8): an EMPTY ``query`` skips vector search entirely
        and lists memories by recency (``last_reactivated_at`` descending),
        honoring the same filters. Combined with ``since``/``active_since``
        this answers "what's new/changed since <ts>" without knowing what to
        query for, and works with no embedder configured.
        """
        # Determine status filter (shared by both modes)
        status_filter = None
        if filters and "status" in filters:
            status_filter = [MemoryStatus(s) for s in filters["status"]]

        if not query.strip():
            # Non-vector enumeration path.
            statuses = status_filter if status_filter is not None else list(MemoryStatus)
            nodes_by_id: dict[str, MemoryNode] = {}
            for status in statuses:
                for node in await self.graph.get_nodes_by_status(status, limit=1000):
                    nodes_by_id[str(node.id)] = node
            filtered = [n for n in nodes_by_id.values() if self._node_passes_filters(n, filters)]
            filtered.sort(key=lambda n: n.last_reactivated_at, reverse=True)
            memories = [self._format_search_hit(n, 0.0) for n in filtered[:k]]
            await self._tag_superseded(memories)
            return {"query": query, "results": memories, "count": len(memories)}

        if not self.embeddings:
            return {"query": query, "results": [], "count": 0}
        embedding = await self.embeddings.embed(query)

        results = await self.graph.vector_search(embedding, k=k, status_filter=status_filter)

        # Apply additional filters post-query
        memories = []
        for node, score in results:
            if not self._node_passes_filters(node, filters):
                continue
            memories.append(self._format_search_hit(node, score))

        await self._tag_superseded(memories)
        return {"query": query, "results": memories, "count": len(memories)}

    async def _tag_superseded(self, memories: list[dict[str, Any]]) -> None:
        """Mark search hits that a newer memory supersedes.

        Recall already tags superseded hits; without this, search/enumeration
        consumers (e.g. a change-cursor reader) see a superseded memory as
        indistinguishable from a current one. Direction: new(source) -> old
        (target), so only edge TARGETS get tagged.
        """
        ids = [m["id"] for m in memories if m.get("id")]
        if not ids:
            return
        try:
            by_id = {m["id"]: m for m in memories if m.get("id")}
            for edge in await self.graph.get_all_edges(ids):
                if edge.type == EdgeType.SUPERSEDES:
                    tgt = str(edge.target_id)
                    if tgt in by_id:
                        by_id[tgt]["superseded_by"] = str(edge.source_id)
        except Exception:
            logger.warning("superseded tagging failed", exc_info=True)

    async def memory_traverse(
        self,
        node_id: str,
        depth: int = 2,
        edge_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Subgraph traversal returning connected nodes."""
        MAX_DEPTH = 10
        if depth > MAX_DEPTH:
            logger.info("memory_traverse depth=%d capped to %d", depth, MAX_DEPTH)
            depth = MAX_DEPTH

        parsed_types = [EdgeType(t) for t in edge_types] if edge_types else None
        org_ids = current_org_ids.get([])
        nodes = await self.graph.traverse(node_id, depth, parsed_types, org_ids=org_ids)

        # Backends diverge on whether traverse() returns the start node itself:
        # the in-memory provider includes it, Postgres excludes it (WHERE id !=
        # start). The traversal's induced subgraph MUST contain the start node,
        # or every edge incident to it is dropped from `edges` — including the
        # only match when edge_types filters to a start-incident edge. Normalize
        # here so all backends behave identically: ensure the start node leads
        # the result set.
        if not any(str(n.id) == node_id for n in nodes):
            start = await self.graph.get_node(node_id)
            if start is not None:
                nodes = [start, *nodes]

        result_nodes = [
            {
                "id": str(n.id),
                "summary": n.content_summary,
                "status": n.status.value,
                "decay_score": round(n.decay_score, 4),
            }
            for n in nodes
        ]

        # Edges of the induced subgraph among the returned nodes (a deliberate
        # superset of the BFS tree, so callers can render/reconstruct paths).
        # Induced-subgraph + edge-type + visibility filtering all live in the
        # storage layer, so no tool-side re-filtering is needed.
        #
        # get_connecting_edges is a NEW Protocol method — Protocols are
        # structural, not enforced, so external providers (genesys-server's
        # postgres/falkordb/mongo/obsidian backends) may not implement it yet.
        # Same precedent as the get_causal_chains_batch guard in memory_recall:
        # degrade to an empty edge list instead of raising AttributeError and
        # breaking one of the original 11 tools on upgrade.
        node_ids = [str(n.id) for n in nodes]
        result_edges: list[dict[str, Any]] = []
        if hasattr(self.graph, "get_connecting_edges"):
            edges = await self.graph.get_connecting_edges(node_ids, parsed_types, org_ids=org_ids)
            result_edges = [
                {
                    "source": str(e.source_id),
                    "target": str(e.target_id),
                    "type": e.type.value,
                    "weight": round(e.weight, 4),
                    "created_by": e.created_by,
                }
                for e in edges
            ]

        return {
            "start_node": node_id,
            "depth": depth,
            "nodes": result_nodes,
            "count": len(result_nodes),
            "edges": result_edges,
            "edge_count": len(result_edges),
        }

    async def memory_explain(self, node_id: str) -> dict[str, Any]:
        """Score breakdown, causal basis, and removal impact for a memory."""
        node = await self.graph.get_node(node_id)
        if not node:
            return {"error": "Node not found", "node_id": node_id}

        causal_weight = await self.graph.get_causal_weight(node_id)
        is_orphan = await self.graph.is_orphan(node_id)

        upstream = await self.graph.get_causal_chain(node_id, "upstream")
        downstream = await self.graph.get_causal_chain(node_id, "downstream")

        all_edges = await self.graph.get_edges(node_id, "both")

        if downstream:
            removal_impact = f"Would orphan {len(downstream)} downstream memories"
        elif is_orphan:
            removal_impact = "Safe to remove — no connections"
        else:
            removal_impact = "Low impact — no downstream dependents"

        # Compute Force-2 (connectivity) and Force-3 (activation) LIVE from this
        # node's current edges/reactivation history, so the reader sees fresh
        # contributions rather than the stored (possibly stale) decay_score.
        import math

        from genesys_memory.engine import config
        from genesys_memory.engine.scoring import base_level_activation

        stats = await self.graph.get_stats()
        max_causal_weight = int(stats.get("max_causal_weight", 0) or 0)
        connectivity_factor_live = _live_connectivity_factor(
            causal_weight, max_causal_weight, is_orphan
        )
        b_i = base_level_activation(node.reactivation_timestamps, node.created_at)
        activation_factor_live = min(max(math.exp(b_i), 0.0), 1.0)

        return {
            "node_id": node_id,
            "summary": node.content_summary,
            "status": node.status.value,
            "decay_score": round(node.decay_score, 4),
            "activation": round(node.decay_score, 4),
            "score_model": {
                "formula": "decay_score = relevance x connectivity_factor x activation_factor",
                "reading": (
                    "Higher = more strongly retained. This is an activation/retention "
                    "score, not a countdown: retrieval RAISES it. A memory is prune-"
                    "eligible only when it falls below "
                    f"{config.FORGETTING_THRESHOLD} AND is orphaned AND not pinned."
                ),
                "forces": {
                    "relevance": "query-dependent; contributes at recall time, not at rest",
                    "connectivity_factor": round(connectivity_factor_live, 4),
                    "activation_factor": round(activation_factor_live, 4),
                },
                "staleness_note": (
                    "stored decay_score is recomputed by evaluate_transitions "
                    "(background worker in hosted deployments); live forces above "
                    "are computed fresh from this node's current edges and "
                    "reactivation history"
                ),
            },
            "stability": round(node.stability, 4),
            "causal_weight": causal_weight,
            "reactivation_count": node.reactivation_count,
            "reactivation_pattern": node.reactivation_pattern.value,
            "pinned": node.pinned,
            "is_orphan": is_orphan,
            "upstream_count": len(upstream),
            "downstream_count": len(downstream),
            "edge_count": len(all_edges),
            "edges": [
                {
                    "id": str(e.id),
                    "type": e.type.value,
                    "weight": round(e.weight, 4),
                    # "target" is the OTHER END of the edge viewed from this
                    # node (legacy name, kept for compatibility). "direction"
                    # disambiguates: an incoming supersedes edge means the
                    # other node supersedes THIS one, not vice versa.
                    "target": str(e.target_id) if str(e.source_id) == node_id else str(e.source_id),
                    "direction": "outgoing" if str(e.source_id) == node_id else "incoming",
                    "reason": e.reason,
                    "created_by": e.created_by,
                    "last_validated_at": e.last_validated_at.isoformat() if e.last_validated_at else None,
                    "stale": _is_edge_stale(e),
                }
                for e in all_edges
            ],
            "removal_impact": removal_impact,
        }

    async def pin_memory(self, node_id: str) -> dict[str, Any]:
        """Pin a memory to core status."""
        node = await self.graph.get_node(node_id)
        if not node:
            return {"error": "Node not found", "node_id": node_id}
        if not _caller_owns_node(node):
            return {"error": "Only the node owner can pin this memory"}

        await self.graph.update_node(node_id, {
            "pinned": True,
            "status": MemoryStatus.CORE,
            "promotion_reason": "user_pinned",
        })
        await self._notify("memory.pinned", {"node_id": node_id})
        return {"node_id": node_id, "status": "pinned", "new_status": "core"}

    async def unpin_memory(self, node_id: str) -> dict[str, Any]:
        """Unpin a memory and re-evaluate core eligibility."""
        node = await self.graph.get_node(node_id)
        if not node:
            return {"error": "Node not found", "node_id": node_id}
        if not _caller_owns_node(node):
            return {"error": "Only the node owner can unpin this memory"}

        await self.graph.update_node(node_id, {"pinned": False})

        # Re-evaluate: refresh node after unpin
        refreshed = await self.graph.get_node(node_id)
        if not refreshed:
            return {"node_id": node_id, "status": "unpinned", "new_status": "unknown"}
        should_stay_core, reason = await evaluate_core_promotion(refreshed, self.graph)

        if not should_stay_core:
            await self.graph.update_node(node_id, {
                "status": MemoryStatus.ACTIVE,
                "promotion_reason": None,
            })
            await self._notify("memory.unpinned", {"node_id": node_id, "new_status": "active"})
            return {"node_id": node_id, "status": "unpinned", "new_status": "active"}

        await self._notify("memory.unpinned", {"node_id": node_id, "new_status": "core"})
        return {"node_id": node_id, "status": "unpinned", "new_status": "core", "reason": reason}

    async def list_core_memories(self, category: str | None = None) -> dict[str, Any]:
        """List all core memories, optionally filtered by category."""
        nodes = await self.graph.get_nodes_by_status(MemoryStatus.CORE, limit=500)

        if category:
            nodes = [n for n in nodes if n.category == category]

        # Sort by causal_weight descending
        nodes.sort(key=lambda n: n.causal_weight, reverse=True)

        memories = [
            {
                "id": str(n.id),
                "summary": n.content_summary,
                "category": n.category,
                "causal_weight": n.causal_weight,
                "pinned": n.pinned,
                "promotion_reason": n.promotion_reason,
                "created_at": n.created_at.isoformat(),
            }
            for n in nodes
        ]

        return {"core_memories": memories, "count": len(memories)}

    async def delete_memory(self, node_id: str) -> dict[str, Any]:
        """Hard delete a memory node and all its edges."""
        node = await self.graph.get_node(node_id)
        if not node:
            return {"error": "Node not found", "node_id": node_id}
        if not _caller_owns_node(node):
            return {"error": "Only the node owner can delete this memory"}

        await self.graph.delete_node(node_id)
        await self._notify("memory.deleted", {"node_id": node_id})
        return {"node_id": node_id, "status": "deleted"}

    async def memory_stats(self) -> dict[str, Any]:
        """Return graph statistics."""
        stats = await self.graph.get_stats()
        return stats

    async def set_core_preferences(
        self,
        auto: list[str] | None = None,
        approval: list[str] | None = None,
        excluded: list[str] | None = None,
    ) -> dict[str, Any]:
        """Update core memory category preferences."""
        result = await self.preferences.update(auto=auto, approval=approval, excluded=excluded)
        return {"status": "updated", "preferences": result}

    async def promote_to_org(
        self,
        node_id: str,
        org_id: str,
        action: str = "keep_private",
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Promote a private memory to org visibility.

        Args:
            action: How to handle cross-boundary edges.
                "keep_private" (default) — edges to private nodes stay but are invisible to other org members.
                "promote_all" — also promote directly linked private nodes (1 level deep).
                "delete_links" — remove edges to private nodes before promoting.
            dry_run: If True, return a preview of what would change without applying it.
        """
        node = await self.graph.get_node(node_id)
        if not node:
            return {"error": "Node not found", "node_id": node_id}
        if node.visibility == Visibility.ORG:
            return {"error": "Already org-visible", "node_id": node_id}
        if org_id not in current_org_ids.get([]):
            return {"error": "org_id not in caller's org memberships"}
        if not _caller_owns_node(node):
            return {"error": "Only the node owner can promote to org"}

        # Check for cross-boundary edges (this node's edges to other private nodes)
        all_edges = await self.graph.get_edges(node_id, "both")
        cross_boundary: list[dict[str, Any]] = []
        for e in all_edges:
            other_id = str(e.target_id) if str(e.source_id) == node_id else str(e.source_id)
            other = await self.graph.get_node(other_id)
            if other and other.visibility == Visibility.PRIVATE:
                cross_boundary.append({
                    "edge_id": str(e.id),
                    "linked_node_id": other_id,
                    "linked_summary": other.content_summary,
                    "edge_type": e.type.value,
                })

        # Compute the diff for each action
        nodes_promoted: list[dict[str, str]] = [{"id": node_id, "summary": node.content_summary}]
        edges_deleted: list[dict[str, str]] = []
        edges_preserved: list[dict[str, str]] = []
        nodes_skipped: list[dict[str, str]] = []

        if action == "delete_links" and cross_boundary:
            for cb in cross_boundary:
                edges_deleted.append({"edge_id": cb["edge_id"], "linked_node_id": cb["linked_node_id"], "linked_summary": cb["linked_summary"]})
        elif action == "promote_all" and cross_boundary:
            for cb in cross_boundary:
                linked = await self.graph.get_node(cb["linked_node_id"])
                if linked and linked.visibility == Visibility.PRIVATE:
                    second_edges = await self.graph.get_edges(cb["linked_node_id"], "both")
                    has_further_private = False
                    for se in second_edges:
                        se_other = str(se.target_id) if str(se.source_id) == cb["linked_node_id"] else str(se.source_id)
                        if se_other == node_id:
                            continue
                        se_node = await self.graph.get_node(se_other)
                        if se_node and se_node.visibility == Visibility.PRIVATE:
                            has_further_private = True
                            break
                    if has_further_private:
                        nodes_skipped.append({"id": cb["linked_node_id"], "summary": cb["linked_summary"], "reason": "has further private links"})
                        continue
                    nodes_promoted.append({"id": cb["linked_node_id"], "summary": cb["linked_summary"]})
        elif action == "keep_private" and cross_boundary:
            for cb in cross_boundary:
                edges_preserved.append({"edge_id": cb["edge_id"], "linked_node_id": cb["linked_node_id"], "linked_summary": cb["linked_summary"]})

        if dry_run:
            return {
                "node_id": node_id,
                "dry_run": True,
                "action": action,
                "nodes_promoted": nodes_promoted,
                "edges_deleted": edges_deleted,
                "edges_preserved": edges_preserved,
                "nodes_skipped": nodes_skipped,
            }

        # Apply changes
        if action == "delete_links" and cross_boundary:
            for cb in cross_boundary:
                await self.graph.delete_edge(cb["edge_id"])
        elif action == "promote_all" and cross_boundary:
            for entry in nodes_promoted[1:]:
                await self.graph.promote_to_org(entry["id"], org_id)

        await self.graph.promote_to_org(node_id, org_id)
        await self._notify("memory.promoted", {"node_id": node_id, "org_id": org_id})
        return {
            "node_id": node_id,
            "status": "promoted",
            "org_id": org_id,
            "action": action,
            "nodes_promoted": nodes_promoted,
            "edges_deleted": edges_deleted,
            "edges_preserved": edges_preserved,
            "nodes_skipped": nodes_skipped,
        }

    async def erase_user(
        self, user_id: str, keep_promoted_nodes: bool = True,
    ) -> dict[str, Any]:
        """GDPR Article 17: erase all data belonging to a user.

        Args:
            keep_promoted_nodes: If True (default), org-promoted nodes are
                anonymized (original_user_id set to "erased_user") and PII
                is scrubbed from connected edge reasons. If False, all nodes
                are deleted regardless of visibility.
        """
        uid = _caller_uid()
        if uid is None:
            raise PermissionError("current_user_id not set — cannot erase")
        role = current_user_role.get(None)
        if uid != user_id and role != "admin":
            return {"error": "Only the user or an admin can erase user data"}

        manifest = await self.graph.erase_user(user_id, keep_promoted_nodes=keep_promoted_nodes)
        await self.cache.delete(f"core_preferences:{user_id}")
        await self._notify("user.erased", {"user_id": user_id, "manifest": manifest})
        return {
            "status": "erased",
            "user_id": user_id,
            **manifest,
        }
