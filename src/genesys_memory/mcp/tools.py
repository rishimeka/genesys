from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
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


class MCPToolHandler:
    def __init__(
        self,
        graph: GraphStorageProvider,
        embeddings: EmbeddingProvider | None,
        cache: CacheProvider,
        event_bus: EventBusProvider | None = None,
        on_change: Callable[..., Any] | None = None,
    ):
        self.graph = graph
        self.embeddings = embeddings
        self.cache = cache
        self.event_bus = event_bus
        self.on_change = on_change
        self.preferences = CoreMemoryPreferences(cache)

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
    ) -> dict[str, Any]:
        """Store a new memory. Returns the node ID.

        Args:
            created_at: Optional ISO 8601 timestamp. Defaults to now.
            visibility: "private" or "org". Defaults to "private".
            org_id: Required when visibility is "org".
        """
        vis = Visibility(visibility)
        if vis == Visibility.ORG and not org_id:
            return {"error": "org_id required when visibility is 'org'"}
        if vis == Visibility.ORG and org_id not in current_org_ids.get([]):
            return {"error": "org_id not in caller's org memberships"}

        embedding = await self.embeddings.embed(content) if self.embeddings else []
        summary = content[:200]

        ts = datetime.fromisoformat(created_at) if created_at else datetime.now(timezone.utc)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
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
            visibility=vis,
            org_id=org_id,
            original_user_id=_caller_uid(),
        )

        node_id = await self.graph.create_node(node)

        # Explicit edges from related_to (with visibility check)
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

        # Auto-link to semantically similar existing memories
        if embedding:
            try:
                org_ids = current_org_ids.get([])
                similar = await self.graph.vector_search(embedding, k=4, org_ids=org_ids)
                for other_node, score in similar:
                    if str(other_node.id) == node_id:
                        continue
                    if score < 0.3:
                        continue
                    # Org boundary rule: org nodes only link to same-org nodes
                    if vis == Visibility.ORG:
                        if other_node.visibility != Visibility.ORG or other_node.org_id != org_id:
                            continue
                    already = await self.graph.edge_exists(node_id, str(other_node.id), EdgeType.RELATED_TO)
                    if not already:
                        edge = MemoryEdge(
                            source_id=node.id,
                            target_id=other_node.id,
                            type=EdgeType.RELATED_TO,
                            weight=round(score, 4),
                            reason=f"cosine similarity {score:.3f}",
                            created_by="auto_link",
                        )
                        await self.graph.create_edge(edge)
            except Exception:
                logger.warning("Auto-linking failed for node %s", node_id, exc_info=True)

        # Promote tagged → active if edges were formed (consolidation signal)
        has_edges = related_to or not await self.graph.is_orphan(node_id)
        if has_edges:
            await self.graph.update_node(node_id, {"status": MemoryStatus.ACTIVE})

        if self.event_bus:
            await self.event_bus.publish("memory.created", {
                "node_id": node_id,
                "content_full": content,
            })

        await self._notify("memory.created", {"node_id": node_id, "content": content[:200]})
        return {"node_id": node_id, "status": "stored", "visibility": vis.value}

    async def memory_recall(
        self,
        query: str,
        k: int = 10,
        max_results: int | None = None,
        read_only: bool = False,
    ) -> dict[str, Any]:
        """Recall memories by hybrid search: vector + keyword, ranked by vector similarity."""
        import asyncio

        MAX_K = 100
        if k > MAX_K:
            logger.info("memory_recall k=%d capped to %d", k, MAX_K)
            k = MAX_K

        # Extract keyword terms while embedding runs
        _stopwords = {
            "what", "when", "where", "who", "how", "why", "which", "does", "did",
            "has", "have", "had", "was", "were", "are", "is", "the", "a", "an",
            "in", "on", "at", "to", "for", "of", "and", "or", "do", "been",
            "from", "that", "this", "with", "about", "some", "any", "many",
            "much", "her", "his", "its", "their", "she", "he", "it", "they",
        }
        terms = [w for w in query.lower().split() if w.strip("?.,!'\"") not in _stopwords and len(w) > 2]

        org_ids = current_org_ids.get([])
        # Run embedding + all keyword searches concurrently
        kw_coros = [self.graph.keyword_search(t.strip("?.,!'\""), k=k, org_ids=org_ids) for t in terms[:5]]
        if not self.embeddings:
            return {"query": query, "results": [], "count": 0}
        embed_and_kw: list[Any] = await asyncio.gather(self.embeddings.embed(query), *kw_coros)
        embedding: list[float] = embed_and_kw[0]
        kw_results_per_term: list[list[MemoryNode]] = embed_and_kw[1:]

        # 1. Vector search (needs embedding)
        vector_results = await self.graph.vector_search(embedding, k=k, org_ids=org_ids)

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
            vector_ids.add(nid)
            merged[nid] = {"node": node, "vec_score": score, "in_both": nid in kw_node_ids}

        # Add keyword-only results — compute their vector similarity for ranking
        for nid, node in kw_nodes_map.items():
            if nid not in merged:
                if node.embedding and embedding:
                    vec_score = cosine_similarity(node.embedding, embedding)
                else:
                    vec_score = 0.0
                merged[nid] = {"node": node, "vec_score": vec_score, "in_both": False}

        # 4. Rank by vector similarity, +0.1 boost for appearing in both
        memories = []
        for nid, info in merged.items():
            mem = await self._format_memory(info["node"], info["vec_score"])
            rank_score = info["vec_score"] + (0.1 if info["in_both"] else 0.0)
            mem["_rank_score"] = rank_score
            memories.append(mem)

            if self.event_bus and nid in vector_ids:
                await self.event_bus.publish("memory.accessed", {"node_id": nid})

        # Inject core memories not already in results
        core_nodes = await self.graph.get_nodes_by_status(MemoryStatus.CORE, limit=50)
        seen_ids = set(merged.keys())
        for cnode in core_nodes:
            if str(cnode.id) not in seen_ids:
                mem = await self._format_memory(cnode, 0.0)
                mem["is_core"] = True
                mem["_rank_score"] = 0.0
                memories.append(mem)

        # Deprioritize superseded nodes (replaced by newer information)
        SUPERSEDED_DECAY = 0.3
        for mem in memories:
            mem_id = mem.get("id")
            if mem_id:
                incoming = await self.graph.get_edges(mem_id, "incoming", EdgeType.SUPERSEDES)
                if incoming:
                    mem["_rank_score"] *= SUPERSEDED_DECAY
                    mem["superseded_by"] = str(
                        incoming[0].source_id if str(incoming[0].target_id) == mem_id else incoming[0].target_id
                    )

        memories.sort(key=lambda m: m["_rank_score"], reverse=True)
        for mem in memories:
            mem.pop("_rank_score", None)

        # Cap final results
        cap = max_results if max_results is not None else k
        memories = memories[:cap]

        # Update reactivation state for returned nodes (skip in read_only mode)
        if not read_only:
            now = datetime.now(timezone.utc)
            for mem in memories:
                mem_id = mem.get("id")
                if not mem_id:
                    continue
                try:
                    recalled = await self.graph.get_node(mem_id)
                    if recalled:
                        stability_delta = 0.1 / recalled.stability
                        await self.graph.atomic_reactivation_update(mem_id, now, stability_delta)
                        mem["reactivation_count"] = recalled.reactivation_count + 1
                except Exception:
                    logger.warning("Reactivation update failed for node %s", mem_id, exc_info=True)

        # Validate edges between co-retrieved nodes (skip in read_only mode)
        if not read_only and len(memories) > 1:
            recalled_ids = {m["id"] for m in memories if m.get("id")}
            try:
                all_edges = await self.graph.get_all_edges(list(recalled_ids))
                for edge in all_edges:
                    src, tgt = str(edge.source_id), str(edge.target_id)
                    if src in recalled_ids and tgt in recalled_ids:
                        await self.graph.validate_edge(str(edge.id))
            except Exception:
                logger.warning("Co-retrieval edge validation failed", exc_info=True)

        return {"query": query, "results": memories, "count": len(memories)}

    async def _format_memory(self, node: MemoryNode, score: float) -> dict[str, Any]:
        """Format a memory node with causal chain info."""
        causal_basis = []
        causal_chain = []
        org_ids = current_org_ids.get([])
        try:
            upstream = await self.graph.get_causal_chain(str(node.id), "upstream", org_ids=org_ids)
            downstream = await self.graph.get_causal_chain(str(node.id), "downstream", org_ids=org_ids)
            # Causal basis: both upstream causes and downstream effects
            seen = set()
            for n in upstream[:5]:
                if str(n.id) not in seen:
                    causal_basis.append({"id": str(n.id), "summary": n.content_summary, "direction": "upstream"})
                    seen.add(str(n.id))
            for n in downstream[:5]:
                if str(n.id) not in seen:
                    causal_basis.append({"id": str(n.id), "summary": n.content_summary, "direction": "downstream"})
                    seen.add(str(n.id))
            # Causal chain: structured array of the upstream path
            if upstream:
                for n in reversed(upstream[:5]):
                    causal_chain.append({"id": str(n.id), "summary": n.content_summary})
                causal_chain.append({"id": str(node.id), "summary": node.content_summary})
        except Exception:
            logger.warning("Causal chain formatting failed for node %s", node.id, exc_info=True)

        result = {
            "id": str(node.id),
            "content": node.content_full or node.content_summary,
            "summary": node.content_summary,
            "status": node.status.value,
            "decay_score": round(node.decay_score, 4),
            "score": round(score, 4),
            "created_at": node.created_at.isoformat(),
            "causal_basis": causal_basis,
            "is_core": node.status == MemoryStatus.CORE,
        }
        if causal_chain:
            result["causal_chain"] = causal_chain
        return result

    async def memory_search(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        k: int = 10,
    ) -> dict[str, Any]:
        """Filtered vector search by status/category/date/entity."""
        if not self.embeddings:
            return {"query": query, "results": [], "count": 0}
        embedding = await self.embeddings.embed(query)

        # Determine status filter
        status_filter = None
        if filters and "status" in filters:
            status_filter = [MemoryStatus(s) for s in filters["status"]]

        results = await self.graph.vector_search(embedding, k=k, status_filter=status_filter)

        # Apply additional filters post-query
        memories = []
        for node, score in results:
            if filters:
                if "category" in filters and node.category != filters["category"]:
                    continue
                if "entity" in filters and filters["entity"] not in node.entity_refs:
                    continue
                if "since" in filters:
                    since_dt = datetime.fromisoformat(filters["since"])
                    if node.created_at < since_dt:
                        continue

            memories.append({
                "id": str(node.id),
                "summary": node.content_summary,
                "status": node.status.value,
                "decay_score": round(node.decay_score, 4),
                "score": round(score, 4),
                "category": node.category,
                "created_at": node.created_at.isoformat(),
            })

        return {"query": query, "results": memories, "count": len(memories)}

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

        result_nodes = [
            {
                "id": str(n.id),
                "summary": n.content_summary,
                "status": n.status.value,
                "decay_score": round(n.decay_score, 4),
            }
            for n in nodes
        ]

        return {"start_node": node_id, "depth": depth, "nodes": result_nodes, "count": len(result_nodes)}

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

        return {
            "node_id": node_id,
            "summary": node.content_summary,
            "status": node.status.value,
            "decay_score": round(node.decay_score, 4),
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
                    "target": str(e.target_id) if str(e.source_id) == node_id else str(e.source_id),
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
