"""PostgreSQL + pgvector implementation of GraphStorageProvider."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import asyncpg

from genesys.context import current_user_id
from genesys.models.edge import MemoryEdge
from genesys.models.enums import CAUSAL_EDGE_TYPES, EdgeType, MemoryStatus, ReactivationPattern
from genesys.models.node import MemoryNode
from genesys.storage.db import get_pool


def _uid() -> str:
    uid = current_user_id.get(None)
    if uid is None:
        raise RuntimeError("No user context — current_user_id not set")
    return uid


def _row_to_node(row: asyncpg.Record) -> MemoryNode:
    return MemoryNode(
        id=row["id"],
        status=MemoryStatus(row["status"]),
        content_summary=row["content_summary"],
        content_full=row["content_full"],
        embedding=json.loads(row["embedding"]) if row["embedding"] else None,
        created_at=row["created_at"],
        last_accessed_at=row["last_accessed_at"] or row["created_at"],
        last_reactivated_at=row["last_reactivated_at"] or row["created_at"],
        decay_score=row["decay_score"],
        causal_weight=row["causal_weight"],
        reactivation_count=row["reactivation_count"],
        reactivation_pattern=ReactivationPattern(row["reactivation_pattern"]) if row["reactivation_pattern"] else ReactivationPattern.SINGLE,
        source_agent=row["source_agent"] or "claude",
        source_session=row["source_session"] or "",
        entity_refs=row["entity_refs"] or [],
        category=row["category"],
        pinned=row["pinned"],
        promotion_reason=row["promotion_reason"],
        reactivation_timestamps=list(row["reactivation_timestamps"]) if row.get("reactivation_timestamps") else [],
        stability=float(row["stability"]) if row.get("stability") is not None else 1.0,
    )


def _row_to_edge(row: asyncpg.Record) -> MemoryEdge:
    return MemoryEdge(
        id=row["id"],
        source_id=row["source_id"],
        target_id=row["target_id"],
        type=EdgeType(row["type"]),
        weight=row["weight"],
        created_at=row["created_at"],
    )


def _embedding_literal(embedding: list[float]) -> str:
    """Format embedding as pgvector literal string."""
    return "[" + ",".join(str(f) for f in embedding) + "]"


async def get_all_user_ids() -> list[str]:
    """Return all distinct user_ids that have memory nodes."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT DISTINCT user_id FROM memory_nodes")
    return [r["user_id"] for r in rows]


class PostgresGraphProvider:
    """GraphStorageProvider backed by PostgreSQL + pgvector."""

    async def initialize(self, user_id: str) -> None:
        # Pool is lazily created; just ensure it's ready
        await get_pool()

    async def destroy(self, user_id: str) -> None:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM memory_edges WHERE user_id = $1", user_id)
            await conn.execute("DELETE FROM memory_nodes WHERE user_id = $1", user_id)

    async def create_node(self, node: MemoryNode) -> str:
        uid = _uid()
        pool = await get_pool()
        embedding_val = _embedding_literal(node.embedding) if node.embedding else None
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO memory_nodes
                   (id, user_id, status, content_summary, content_full, embedding,
                    category, entity_refs, decay_score, causal_weight,
                    reactivation_count, reactivation_pattern, pinned, promotion_reason,
                    source_agent, source_session, created_at, last_accessed_at,
                    last_reactivated_at, metadata, reactivation_timestamps)
                   VALUES ($1,$2,$3,$4,$5,$6::vector,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21)""",
                node.id, uid, node.status.value, node.content_summary, node.content_full,
                embedding_val,
                node.category, node.entity_refs, node.decay_score, node.causal_weight,
                node.reactivation_count, node.reactivation_pattern.value,
                node.pinned, node.promotion_reason,
                node.source_agent, node.source_session,
                node.created_at, node.last_accessed_at, node.last_reactivated_at,
                json.dumps({}),
                node.reactivation_timestamps or [node.created_at],
            )
        return str(node.id)

    async def get_node(self, node_id: str) -> MemoryNode | None:
        uid = _uid()
        pool = await get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT *, embedding::text as embedding FROM memory_nodes WHERE id = $1 AND user_id = $2",
                uuid.UUID(node_id), uid,
            )
        return _row_to_node(row) if row else None

    _ALLOWED_UPDATE_COLUMNS = frozenset({
        "status", "content_summary", "content_full", "embedding", "category",
        "entity_refs", "decay_score", "causal_weight", "reactivation_count",
        "reactivation_pattern", "pinned", "promotion_reason", "source_agent",
        "source_session", "last_accessed_at", "last_reactivated_at", "metadata",
        "reactivation_timestamps", "stability", "irrelevance_counter",
    })

    async def update_node(self, node_id: str, updates: dict) -> None:
        uid = _uid()
        if not updates:
            return
        pool = await get_pool()

        set_parts = []
        args: list = []
        idx = 1

        for key, val in updates.items():
            if key not in self._ALLOWED_UPDATE_COLUMNS:
                raise ValueError(f"Invalid column for update: {key}")
            if key == "embedding":
                if val is not None:
                    set_parts.append(f"embedding = ${idx}::vector")
                    args.append(_embedding_literal(val))
                else:
                    set_parts.append(f"embedding = NULL")
                idx += 1
                continue
            if key == "status" and isinstance(val, MemoryStatus):
                val = val.value
            if key == "reactivation_pattern" and isinstance(val, ReactivationPattern):
                val = val.value
            set_parts.append(f"{key} = ${idx}")
            args.append(val)
            idx += 1

        args.append(uuid.UUID(node_id))
        args.append(uid)
        query = f"UPDATE memory_nodes SET {', '.join(set_parts)} WHERE id = ${idx} AND user_id = ${idx + 1}"

        async with pool.acquire() as conn:
            await conn.execute(query, *args)

    async def delete_node(self, node_id: str) -> None:
        uid = _uid()
        pool = await get_pool()
        nid = uuid.UUID(node_id)
        async with pool.acquire() as conn:
            # Edges cascade on delete
            await conn.execute("DELETE FROM memory_nodes WHERE id = $1 AND user_id = $2", nid, uid)

    async def get_nodes_by_status(self, status: MemoryStatus, limit: int = 100) -> list[MemoryNode]:
        uid = _uid()
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT *, embedding::text as embedding FROM memory_nodes WHERE user_id = $1 AND status = $2 ORDER BY created_at DESC LIMIT $3",
                uid, status.value, limit,
            )
        return [_row_to_node(r) for r in rows]

    async def create_edge(self, edge: MemoryEdge) -> str:
        uid = _uid()
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO memory_edges (id, user_id, source_id, target_id, type, weight, created_at)
                   VALUES ($1,$2,$3,$4,$5,$6,$7)
                   ON CONFLICT (source_id, target_id, type) DO UPDATE SET weight = $6""",
                edge.id, uid, edge.source_id, edge.target_id, edge.type.value, edge.weight, edge.created_at,
            )
        return str(edge.id)

    async def get_edges(self, node_id: str, direction: str, edge_type: EdgeType | None = None) -> list[MemoryEdge]:
        uid = _uid()
        pool = await get_pool()
        nid = uuid.UUID(node_id)
        conditions = ["user_id = $1"]
        args: list = [uid]
        idx = 2

        if direction == "outgoing":
            conditions.append(f"source_id = ${idx}")
            args.append(nid)
            idx += 1
        elif direction == "incoming":
            conditions.append(f"target_id = ${idx}")
            args.append(nid)
            idx += 1
        else:  # both
            conditions.append(f"(source_id = ${idx} OR target_id = ${idx})")
            args.append(nid)
            idx += 1

        if edge_type:
            conditions.append(f"type = ${idx}")
            args.append(edge_type.value)

        query = f"SELECT * FROM memory_edges WHERE {' AND '.join(conditions)}"
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *args)
        return [_row_to_edge(r) for r in rows]

    async def get_all_edges(self, node_ids: list[str] | None = None) -> list[MemoryEdge]:
        uid = _uid()
        pool = await get_pool()
        async with pool.acquire() as conn:
            if node_ids:
                ids = [uuid.UUID(nid) for nid in node_ids]
                rows = await conn.fetch(
                    "SELECT * FROM memory_edges WHERE user_id = $1 AND (source_id = ANY($2) OR target_id = ANY($2))",
                    uid, ids,
                )
            else:
                rows = await conn.fetch("SELECT * FROM memory_edges WHERE user_id = $1", uid)
        return [_row_to_edge(r) for r in rows]

    async def update_edge_weight(self, edge_id: str, weight: float) -> None:
        uid = _uid()
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE memory_edges SET weight = $1 WHERE id = $2 AND user_id = $3",
                weight, uuid.UUID(edge_id), uid,
            )

    async def delete_edge(self, edge_id: str) -> None:
        uid = _uid()
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM memory_edges WHERE id = $1 AND user_id = $2", uuid.UUID(edge_id), uid)

    async def edge_exists(self, source_id: str, target_id: str, edge_type: EdgeType) -> bool:
        uid = _uid()
        pool = await get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT 1 FROM memory_edges WHERE user_id = $1 AND source_id = $2 AND target_id = $3 AND type = $4",
                uid, uuid.UUID(source_id), uuid.UUID(target_id), edge_type.value,
            )
        return row is not None

    async def traverse(self, start_id: str, depth: int, edge_types: list[EdgeType] | None = None) -> list[MemoryNode]:
        uid = _uid()
        pool = await get_pool()
        sid = uuid.UUID(start_id)

        type_filter = ""
        args: list = [uid, sid, depth]
        if edge_types:
            placeholders = ", ".join(f"${i + 4}" for i in range(len(edge_types)))
            type_filter = f"AND e.type IN ({placeholders})"
            args.extend(et.value for et in edge_types)

        query = f"""
            WITH RECURSIVE graph AS (
                SELECT n.id, 0 AS d
                FROM memory_nodes n
                WHERE n.id = $2 AND n.user_id = $1
                UNION
                SELECT CASE WHEN e.source_id = g.id THEN e.target_id ELSE e.source_id END, g.d + 1
                FROM graph g
                JOIN memory_edges e ON (e.source_id = g.id OR e.target_id = g.id) AND e.user_id = $1 {type_filter}
                WHERE g.d < $3
            )
            SELECT DISTINCT mn.*, mn.embedding::text as embedding
            FROM graph g
            JOIN memory_nodes mn ON mn.id = g.id AND mn.user_id = $1
            WHERE mn.id != $2
        """
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *args)
        return [_row_to_node(r) for r in rows]

    async def get_causal_chain(self, node_id: str, direction: str) -> list[MemoryNode]:
        uid = _uid()
        pool = await get_pool()
        nid = uuid.UUID(node_id)
        causal_types = [et.value for et in CAUSAL_EDGE_TYPES]
        placeholders = ", ".join(f"${i + 3}" for i in range(len(causal_types)))

        if direction == "upstream":
            join_cond = "e.target_id = c.id"
            select_col = "e.source_id"
        else:
            join_cond = "e.source_id = c.id"
            select_col = "e.target_id"

        query = f"""
            WITH RECURSIVE chain AS (
                SELECT $2::uuid AS id, 0 AS d
                UNION
                SELECT {select_col}, c.d + 1
                FROM chain c
                JOIN memory_edges e ON {join_cond} AND e.user_id = $1 AND e.type IN ({placeholders})
                WHERE c.d < 10
            )
            SELECT DISTINCT mn.*, mn.embedding::text as embedding
            FROM chain c
            JOIN memory_nodes mn ON mn.id = c.id AND mn.user_id = $1
            WHERE mn.id != $2
        """
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, uid, nid, *causal_types)
        return [_row_to_node(r) for r in rows]

    async def get_causal_weight(self, node_id: str) -> int:
        uid = _uid()
        pool = await get_pool()
        nid = uuid.UUID(node_id)
        causal_types = [et.value for et in CAUSAL_EDGE_TYPES]
        placeholders = ", ".join(f"${i + 3}" for i in range(len(causal_types)))
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT count(*) as cnt FROM memory_edges WHERE user_id = $1 AND (source_id = $2 OR target_id = $2) AND type IN ({placeholders})",
                uid, nid, *causal_types,
            )
        return row["cnt"] if row else 0

    async def get_degree(self, node_id: str) -> int:
        uid = _uid()
        pool = await get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT count(*) as cnt FROM memory_edges WHERE user_id = $1 AND (source_id = $2 OR target_id = $2)",
                uid, uuid.UUID(node_id),
            )
        return row["cnt"] if row else 0

    async def is_orphan(self, node_id: str) -> bool:
        return (await self.get_degree(node_id)) == 0

    async def get_orphans(self) -> list[MemoryNode]:
        uid = _uid()
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT mn.*, mn.embedding::text as embedding FROM memory_nodes mn
                   WHERE mn.user_id = $1
                     AND NOT EXISTS (
                       SELECT 1 FROM memory_edges me
                       WHERE me.user_id = $1 AND (me.source_id = mn.id OR me.target_id = mn.id)
                     )""",
                uid,
            )
        return [_row_to_node(r) for r in rows]

    async def vector_search(
        self, embedding: list[float], k: int = 10,
        status_filter: list[MemoryStatus] | None = None,
    ) -> list[tuple[MemoryNode, float]]:
        uid = _uid()
        pool = await get_pool()
        emb_lit = _embedding_literal(embedding)
        k = int(k)  # Enforce integer to prevent injection

        conditions = ["user_id = $1", "embedding IS NOT NULL"]
        args: list = [uid]
        idx = 2

        if status_filter:
            placeholders = ", ".join(f"${i + idx}" for i in range(len(status_filter)))
            conditions.append(f"status IN ({placeholders})")
            args.extend(s.value for s in status_filter)
            idx += len(status_filter)

        # Parameterize embedding and limit
        args.append(emb_lit)
        emb_param = f"${idx}"
        idx += 1
        args.append(k)
        limit_param = f"${idx}"

        where = " AND ".join(conditions)
        query = f"""
            SELECT *, embedding::text as embedding,
                   1 - (embedding <=> {emb_param}::vector) AS similarity
            FROM memory_nodes
            WHERE {where}
            ORDER BY embedding <=> {emb_param}::vector
            LIMIT {limit_param}
        """
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *args)

        return [((_row_to_node(r)), r["similarity"]) for r in rows]

    async def keyword_search(self, query: str, entity_refs: list[str] | None = None, k: int = 10) -> list[MemoryNode]:
        uid = _uid()
        pool = await get_pool()
        k = int(k)  # Enforce integer
        conditions = ["user_id = $1"]
        args: list = [uid]
        idx = 2

        if query:
            conditions.append(f"(content_summary ILIKE ${idx} OR content_full ILIKE ${idx})")
            args.append(f"%{query}%")
            idx += 1

        if entity_refs:
            conditions.append(f"entity_refs && ${idx}")
            args.append(entity_refs)
            idx += 1

        args.append(k)
        limit_param = f"${idx}"

        where = " AND ".join(conditions)
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT *, embedding::text as embedding FROM memory_nodes WHERE {where} LIMIT {limit_param}",
                *args,
            )
        return [_row_to_node(r) for r in rows]

    async def get_stats(self) -> dict:
        uid = _uid()
        pool = await get_pool()
        async with pool.acquire() as conn:
            node_rows = await conn.fetch(
                "SELECT status, count(*) as cnt FROM memory_nodes WHERE user_id = $1 GROUP BY status", uid
            )
            edge_rows = await conn.fetch(
                "SELECT type, count(*) as cnt FROM memory_edges WHERE user_id = $1 GROUP BY type", uid
            )
            orphan_row = await conn.fetchrow(
                """SELECT count(*) as cnt FROM memory_nodes mn
                   WHERE mn.user_id = $1
                     AND NOT EXISTS (
                       SELECT 1 FROM memory_edges me
                       WHERE me.user_id = $1 AND (me.source_id = mn.id OR me.target_id = mn.id)
                     )""",
                uid,
            )
            causal_types = [et.value for et in CAUSAL_EDGE_TYPES]
            placeholders = ", ".join(f"${i + 2}" for i in range(len(causal_types)))
            max_cw_row = await conn.fetchrow(
                f"""SELECT COALESCE(MAX(cnt), 0) as mcw FROM (
                    SELECT count(*) as cnt FROM memory_edges
                    WHERE user_id = $1 AND type IN ({placeholders})
                    GROUP BY source_id
                ) sub""",
                uid, *causal_types,
            )
        nodes_by_status = {r["status"]: r["cnt"] for r in node_rows}
        edges_by_type = {r["type"]: r["cnt"] for r in edge_rows}
        total_nodes = sum(nodes_by_status.values())
        total_edges = sum(edges_by_type.values())
        return {
            "total_nodes": total_nodes,
            "nodes_by_status": nodes_by_status,
            "total_edges": total_edges,
            "edges_by_type": edges_by_type,
            "orphan_count": orphan_row["cnt"] if orphan_row else 0,
            "max_causal_weight": max_cw_row["mcw"] if max_cw_row else 0,
        }
