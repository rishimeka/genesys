from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import falkordb as fdb

from genesys.models.edge import MemoryEdge
from genesys.models.enums import CAUSAL_EDGE_TYPES, EdgeType, MemoryStatus
from genesys.models.node import MemoryNode


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _parse_dt(val: str | None) -> datetime:
    if val is None:
        return datetime.now(timezone.utc)
    return datetime.fromisoformat(val)


def _node_to_params(node: MemoryNode) -> dict:
    return {
        "id": str(node.id),
        "status": node.status.value,
        "content_summary": node.content_summary,
        "content_full": node.content_full or "",
        "content_ref": node.content_ref or "",
        "created_at": _iso(node.created_at),
        "last_accessed_at": _iso(node.last_accessed_at),
        "last_reactivated_at": _iso(node.last_reactivated_at),
        "decay_score": node.decay_score,
        "causal_weight": node.causal_weight,
        "reactivation_count": node.reactivation_count,
        "reactivation_pattern": node.reactivation_pattern.value,
        "irrelevance_counter": node.irrelevance_counter,
        "source_agent": node.source_agent,
        "source_session": node.source_session,
        "entity_refs": json.dumps(node.entity_refs),
        "category": node.category or "",
        "pinned": node.pinned,
        "promotion_reason": node.promotion_reason or "",
        "reactivation_timestamps": json.dumps([_iso(t) for t in node.reactivation_timestamps]),
        "stability": node.stability,
    }


def _record_to_node(props: dict) -> MemoryNode:
    return MemoryNode(
        id=uuid.UUID(props["id"]),
        status=MemoryStatus(props["status"]),
        content_summary=props["content_summary"],
        content_full=props.get("content_full") or None,
        content_ref=props.get("content_ref") or None,
        embedding=None,  # not returned from queries by default
        created_at=_parse_dt(props.get("created_at")),
        last_accessed_at=_parse_dt(props.get("last_accessed_at")),
        last_reactivated_at=_parse_dt(props.get("last_reactivated_at")),
        decay_score=float(props.get("decay_score", 1.0)),
        causal_weight=int(props.get("causal_weight", 0)),
        reactivation_count=int(props.get("reactivation_count", 0)),
        reactivation_pattern=props.get("reactivation_pattern", "single"),
        irrelevance_counter=int(props.get("irrelevance_counter", 0)),
        source_agent=props.get("source_agent", ""),
        source_session=props.get("source_session", ""),
        entity_refs=json.loads(props.get("entity_refs", "[]")),
        category=props.get("category") or None,
        pinned=bool(props.get("pinned", False)),
        promotion_reason=props.get("promotion_reason") or None,
        reactivation_timestamps=[
            _parse_dt(t) for t in json.loads(props.get("reactivation_timestamps", "[]"))
        ],
        stability=float(props.get("stability", 1.0)),
    )


def _result_to_props(record, alias: str = "m") -> dict:
    """Extract properties from a FalkorDB result record."""
    node = record[0] if isinstance(record, (list, tuple)) else record
    return node.properties


class FalkorDBProvider:
    def __init__(self, host: str = "localhost", port: int = 6379):
        self._host = host
        self._port = port
        self._db: fdb.FalkorDB | None = None
        self._graph = None
        self._user_id: str | None = None

    def _get_db(self) -> fdb.FalkorDB:
        if self._db is None:
            self._db = fdb.FalkorDB(host=self._host, port=self._port)
        return self._db

    def _graph_name(self, user_id: str) -> str:
        return f"genesys_user_{user_id}"

    async def initialize(self, user_id: str) -> None:
        self._user_id = user_id
        db = self._get_db()
        self._graph = db.select_graph(self._graph_name(user_id))
        # Create vector index if not exists
        try:
            self._graph.query(
                "CREATE VECTOR INDEX FOR (m:Memory) ON (m.embedding) OPTIONS "
                "{dimension: 1536, similarityFunction: 'cosine'}"
            )
        except Exception:
            pass  # index already exists

    async def destroy(self, user_id: str) -> None:
        db = self._get_db()
        try:
            g = db.select_graph(self._graph_name(user_id))
            g.delete()
        except Exception:
            pass

    async def create_node(self, node: MemoryNode) -> str:
        params = _node_to_params(node)
        # Build embedding parameter
        embedding_clause = ""
        if node.embedding:
            params["embedding"] = node.embedding
            embedding_clause = ", embedding: vecf32($embedding)"

        query = (
            "CREATE (m:Memory {"
            "id: $id, status: $status, content_summary: $content_summary, "
            "content_full: $content_full, content_ref: $content_ref, "
            "created_at: $created_at, last_accessed_at: $last_accessed_at, "
            "last_reactivated_at: $last_reactivated_at, "
            "decay_score: $decay_score, causal_weight: $causal_weight, "
            "reactivation_count: $reactivation_count, reactivation_pattern: $reactivation_pattern, "
            "irrelevance_counter: $irrelevance_counter, "
            "source_agent: $source_agent, source_session: $source_session, "
            "entity_refs: $entity_refs, category: $category, "
            "pinned: $pinned, promotion_reason: $promotion_reason, "
            f"reactivation_timestamps: $reactivation_timestamps, stability: $stability{embedding_clause}"
            "}) RETURN m.id"
        )
        self._graph.query(query, params)
        return str(node.id)

    async def get_node(self, node_id: str) -> MemoryNode | None:
        result = self._graph.query(
            "MATCH (m:Memory {id: $id}) RETURN m",
            {"id": node_id},
        )
        if not result.result_set:
            return None
        props = result.result_set[0][0].properties
        return _record_to_node(props)

    async def update_node(self, node_id: str, updates: dict) -> None:
        if not updates:
            return
        set_clauses = []
        params = {"id": node_id}
        for key, val in updates.items():
            param_name = f"u_{key}"
            if isinstance(val, datetime):
                params[param_name] = _iso(val)
            elif isinstance(val, list):
                params[param_name] = json.dumps(val) if key in ("entity_refs", "reactivation_timestamps") else val
            elif isinstance(val, MemoryStatus):
                params[param_name] = val.value
            else:
                params[param_name] = val
            set_clauses.append(f"m.{key} = ${param_name}")

        query = f"MATCH (m:Memory {{id: $id}}) SET {', '.join(set_clauses)}"
        self._graph.query(query, params)

    async def delete_node(self, node_id: str) -> None:
        self._graph.query(
            "MATCH (m:Memory {id: $id}) DETACH DELETE m",
            {"id": node_id},
        )

    async def get_nodes_by_status(self, status: MemoryStatus, limit: int = 100) -> list[MemoryNode]:
        result = self._graph.query(
            "MATCH (m:Memory {status: $status}) RETURN m LIMIT $limit",
            {"status": status.value, "limit": limit},
        )
        return [_record_to_node(r[0].properties) for r in result.result_set]

    async def create_edge(self, edge: MemoryEdge) -> str:
        rel_type = edge.type.value.upper()
        params = {
            "source_id": str(edge.source_id),
            "target_id": str(edge.target_id),
            "edge_id": str(edge.id),
            "weight": edge.weight,
            "created_at": _iso(edge.created_at),
            "metadata": json.dumps(edge.metadata) if edge.metadata else "{}",
        }
        query = (
            f"MATCH (s:Memory {{id: $source_id}}), (t:Memory {{id: $target_id}}) "
            f"CREATE (s)-[r:{rel_type} {{id: $edge_id, weight: $weight, "
            f"created_at: $created_at, metadata: $metadata}}]->(t) RETURN r"
        )
        self._graph.query(query, params)
        return str(edge.id)

    async def get_edges(self, node_id: str, direction: str, edge_type: EdgeType | None = None) -> list[MemoryEdge]:
        if direction == "out":
            pattern = "(m:Memory {id: $id})-[r]->(n:Memory)"
        elif direction == "in":
            pattern = "(n:Memory)-[r]->(m:Memory {id: $id})"
        else:
            pattern = "(m:Memory {id: $id})-[r]-(n:Memory)"

        # FalkorDB doesn't support variable rel type filter easily, use post-filter
        query = f"MATCH {pattern} RETURN r, m.id, n.id, type(r)"
        result = self._graph.query(query, {"id": node_id})

        edges = []
        for record in result.result_set:
            rel = record[0]
            rel_type_str = record[3]
            if edge_type and rel_type_str.lower() != edge_type.value:
                continue
            props = rel.properties
            # Determine source/target based on direction
            if direction == "in":
                source_id_str = str(record[2]) if isinstance(record[2], str) else node_id
                target_id_str = node_id
            else:
                source_id_str = node_id
                target_id_str = str(record[2]) if isinstance(record[2], str) else node_id

            edges.append(MemoryEdge(
                id=uuid.UUID(props["id"]),
                source_id=uuid.UUID(source_id_str) if source_id_str != node_id else uuid.UUID(node_id),
                target_id=uuid.UUID(target_id_str) if target_id_str != node_id else uuid.UUID(node_id),
                type=EdgeType(rel_type_str.lower()),
                weight=float(props.get("weight", 0.7)),
                created_at=_parse_dt(props.get("created_at")),
                metadata=json.loads(props.get("metadata", "{}")) or None,
            ))
        return edges

    async def get_all_edges(self, node_ids: list[str] | None = None) -> list[MemoryEdge]:
        if node_ids:
            result = self._graph.query(
                "MATCH (a)-[r]->(b) WHERE a.id IN $ids OR b.id IN $ids RETURN r",
                {"ids": node_ids},
            )
        else:
            result = self._graph.query("MATCH ()-[r]->() RETURN r")
        return [self._parse_edge(r[0]) for r in result.result_set]

    async def update_edge_weight(self, edge_id: str, weight: float) -> None:
        self._graph.query(
            "MATCH ()-[r {id: $id}]->() SET r.weight = $weight",
            {"id": edge_id, "weight": weight},
        )

    async def delete_edge(self, edge_id: str) -> None:
        self._graph.query(
            "MATCH ()-[r {id: $id}]->() DELETE r",
            {"id": edge_id},
        )

    async def edge_exists(self, source_id: str, target_id: str, edge_type: EdgeType) -> bool:
        rel_type = edge_type.value.upper()
        result = self._graph.query(
            f"MATCH (s:Memory {{id: $sid}})-[r:{rel_type}]->(t:Memory {{id: $tid}}) RETURN count(r)",
            {"sid": source_id, "tid": target_id},
        )
        return result.result_set[0][0] > 0

    async def traverse(self, start_id: str, depth: int, edge_types: list[EdgeType] | None = None) -> list[MemoryNode]:
        if edge_types:
            rel_types = "|".join(e.value.upper() for e in edge_types)
            pattern = f"[:{ rel_types }*1..{depth}]"
        else:
            pattern = f"[*1..{depth}]"

        result = self._graph.query(
            f"MATCH (s:Memory {{id: $id}})-{pattern}-(n:Memory) WHERE n.id <> $id RETURN DISTINCT n",
            {"id": start_id},
        )
        return [_record_to_node(r[0].properties) for r in result.result_set]

    async def get_causal_chain(self, node_id: str, direction: str) -> list[MemoryNode]:
        causal_types = "|".join(e.value.upper() for e in CAUSAL_EDGE_TYPES)
        if direction == "upstream":
            # What caused this node: follow outgoing CAUSED_BY edges (this node -> cause)
            pattern = f"(s:Memory {{id: $id}})-[:{causal_types}*1..10]->(n:Memory)"
        else:
            # What this node caused: follow incoming CAUSED_BY edges (dependent -> this node)
            pattern = f"(n:Memory)-[:{causal_types}*1..10]->(s:Memory {{id: $id}})"

        result = self._graph.query(
            f"MATCH {pattern} WHERE n.id <> $id RETURN DISTINCT n",
            {"id": node_id},
        )
        return [_record_to_node(r[0].properties) for r in result.result_set]

    async def get_causal_weight(self, node_id: str) -> int:
        causal_types = "|".join(e.value.upper() for e in CAUSAL_EDGE_TYPES)
        result = self._graph.query(
            f"MATCH (dep:Memory)-[:{causal_types}*1..10]->(a:Memory {{id: $id}}) "
            "RETURN count(DISTINCT dep)",
            {"id": node_id},
        )
        return int(result.result_set[0][0]) if result.result_set else 0

    async def get_degree(self, node_id: str) -> int:
        result = self._graph.query(
            "MATCH (m:Memory {id: $id})-[r]-() RETURN count(r)",
            {"id": node_id},
        )
        return int(result.result_set[0][0]) if result.result_set else 0

    async def is_orphan(self, node_id: str) -> bool:
        return (await self.get_degree(node_id)) == 0

    async def get_orphans(self) -> list[MemoryNode]:
        result = self._graph.query(
            "MATCH (m:Memory) WHERE NOT (m)-[]-() AND m.status <> 'core' AND m.pinned = false RETURN m"
        )
        return [_record_to_node(r[0].properties) for r in result.result_set]

    async def vector_search(
        self,
        embedding: list[float],
        k: int = 10,
        status_filter: list[MemoryStatus] | None = None,
    ) -> list[tuple[MemoryNode, float]]:
        params = {"vec": embedding, "k": k}
        query = "CALL db.idx.vector.queryNodes('Memory', 'embedding', $k, vecf32($vec)) YIELD node, score"

        if status_filter:
            statuses = [s.value for s in status_filter]
            params["statuses"] = statuses
            # Always exclude pruned
            query += " WHERE node.status IN $statuses"
        else:
            query += " WHERE node.status <> 'pruned'"

        query += " RETURN node, score ORDER BY score ASC"

        result = self._graph.query(query, params)
        results = []
        for record in result.result_set:
            node = _record_to_node(record[0].properties)
            distance = float(record[1])
            # Convert cosine distance to similarity (consistent with Postgres provider)
            similarity = max(0.0, 1.0 - distance)
            results.append((node, similarity))
        return results

    async def keyword_search(self, query: str, entity_refs: list[str] | None = None, k: int = 10) -> list[MemoryNode]:
        params: dict = {"k": k}
        if entity_refs:
            # Match against entity_refs JSON string
            conditions = []
            for i, ref in enumerate(entity_refs):
                pname = f"ref_{i}"
                params[pname] = ref.lower()
                conditions.append(f"toLower(m.entity_refs) CONTAINS ${pname}")
            where = " OR ".join(conditions)
        else:
            params["q"] = query.lower()
            where = "toLower(m.content_summary) CONTAINS $q OR toLower(m.entity_refs) CONTAINS $q"

        cypher = f"MATCH (m:Memory) WHERE ({where}) AND m.status <> 'pruned' RETURN m LIMIT $k"
        result = self._graph.query(cypher, params)
        return [_record_to_node(r[0].properties) for r in result.result_set]

    async def get_stats(self) -> dict:
        stats: dict = {}
        r = self._graph.query("MATCH (m:Memory) RETURN count(m)")
        stats["node_count"] = r.result_set[0][0] if r.result_set else 0

        r = self._graph.query("MATCH (m:Memory) RETURN m.status, count(m)")
        stats["node_count_by_status"] = {row[0]: row[1] for row in r.result_set}

        r = self._graph.query("MATCH ()-[r]->() RETURN count(r)")
        stats["edge_count"] = r.result_set[0][0] if r.result_set else 0

        r = self._graph.query("MATCH ()-[r]->() RETURN type(r), count(r)")
        stats["edge_count_by_type"] = {row[0]: row[1] for row in r.result_set}

        r = self._graph.query(
            "MATCH (m:Memory) WHERE NOT (m)-[]-() AND m.status <> 'core' AND m.pinned = false RETURN count(m)"
        )
        stats["orphan_count"] = r.result_set[0][0] if r.result_set else 0

        stats["avg_causal_weight"] = 0.0
        stats["max_causal_weight"] = 0
        return stats
