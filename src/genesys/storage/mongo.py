"""MongoDB implementation of GraphStorageProvider."""
from __future__ import annotations

import os
import uuid
from collections import deque
from datetime import datetime, timezone

import numpy as np

from genesys.context import current_user_id
from genesys.models.edge import MemoryEdge
from genesys.models.enums import CAUSAL_EDGE_TYPES, EdgeType, MemoryStatus, ReactivationPattern
from genesys.models.node import MemoryNode


def _uid() -> str:
    uid = current_user_id.get(None)
    if uid is None:
        raise RuntimeError("No user context — current_user_id not set")
    return uid


def _embedding_field() -> str:
    return "embedding_384" if os.getenv("GENESYS_EMBEDDER") == "local" else "embedding_1536"


def _doc_to_node(doc: dict) -> MemoryNode:
    embedding = doc.get("embedding_1536") or doc.get("embedding_384")
    return MemoryNode(
        id=uuid.UUID(doc["_id"]),
        status=MemoryStatus(doc["status"]),
        content_summary=doc["content_summary"],
        content_full=doc.get("content_full"),
        embedding=embedding,
        created_at=doc.get("created_at", datetime.now(timezone.utc)),
        last_accessed_at=doc.get("last_accessed_at") or doc.get("created_at", datetime.now(timezone.utc)),
        last_reactivated_at=doc.get("last_reactivated_at") or doc.get("created_at", datetime.now(timezone.utc)),
        decay_score=doc.get("decay_score", 1.0),
        causal_weight=doc.get("causal_weight", 0),
        reactivation_count=doc.get("reactivation_count", 0),
        reactivation_pattern=ReactivationPattern(doc["reactivation_pattern"]) if doc.get("reactivation_pattern") else ReactivationPattern.SINGLE,
        source_agent=doc.get("source_agent", "claude"),
        source_session=doc.get("source_session", ""),
        entity_refs=doc.get("entity_refs", []),
        category=doc.get("category"),
        pinned=doc.get("pinned", False),
        promotion_reason=doc.get("promotion_reason"),
        reactivation_timestamps=doc.get("reactivation_timestamps", []),
        stability=float(doc.get("stability", 1.0)),
        irrelevance_counter=doc.get("irrelevance_counter", 0),
    )


def _doc_to_edge(doc: dict) -> MemoryEdge:
    return MemoryEdge(
        id=uuid.UUID(doc["_id"]),
        source_id=uuid.UUID(doc["source_id"]),
        target_id=uuid.UUID(doc["target_id"]),
        type=EdgeType(doc["type"]),
        weight=doc.get("weight", 0.7),
        created_at=doc.get("created_at", datetime.now(timezone.utc)),
    )


def _node_to_doc(node: MemoryNode, user_id: str) -> dict:
    field = _embedding_field()
    doc = {
        "_id": str(node.id),
        "user_id": user_id,
        "status": node.status.value,
        "content_summary": node.content_summary,
        "content_full": node.content_full,
        "category": node.category,
        "entity_refs": node.entity_refs,
        "decay_score": node.decay_score,
        "causal_weight": node.causal_weight,
        "reactivation_count": node.reactivation_count,
        "reactivation_pattern": node.reactivation_pattern.value,
        "pinned": node.pinned,
        "promotion_reason": node.promotion_reason,
        "source_agent": node.source_agent,
        "source_session": node.source_session,
        "created_at": node.created_at,
        "last_accessed_at": node.last_accessed_at,
        "last_reactivated_at": node.last_reactivated_at,
        "reactivation_timestamps": node.reactivation_timestamps or [node.created_at],
        "stability": node.stability,
        "irrelevance_counter": node.irrelevance_counter,
    }
    if node.embedding:
        doc[field] = node.embedding
    return doc


def _edge_to_doc(edge: MemoryEdge, user_id: str) -> dict:
    return {
        "_id": str(edge.id),
        "user_id": user_id,
        "source_id": str(edge.source_id),
        "target_id": str(edge.target_id),
        "type": edge.type.value,
        "weight": edge.weight,
        "created_at": edge.created_at,
    }


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr, b_arr = np.array(a), np.array(b)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if denom == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)


class MongoGraphProvider:
    """GraphStorageProvider backed by MongoDB (supports local and Atlas)."""

    def __init__(self, connection_string: str = "mongodb://localhost:27017", database: str = "genesys"):
        from motor.motor_asyncio import AsyncIOMotorClient
        self._client = AsyncIOMotorClient(connection_string)
        self._db = self._client[database]
        self._nodes = self._db["memory_nodes"]
        self._edges = self._db["memory_edges"]
        self._vector_search_available = False

    async def initialize(self, user_id: str) -> None:
        # Node indexes
        await self._nodes.create_index([("user_id", 1), ("status", 1)])
        await self._nodes.create_index([("user_id", 1)])
        await self._nodes.create_index(
            [("content_summary", "text"), ("content_full", "text")],
            default_language="english",
        )

        # Edge indexes
        await self._edges.create_index([("user_id", 1), ("source_id", 1)])
        await self._edges.create_index([("user_id", 1), ("target_id", 1)])
        await self._edges.create_index(
            [("user_id", 1), ("source_id", 1), ("target_id", 1), ("type", 1)],
            unique=True,
        )

        # Try to create Atlas vector search indexes
        try:
            for field, dim in [("embedding_1536", 1536), ("embedding_384", 384)]:
                await self._db.command({
                    "createSearchIndexes": "memory_nodes",
                    "indexes": [{
                        "name": f"vector_{field}",
                        "type": "vectorSearch",
                        "definition": {
                            "fields": [{
                                "type": "vector",
                                "path": field,
                                "numDimensions": dim,
                                "similarity": "cosine",
                            }]
                        },
                    }],
                })
            self._vector_search_available = True
        except Exception:
            # Local MongoDB without Atlas Search — use Python fallback
            self._vector_search_available = False

    async def destroy(self, user_id: str) -> None:
        await self._edges.delete_many({"user_id": user_id})
        await self._nodes.delete_many({"user_id": user_id})

    async def create_node(self, node: MemoryNode) -> str:
        uid = _uid()
        doc = _node_to_doc(node, uid)
        await self._nodes.insert_one(doc)
        return str(node.id)

    async def get_node(self, node_id: str) -> MemoryNode | None:
        uid = _uid()
        doc = await self._nodes.find_one({"_id": node_id, "user_id": uid})
        return _doc_to_node(doc) if doc else None

    async def update_node(self, node_id: str, updates: dict) -> None:
        uid = _uid()
        if not updates:
            return
        mongo_updates: dict = {}
        for key, val in updates.items():
            if key == "embedding":
                field = _embedding_field()
                mongo_updates[field] = val
            elif key == "status" and isinstance(val, MemoryStatus):
                mongo_updates["status"] = val.value
            elif key == "reactivation_pattern" and isinstance(val, ReactivationPattern):
                mongo_updates["reactivation_pattern"] = val.value
            else:
                mongo_updates[key] = val
        await self._nodes.update_one(
            {"_id": node_id, "user_id": uid},
            {"$set": mongo_updates},
        )

    async def delete_node(self, node_id: str) -> None:
        uid = _uid()
        await self._edges.delete_many({
            "user_id": uid,
            "$or": [{"source_id": node_id}, {"target_id": node_id}],
        })
        await self._nodes.delete_one({"_id": node_id, "user_id": uid})

    async def get_nodes_by_status(self, status: MemoryStatus, limit: int = 100) -> list[MemoryNode]:
        uid = _uid()
        cursor = self._nodes.find(
            {"user_id": uid, "status": status.value}
        ).sort("created_at", -1).limit(limit)
        return [_doc_to_node(doc) async for doc in cursor]

    async def create_edge(self, edge: MemoryEdge) -> str:
        uid = _uid()
        doc = _edge_to_doc(edge, uid)
        # Upsert: dedup on (user_id, source_id, target_id, type)
        await self._edges.replace_one(
            {
                "user_id": uid,
                "source_id": str(edge.source_id),
                "target_id": str(edge.target_id),
                "type": edge.type.value,
            },
            doc,
            upsert=True,
        )
        return str(edge.id)

    async def get_edges(self, node_id: str, direction: str, edge_type: EdgeType | None = None) -> list[MemoryEdge]:
        uid = _uid()
        query: dict = {"user_id": uid}
        if direction == "outgoing":
            query["source_id"] = node_id
        elif direction == "incoming":
            query["target_id"] = node_id
        else:
            query["$or"] = [{"source_id": node_id}, {"target_id": node_id}]
        if edge_type:
            query["type"] = edge_type.value
        cursor = self._edges.find(query)
        return [_doc_to_edge(doc) async for doc in cursor]

    async def get_all_edges(self, node_ids: list[str] | None = None) -> list[MemoryEdge]:
        uid = _uid()
        query: dict = {"user_id": uid}
        if node_ids:
            query["$or"] = [
                {"source_id": {"$in": node_ids}},
                {"target_id": {"$in": node_ids}},
            ]
        cursor = self._edges.find(query)
        return [_doc_to_edge(doc) async for doc in cursor]

    async def update_edge_weight(self, edge_id: str, weight: float) -> None:
        uid = _uid()
        await self._edges.update_one(
            {"_id": edge_id, "user_id": uid},
            {"$set": {"weight": weight}},
        )

    async def delete_edge(self, edge_id: str) -> None:
        uid = _uid()
        await self._edges.delete_one({"_id": edge_id, "user_id": uid})

    async def edge_exists(self, source_id: str, target_id: str, edge_type: EdgeType) -> bool:
        uid = _uid()
        doc = await self._edges.find_one({
            "user_id": uid,
            "source_id": source_id,
            "target_id": target_id,
            "type": edge_type.value,
        })
        return doc is not None

    async def traverse(self, start_id: str, depth: int, edge_types: list[EdgeType] | None = None) -> list[MemoryNode]:
        uid = _uid()
        visited: set[str] = {start_id}
        queue: deque[tuple[str, int]] = deque([(start_id, 0)])
        result_ids: list[str] = []

        while queue:
            current, d = queue.popleft()
            if d >= depth:
                continue
            query: dict = {
                "user_id": uid,
                "$or": [{"source_id": current}, {"target_id": current}],
            }
            if edge_types:
                query["type"] = {"$in": [et.value for et in edge_types]}
            async for edge_doc in self._edges.find(query):
                neighbor = edge_doc["target_id"] if edge_doc["source_id"] == current else edge_doc["source_id"]
                if neighbor not in visited:
                    visited.add(neighbor)
                    result_ids.append(neighbor)
                    queue.append((neighbor, d + 1))

        if not result_ids:
            return []
        cursor = self._nodes.find({"_id": {"$in": result_ids}, "user_id": uid})
        return [_doc_to_node(doc) async for doc in cursor]

    async def get_causal_chain(self, node_id: str, direction: str) -> list[MemoryNode]:
        uid = _uid()
        causal_types = [et.value for et in CAUSAL_EDGE_TYPES]
        visited: set[str] = {node_id}
        queue: deque[str] = deque([node_id])
        result_ids: list[str] = []

        while queue:
            current = queue.popleft()
            if direction == "upstream":
                query = {"user_id": uid, "target_id": current, "type": {"$in": causal_types}}
                neighbor_field = "source_id"
            else:
                query = {"user_id": uid, "source_id": current, "type": {"$in": causal_types}}
                neighbor_field = "target_id"

            async for edge_doc in self._edges.find(query):
                neighbor = edge_doc[neighbor_field]
                if neighbor not in visited:
                    visited.add(neighbor)
                    result_ids.append(neighbor)
                    queue.append(neighbor)
                    if len(visited) > 100:
                        break
            if len(visited) > 100:
                break

        if not result_ids:
            return []
        cursor = self._nodes.find({"_id": {"$in": result_ids}, "user_id": uid})
        return [_doc_to_node(doc) async for doc in cursor]

    async def get_causal_weight(self, node_id: str) -> int:
        uid = _uid()
        causal_types = [et.value for et in CAUSAL_EDGE_TYPES]
        return await self._edges.count_documents({
            "user_id": uid,
            "$or": [{"source_id": node_id}, {"target_id": node_id}],
            "type": {"$in": causal_types},
        })

    async def get_degree(self, node_id: str) -> int:
        uid = _uid()
        return await self._edges.count_documents({
            "user_id": uid,
            "$or": [{"source_id": node_id}, {"target_id": node_id}],
        })

    async def is_orphan(self, node_id: str) -> bool:
        return (await self.get_degree(node_id)) == 0

    async def get_orphans(self) -> list[MemoryNode]:
        uid = _uid()
        # Get all node IDs and all IDs referenced in edges, then diff
        node_ids: set[str] = set()
        async for doc in self._nodes.find({"user_id": uid}, {"_id": 1}):
            node_ids.add(doc["_id"])

        connected_ids: set[str] = set()
        async for doc in self._edges.find({"user_id": uid}, {"source_id": 1, "target_id": 1}):
            connected_ids.add(doc["source_id"])
            connected_ids.add(doc["target_id"])

        orphan_ids = list(node_ids - connected_ids)
        if not orphan_ids:
            return []
        cursor = self._nodes.find({"_id": {"$in": orphan_ids}, "user_id": uid})
        return [_doc_to_node(doc) async for doc in cursor]

    async def vector_search(
        self, embedding: list[float], k: int = 10,
        status_filter: list[MemoryStatus] | None = None,
    ) -> list[tuple[MemoryNode, float]]:
        uid = _uid()
        field = _embedding_field()

        if self._vector_search_available:
            return await self._atlas_vector_search(embedding, k, status_filter, uid, field)
        return await self._fallback_vector_search(embedding, k, status_filter, uid, field)

    async def _atlas_vector_search(
        self, embedding: list[float], k: int,
        status_filter: list[MemoryStatus] | None,
        uid: str, field: str,
    ) -> list[tuple[MemoryNode, float]]:
        pre_filter: dict = {"user_id": uid}
        if status_filter:
            pre_filter["status"] = {"$in": [s.value for s in status_filter]}

        pipeline = [
            {
                "$vectorSearch": {
                    "index": f"vector_{field}",
                    "path": field,
                    "queryVector": embedding,
                    "numCandidates": k * 10,
                    "limit": k,
                    "filter": pre_filter,
                }
            },
            {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
        ]
        results = []
        async for doc in self._nodes.aggregate(pipeline):
            score = doc.pop("score", 0.0)
            results.append((_doc_to_node(doc), float(score)))
        return results

    async def _fallback_vector_search(
        self, embedding: list[float], k: int,
        status_filter: list[MemoryStatus] | None,
        uid: str, field: str,
    ) -> list[tuple[MemoryNode, float]]:
        query: dict = {"user_id": uid, field: {"$exists": True, "$ne": None}}
        if status_filter:
            query["status"] = {"$in": [s.value for s in status_filter]}

        scored: list[tuple[dict, float]] = []
        async for doc in self._nodes.find(query):
            doc_emb = doc.get(field)
            if not doc_emb:
                continue
            sim = _cosine_similarity(embedding, doc_emb)
            scored.append((doc, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [(_doc_to_node(doc), sim) for doc, sim in scored[:k]]

    async def keyword_search(self, query: str, entity_refs: list[str] | None = None, k: int = 10) -> list[MemoryNode]:
        uid = _uid()
        mongo_query: dict = {"user_id": uid}
        conditions: list[dict] = []

        if query:
            # Use $text search if available, fall back to regex
            conditions.append({
                "$or": [
                    {"content_summary": {"$regex": query, "$options": "i"}},
                    {"content_full": {"$regex": query, "$options": "i"}},
                ]
            })

        if entity_refs:
            conditions.append({"entity_refs": {"$in": entity_refs}})

        if conditions:
            mongo_query["$and"] = conditions

        cursor = self._nodes.find(mongo_query).limit(k)
        return [_doc_to_node(doc) async for doc in cursor]

    async def get_stats(self) -> dict:
        uid = _uid()
        # Nodes by status
        pipeline_nodes = [
            {"$match": {"user_id": uid}},
            {"$group": {"_id": "$status", "count": {"$sum": 1}}},
        ]
        nodes_by_status = {}
        async for doc in self._nodes.aggregate(pipeline_nodes):
            nodes_by_status[doc["_id"]] = doc["count"]

        # Edges by type
        pipeline_edges = [
            {"$match": {"user_id": uid}},
            {"$group": {"_id": "$type", "count": {"$sum": 1}}},
        ]
        edges_by_type = {}
        async for doc in self._edges.aggregate(pipeline_edges):
            edges_by_type[doc["_id"]] = doc["count"]

        total_nodes = sum(nodes_by_status.values())
        total_edges = sum(edges_by_type.values())

        # Orphan count
        connected_ids: set[str] = set()
        async for doc in self._edges.find({"user_id": uid}, {"source_id": 1, "target_id": 1}):
            connected_ids.add(doc["source_id"])
            connected_ids.add(doc["target_id"])
        orphan_count = await self._nodes.count_documents({
            "user_id": uid,
            "_id": {"$nin": list(connected_ids)},
        })

        # Max causal weight
        causal_types = [et.value for et in CAUSAL_EDGE_TYPES]
        pipeline_cw = [
            {"$match": {"user_id": uid, "type": {"$in": causal_types}}},
            {"$group": {"_id": "$source_id", "cnt": {"$sum": 1}}},
            {"$sort": {"cnt": -1}},
            {"$limit": 1},
        ]
        max_cw = 0
        async for doc in self._edges.aggregate(pipeline_cw):
            max_cw = doc["cnt"]

        return {
            "total_nodes": total_nodes,
            "nodes_by_status": nodes_by_status,
            "total_edges": total_edges,
            "edges_by_type": edges_by_type,
            "orphan_count": orphan_count,
            "max_causal_weight": max_cw,
        }
