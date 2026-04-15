"""In-memory storage providers for running without Docker/Redis/FalkorDB."""
from __future__ import annotations

import json as _json
from pathlib import Path

from genesys.context import current_user_id
from genesys.engine.scoring import cosine_similarity
from genesys.models.edge import MemoryEdge
from genesys.models.enums import CAUSAL_EDGE_TYPES, EdgeType, MemoryStatus
from genesys.models.node import MemoryNode


def _uid() -> str:
    uid = current_user_id.get(None)
    if uid is None:
        raise RuntimeError("No user context — current_user_id not set")
    return uid


class InMemoryCacheProvider:
    """CacheProvider backed by a plain dict."""

    def __init__(self):
        self._data: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self._data.get(key)

    async def set(self, key: str, value: str, ttl_seconds: int = 300) -> None:
        self._data[key] = value

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)

    async def exists(self, key: str) -> bool:
        return key in self._data


class InMemoryGraphProvider:
    """GraphStorageProvider backed by plain dicts with per-user isolation.

    If ``persist_path`` is set, state is saved to / loaded from that JSON file
    so data survives across process restarts (e.g. separate ``claude -p`` calls).
    """

    def __init__(self, persist_path: str | None = None):
        # Per-user storage: {user_id: {node_id: MemoryNode}}
        self._user_nodes: dict[str, dict[str, MemoryNode]] = {}
        self._user_edges: dict[str, list[MemoryEdge]] = {}
        self._persist_path = Path(persist_path) if persist_path else None

    @property
    def nodes(self) -> dict[str, MemoryNode]:
        """Return nodes for the current user."""
        uid = _uid()
        if uid not in self._user_nodes:
            self._user_nodes[uid] = {}
        return self._user_nodes[uid]

    @property
    def edges(self) -> list[MemoryEdge]:
        """Return edges for the current user."""
        uid = _uid()
        if uid not in self._user_edges:
            self._user_edges[uid] = []
        return self._user_edges[uid]

    @edges.setter
    def edges(self, value: list[MemoryEdge]) -> None:
        uid = _uid()
        self._user_edges[uid] = value

    # -- persistence helpers --------------------------------------------------

    def _save(self) -> None:
        if not self._persist_path:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        for uid, nodes in self._user_nodes.items():
            data[uid] = {
                "nodes": {nid: n.model_dump(mode="json") for nid, n in nodes.items()},
                "edges": [e.model_dump(mode="json") for e in self._user_edges.get(uid, [])],
            }
        self._persist_path.write_text(_json.dumps(data))

    def _load(self) -> None:
        if not self._persist_path or not self._persist_path.exists():
            return
        raw = _json.loads(self._persist_path.read_text())
        # Support both old format (flat) and new format (per-user)
        if "nodes" in raw and "edges" in raw:
            # Old flat format — migrate to default user
            uid = _uid()
            self._user_nodes[uid] = {nid: MemoryNode(**v) for nid, v in raw.get("nodes", {}).items()}
            self._user_edges[uid] = [MemoryEdge(**e) for e in raw.get("edges", [])]
        else:
            # New per-user format
            for uid, user_data in raw.items():
                self._user_nodes[uid] = {nid: MemoryNode(**v) for nid, v in user_data.get("nodes", {}).items()}
                self._user_edges[uid] = [MemoryEdge(**e) for e in user_data.get("edges", [])]

    async def initialize(self, user_id: str) -> None:
        self._load()

    async def destroy(self, user_id: str) -> None:
        self._user_nodes.pop(user_id, None)
        self._user_edges.pop(user_id, None)
        self._save()

    async def create_node(self, node: MemoryNode) -> str:
        nid = str(node.id)
        self.nodes[nid] = node
        self._save()
        return nid

    async def get_node(self, node_id: str) -> MemoryNode | None:
        return self.nodes.get(node_id)

    async def update_node(self, node_id: str, updates: dict) -> None:
        node = self.nodes.get(node_id)
        if node:
            for k, v in updates.items():
                setattr(node, k, v)
            self._save()

    async def delete_node(self, node_id: str) -> None:
        self.nodes.pop(node_id, None)
        self.edges = [e for e in self.edges if str(e.source_id) != node_id and str(e.target_id) != node_id]
        self._save()

    async def get_nodes_by_status(self, status: MemoryStatus, limit: int = 100) -> list[MemoryNode]:
        return [n for n in self.nodes.values() if n.status == status][:limit]

    async def create_edge(self, edge: MemoryEdge) -> str:
        self.edges.append(edge)
        self._save()
        return str(edge.id)

    async def get_edges(self, node_id: str, direction: str, edge_type: EdgeType | None = None) -> list[MemoryEdge]:
        result = []
        for e in self.edges:
            src, tgt = str(e.source_id), str(e.target_id)
            match = False
            if direction in ("out", "both") and src == node_id:
                match = True
            if direction in ("in", "both") and tgt == node_id:
                match = True
            if match and (edge_type is None or e.type == edge_type):
                result.append(e)
        return result

    async def get_all_edges(self, node_ids: list[str] | None = None) -> list[MemoryEdge]:
        if node_ids is None:
            return list(self.edges)
        ids = set(node_ids)
        return [e for e in self.edges if str(e.source_id) in ids or str(e.target_id) in ids]

    async def update_edge_weight(self, edge_id: str, weight: float) -> None:
        for e in self.edges:
            if str(e.id) == edge_id:
                e.weight = weight
                self._save()
                return

    async def delete_edge(self, edge_id: str) -> None:
        self.edges = [e for e in self.edges if str(e.id) != edge_id]
        self._save()

    async def edge_exists(self, source_id: str, target_id: str, edge_type: EdgeType) -> bool:
        return any(
            str(e.source_id) == source_id and str(e.target_id) == target_id and e.type == edge_type
            for e in self.edges
        )

    async def traverse(self, start_id: str, depth: int, edge_types: list[EdgeType] | None = None) -> list[MemoryNode]:
        visited = set()
        queue = [(start_id, 0)]
        result = []
        while queue:
            nid, d = queue.pop(0)
            if nid in visited or d > depth:
                continue
            visited.add(nid)
            node = self.nodes.get(nid)
            if node:
                result.append(node)
            if d < depth:
                for e in self.edges:
                    src, tgt = str(e.source_id), str(e.target_id)
                    if edge_types and e.type not in edge_types:
                        continue
                    if src == nid and tgt not in visited:
                        queue.append((tgt, d + 1))
                    if tgt == nid and src not in visited:
                        queue.append((src, d + 1))
        return result

    async def get_causal_chain(self, node_id: str, direction: str) -> list[MemoryNode]:
        visited = set()
        queue = [node_id]
        result = []
        while queue:
            nid = queue.pop(0)
            if nid in visited:
                continue
            visited.add(nid)
            for e in self.edges:
                if e.type not in CAUSAL_EDGE_TYPES:
                    continue
                src, tgt = str(e.source_id), str(e.target_id)
                next_id = None
                if direction == "upstream" and tgt == nid and src not in visited:
                    next_id = src
                elif direction == "downstream" and src == nid and tgt not in visited:
                    next_id = tgt
                if next_id:
                    queue.append(next_id)
                    node = self.nodes.get(next_id)
                    if node:
                        result.append(node)
        return result

    async def get_causal_weight(self, node_id: str) -> int:
        count = 0
        for e in self.edges:
            if e.type in CAUSAL_EDGE_TYPES:
                if str(e.source_id) == node_id or str(e.target_id) == node_id:
                    count += 1
        return count

    async def get_degree(self, node_id: str) -> int:
        count = 0
        for e in self.edges:
            if str(e.source_id) == node_id or str(e.target_id) == node_id:
                count += 1
        return count

    async def is_orphan(self, node_id: str) -> bool:
        return (await self.get_degree(node_id)) == 0

    async def get_orphans(self) -> list[MemoryNode]:
        connected = set()
        for e in self.edges:
            connected.add(str(e.source_id))
            connected.add(str(e.target_id))
        return [n for nid, n in self.nodes.items() if nid not in connected]

    async def vector_search(
        self, embedding: list[float], k: int = 10, status_filter: list[MemoryStatus] | None = None
    ) -> list[tuple[MemoryNode, float]]:
        scored = []
        for node in self.nodes.values():
            if status_filter and node.status not in status_filter:
                continue
            if node.embedding:
                sim = cosine_similarity(embedding, node.embedding)
                scored.append((node, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    async def keyword_search(self, query: str, entity_refs: list[str] | None = None, k: int = 10) -> list[MemoryNode]:
        query_lower = query.lower()
        results = []
        for node in self.nodes.values():
            if query_lower in (node.content_full or node.content_summary).lower():
                results.append(node)
        return results[:k]

    async def get_stats(self) -> dict:
        max_cw = 0
        for nid in self.nodes:
            cw = 0
            for e in self.edges:
                if e.type in CAUSAL_EDGE_TYPES:
                    if str(e.source_id) == nid or str(e.target_id) == nid:
                        cw += 1
            if cw > max_cw:
                max_cw = cw
        return {"nodes": len(self.nodes), "edges": len(self.edges), "max_causal_weight": max_cw}
