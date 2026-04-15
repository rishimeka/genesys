"""Obsidian vault storage backend for Genesys.

Reads markdown files as memory nodes, wikilinks as edges.
All Genesys metadata lives in a sidecar SQLite database at {vault}/.genesys/index.db.
Vault files are never modified.
"""
from __future__ import annotations

import hashlib
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite
import numpy as np
import yaml

from genesys.models.edge import MemoryEdge
from genesys.models.enums import CAUSAL_EDGE_TYPES, EdgeType, MemoryStatus
from genesys.models.node import MemoryNode

WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")
_VAULT_USER_ID = "vault_user"


class ObsidianGraphProvider:
    """GraphStorageProvider backed by an Obsidian vault + SQLite sidecar."""

    def __init__(self, vault_path: str):
        self.vault_path = Path(vault_path)
        self.db_dir = self.vault_path / ".genesys"
        self.db_path = self.db_dir / "index.db"
        self._db: aiosqlite.Connection | None = None
        # In-memory embedding cache: {node_id_str: np.array}
        self._embedding_cache: dict[str, np.ndarray] = {}

    # -- lifecycle ------------------------------------------------------------

    async def initialize(self, user_id: str = _VAULT_USER_ID) -> None:
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.db_path))
        self._db.row_factory = aiosqlite.Row
        await self._create_tables()
        await self._full_index()
        await self._load_embeddings()

    async def destroy(self, user_id: str = _VAULT_USER_ID) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    @property
    def db(self) -> aiosqlite.Connection:
        assert self._db is not None, "Call initialize() first"
        return self._db

    # -- table setup ----------------------------------------------------------

    async def _create_tables(self) -> None:
        await self.db.executescript("""
            CREATE TABLE IF NOT EXISTS memory_nodes (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'active',
                content_summary TEXT NOT NULL,
                content_full TEXT,
                content_ref TEXT,
                created_at TEXT NOT NULL,
                last_accessed_at TEXT NOT NULL,
                last_reactivated_at TEXT NOT NULL,
                decay_score REAL NOT NULL DEFAULT 1.0,
                causal_weight INTEGER NOT NULL DEFAULT 0,
                reactivation_count INTEGER NOT NULL DEFAULT 0,
                reactivation_pattern TEXT NOT NULL DEFAULT 'single',
                irrelevance_counter INTEGER NOT NULL DEFAULT 0,
                source_agent TEXT NOT NULL DEFAULT 'obsidian',
                source_session TEXT NOT NULL DEFAULT '',
                entity_refs TEXT NOT NULL DEFAULT '[]',
                category TEXT,
                stability REAL NOT NULL DEFAULT 1.0,
                pinned INTEGER NOT NULL DEFAULT 0,
                promotion_reason TEXT,
                reactivation_timestamps TEXT NOT NULL DEFAULT '[]',
                vault_path TEXT
            );
            CREATE TABLE IF NOT EXISTS memory_edges (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                type TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 0.7,
                created_at TEXT NOT NULL,
                metadata TEXT,
                FOREIGN KEY (source_id) REFERENCES memory_nodes(id),
                FOREIGN KEY (target_id) REFERENCES memory_nodes(id)
            );
            CREATE TABLE IF NOT EXISTS embeddings (
                node_id TEXT PRIMARY KEY,
                vector BLOB NOT NULL,
                FOREIGN KEY (node_id) REFERENCES memory_nodes(id)
            );
            CREATE TABLE IF NOT EXISTS sync_state (
                file_path TEXT PRIMARY KEY,
                node_id TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                last_synced TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_edges_source ON memory_edges(source_id);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON memory_edges(target_id);
            CREATE INDEX IF NOT EXISTS idx_nodes_status ON memory_nodes(status);
        """)

    # -- vault parsing --------------------------------------------------------

    def _scan_vault(self) -> list[Path]:
        """Recursively find all .md files, skipping dotfiles/dirs."""
        results = []
        for p in self.vault_path.rglob("*.md"):
            parts = p.relative_to(self.vault_path).parts
            if any(part.startswith(".") for part in parts):
                continue
            results.append(p)
        return results

    def _parse_note(self, path: Path) -> tuple[dict, str, list[str]]:
        """Parse a note into (frontmatter, body, wikilinks)."""
        text = path.read_text(encoding="utf-8", errors="replace")
        frontmatter: dict = {}
        body = text

        # Extract YAML frontmatter
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                try:
                    frontmatter = yaml.safe_load(parts[1]) or {}
                except yaml.YAMLError:
                    frontmatter = {}
                body = parts[2].strip()

        wikilinks = WIKILINK_RE.findall(body)
        return frontmatter, body, wikilinks

    def _content_hash(self, content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    # -- full index -----------------------------------------------------------

    async def _full_index(self) -> None:
        """Scan all vault files, upsert nodes and wikilink edges."""
        md_files = self._scan_vault()
        # Map filename (stem) → node_id for wikilink resolution
        stem_to_id: dict[str, str] = {}

        # First pass: load existing sync state
        existing: dict[str, tuple[str, str]] = {}  # file_path → (node_id, hash)
        async with self.db.execute("SELECT file_path, node_id, content_hash FROM sync_state") as cur:
            async for row in cur:
                existing[row["file_path"]] = (row["node_id"], row["content_hash"])

        # Second pass: upsert nodes
        for path in md_files:
            rel = str(path.relative_to(self.vault_path))
            text = path.read_text(encoding="utf-8", errors="replace")
            h = self._content_hash(text)

            if rel in existing and existing[rel][1] == h:
                # Unchanged
                stem_to_id[path.stem.lower()] = existing[rel][0]
                continue

            frontmatter, body, _ = self._parse_note(path)
            now = datetime.now(timezone.utc).isoformat()

            if rel in existing:
                node_id = existing[rel][0]
                await self.db.execute(
                    "UPDATE memory_nodes SET content_summary=?, content_full=?, entity_refs=?, vault_path=? WHERE id=?",
                    (body[:200], body, _json_dumps(frontmatter.get("tags", [])), rel, node_id),
                )
            else:
                node_id = str(uuid.uuid4())
                tags = frontmatter.get("tags", [])
                if isinstance(tags, str):
                    tags = [tags]
                await self.db.execute(
                    """INSERT INTO memory_nodes
                       (id, status, content_summary, content_full, content_ref, created_at,
                        last_accessed_at, last_reactivated_at, entity_refs, source_agent, vault_path)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (node_id, "active", body[:200], body, rel, now, now, now,
                     _json_dumps(tags), "obsidian", rel),
                )

            await self.db.execute(
                "INSERT OR REPLACE INTO sync_state (file_path, node_id, content_hash, last_synced) VALUES (?,?,?,?)",
                (rel, node_id, h, now),
            )
            stem_to_id[path.stem.lower()] = node_id

        # Populate stem_to_id for unchanged files too
        async with self.db.execute("SELECT file_path, node_id FROM sync_state") as cur:
            async for row in cur:
                stem = Path(row["file_path"]).stem.lower()
                stem_to_id[stem] = row["node_id"]

        # Third pass: create wikilink edges
        # Clear old wikilink edges and rebuild
        await self.db.execute("DELETE FROM memory_edges WHERE type = ?", (EdgeType.RELATED_TO.value,))

        for path in md_files:
            rel = str(path.relative_to(self.vault_path))
            _, _, wikilinks = self._parse_note(path)
            source_id = stem_to_id.get(path.stem.lower())
            if not source_id:
                continue
            for link in wikilinks:
                # Handle section links: [[Page#Section]] → Page
                link_stem = link.split("#")[0].strip().lower()
                target_id = stem_to_id.get(link_stem)
                if target_id and target_id != source_id:
                    edge_id = str(uuid.uuid4())
                    now = datetime.now(timezone.utc).isoformat()
                    await self.db.execute(
                        "INSERT INTO memory_edges (id, source_id, target_id, type, weight, created_at) VALUES (?,?,?,?,?,?)",
                        (edge_id, source_id, target_id, EdgeType.RELATED_TO.value, 0.7, now),
                    )

        await self.db.commit()

    async def _incremental_index(self, changed_files: list[str]) -> None:
        """Re-parse only changed files."""
        for rel in changed_files:
            path = self.vault_path / rel
            if not path.exists():
                # File deleted — mark node as PRUNED
                async with self.db.execute("SELECT node_id FROM sync_state WHERE file_path=?", (rel,)) as cur:
                    row = await cur.fetchone()
                if row:
                    await self.db.execute("UPDATE memory_nodes SET status=? WHERE id=?", ("pruned", row["node_id"]))
                    await self.db.execute("DELETE FROM sync_state WHERE file_path=?", (rel,))
                continue

            text = path.read_text(encoding="utf-8", errors="replace")
            h = self._content_hash(text)

            async with self.db.execute("SELECT node_id, content_hash FROM sync_state WHERE file_path=?", (rel,)) as cur:
                row = await cur.fetchone()

            if row and row["content_hash"] == h:
                continue  # unchanged

            frontmatter, body, _ = self._parse_note(path)
            now = datetime.now(timezone.utc).isoformat()
            tags = frontmatter.get("tags", [])
            if isinstance(tags, str):
                tags = [tags]

            if row:
                node_id = row["node_id"]
                await self.db.execute(
                    "UPDATE memory_nodes SET content_summary=?, content_full=?, entity_refs=?, vault_path=? WHERE id=?",
                    (body[:200], body, _json_dumps(tags), rel, node_id),
                )
            else:
                node_id = str(uuid.uuid4())
                await self.db.execute(
                    """INSERT INTO memory_nodes
                       (id, status, content_summary, content_full, content_ref, created_at,
                        last_accessed_at, last_reactivated_at, entity_refs, source_agent, vault_path)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (node_id, "active", body[:200], body, rel, now, now, now,
                     _json_dumps(tags), "obsidian", rel),
                )

            await self.db.execute(
                "INSERT OR REPLACE INTO sync_state (file_path, node_id, content_hash, last_synced) VALUES (?,?,?,?)",
                (rel, node_id, h, now),
            )

        await self.db.commit()

    # -- embedding cache ------------------------------------------------------

    async def _load_embeddings(self) -> None:
        self._embedding_cache.clear()
        async with self.db.execute("SELECT node_id, vector FROM embeddings") as cur:
            async for row in cur:
                vec = np.frombuffer(row["vector"], dtype=np.float32)
                self._embedding_cache[row["node_id"]] = vec

    async def store_embedding(self, node_id: str, embedding: list[float]) -> None:
        vec = np.array(embedding, dtype=np.float32)
        blob = vec.tobytes()
        await self.db.execute(
            "INSERT OR REPLACE INTO embeddings (node_id, vector) VALUES (?,?)",
            (node_id, blob),
        )
        await self.db.commit()
        self._embedding_cache[node_id] = vec

    # -- node CRUD ------------------------------------------------------------

    async def create_node(self, node: MemoryNode) -> str:
        nid = str(node.id)
        now = datetime.now(timezone.utc).isoformat()
        await self.db.execute(
            """INSERT INTO memory_nodes
               (id, status, content_summary, content_full, content_ref, created_at,
                last_accessed_at, last_reactivated_at, decay_score, causal_weight,
                reactivation_count, reactivation_pattern, irrelevance_counter,
                source_agent, source_session, entity_refs, category, stability,
                pinned, promotion_reason, reactivation_timestamps)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (nid, node.status.value, node.content_summary, node.content_full,
             node.content_ref, node.created_at.isoformat(), node.last_accessed_at.isoformat(),
             node.last_reactivated_at.isoformat(), node.decay_score, node.causal_weight,
             node.reactivation_count, node.reactivation_pattern.value, node.irrelevance_counter,
             node.source_agent, node.source_session, _json_dumps(node.entity_refs),
             node.category, node.stability, int(node.pinned), node.promotion_reason,
             _json_dumps([t.isoformat() for t in node.reactivation_timestamps])),
        )
        if node.embedding:
            await self.store_embedding(nid, node.embedding)
        await self.db.commit()
        return nid

    async def get_node(self, node_id: str) -> MemoryNode | None:
        async with self.db.execute("SELECT * FROM memory_nodes WHERE id=?", (node_id,)) as cur:
            row = await cur.fetchone()
        if not row:
            return None
        return _row_to_node(row, self._embedding_cache.get(node_id))

    async def update_node(self, node_id: str, updates: dict) -> None:
        if not updates:
            return
        cols = []
        vals = []
        for k, v in updates.items():
            if k == "embedding":
                if v is not None:
                    await self.store_embedding(node_id, v)
                continue
            if k == "entity_refs":
                v = _json_dumps(v)
            elif k == "reactivation_timestamps":
                v = _json_dumps([t.isoformat() if isinstance(t, datetime) else t for t in v])
            elif k == "status" and isinstance(v, MemoryStatus):
                v = v.value
            elif k == "pinned":
                v = int(v)
            cols.append(f"{k}=?")
            vals.append(v)
        if cols:
            vals.append(node_id)
            await self.db.execute(f"UPDATE memory_nodes SET {','.join(cols)} WHERE id=?", vals)
            await self.db.commit()

    async def delete_node(self, node_id: str) -> None:
        await self.db.execute("DELETE FROM memory_edges WHERE source_id=? OR target_id=?", (node_id, node_id))
        await self.db.execute("DELETE FROM embeddings WHERE node_id=?", (node_id,))
        await self.db.execute("DELETE FROM memory_nodes WHERE id=?", (node_id,))
        await self.db.commit()
        self._embedding_cache.pop(node_id, None)

    async def get_nodes_by_status(self, status: MemoryStatus, limit: int = 100) -> list[MemoryNode]:
        async with self.db.execute(
            "SELECT * FROM memory_nodes WHERE status=? LIMIT ?", (status.value, limit)
        ) as cur:
            rows = await cur.fetchall()
        return [_row_to_node(r, self._embedding_cache.get(r["id"])) for r in rows]

    # -- edge CRUD ------------------------------------------------------------

    async def create_edge(self, edge: MemoryEdge) -> str:
        eid = str(edge.id)
        await self.db.execute(
            "INSERT INTO memory_edges (id, source_id, target_id, type, weight, created_at, metadata) VALUES (?,?,?,?,?,?,?)",
            (eid, str(edge.source_id), str(edge.target_id), edge.type.value,
             edge.weight, edge.created_at.isoformat(),
             _json_dumps(edge.metadata) if edge.metadata else None),
        )
        await self.db.commit()
        return eid

    async def get_edges(self, node_id: str, direction: str, edge_type: EdgeType | None = None) -> list[MemoryEdge]:
        if direction == "outgoing":
            sql = "SELECT * FROM memory_edges WHERE source_id=?"
        elif direction == "incoming":
            sql = "SELECT * FROM memory_edges WHERE target_id=?"
        else:
            sql = "SELECT * FROM memory_edges WHERE source_id=? OR target_id=?"
        params: list = [node_id]
        if direction == "both":
            params.append(node_id)
        if edge_type:
            sql += " AND type=?"
            params.append(edge_type.value)
        async with self.db.execute(sql, params) as cur:
            rows = await cur.fetchall()
        return [_row_to_edge(r) for r in rows]

    async def update_edge_weight(self, edge_id: str, weight: float) -> None:
        await self.db.execute("UPDATE memory_edges SET weight=? WHERE id=?", (weight, edge_id))
        await self.db.commit()

    async def delete_edge(self, edge_id: str) -> None:
        await self.db.execute("DELETE FROM memory_edges WHERE id=?", (edge_id,))
        await self.db.commit()

    async def get_all_edges(self, node_ids: list[str] | None = None) -> list[MemoryEdge]:
        if node_ids:
            placeholders = ",".join("?" for _ in node_ids)
            sql = f"SELECT * FROM memory_edges WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})"
            params = node_ids + node_ids
        else:
            sql = "SELECT * FROM memory_edges"
            params = []
        async with self.db.execute(sql, params) as cur:
            rows = await cur.fetchall()
        return [_row_to_edge(r) for r in rows]

    async def edge_exists(self, source_id: str, target_id: str, edge_type: EdgeType) -> bool:
        async with self.db.execute(
            "SELECT 1 FROM memory_edges WHERE source_id=? AND target_id=? AND type=?",
            (source_id, target_id, edge_type.value),
        ) as cur:
            return await cur.fetchone() is not None

    # -- graph traversal ------------------------------------------------------

    async def traverse(self, start_id: str, depth: int, edge_types: list[EdgeType] | None = None) -> list[MemoryNode]:
        visited: set[str] = set()
        frontier = {start_id}
        for _ in range(depth):
            next_frontier: set[str] = set()
            for nid in frontier:
                if nid in visited:
                    continue
                visited.add(nid)
                edges = await self.get_edges(nid, "both")
                for e in edges:
                    if edge_types and e.type not in edge_types:
                        continue
                    neighbor = str(e.target_id) if str(e.source_id) == nid else str(e.source_id)
                    if neighbor not in visited:
                        next_frontier.add(neighbor)
            frontier = next_frontier
        visited.discard(start_id)
        nodes = []
        for nid in visited:
            n = await self.get_node(nid)
            if n:
                nodes.append(n)
        return nodes

    async def get_causal_chain(self, node_id: str, direction: str) -> list[MemoryNode]:
        causal_types = [et for et in CAUSAL_EDGE_TYPES]
        visited: set[str] = set()
        result: list[MemoryNode] = []
        frontier = [node_id]
        while frontier:
            nid = frontier.pop(0)
            if nid in visited:
                continue
            visited.add(nid)
            edges = await self.get_edges(nid, direction)
            for e in edges:
                if e.type not in causal_types:
                    continue
                neighbor = str(e.target_id) if direction == "outgoing" else str(e.source_id)
                if neighbor not in visited:
                    n = await self.get_node(neighbor)
                    if n:
                        result.append(n)
                    frontier.append(neighbor)
        return result

    async def get_causal_weight(self, node_id: str) -> int:
        count = 0
        async with self.db.execute(
            "SELECT COUNT(*) FROM memory_edges WHERE (source_id=? OR target_id=?) AND type IN (?,?,?)",
            (node_id, node_id, EdgeType.CAUSED_BY.value, EdgeType.SUPPORTS.value, EdgeType.DERIVED_FROM.value),
        ) as cur:
            row = await cur.fetchone()
            count = row[0] if row else 0
        return count

    async def get_degree(self, node_id: str) -> int:
        async with self.db.execute(
            "SELECT COUNT(*) FROM memory_edges WHERE source_id=? OR target_id=?",
            (node_id, node_id),
        ) as cur:
            row = await cur.fetchone()
            return row[0] if row else 0

    async def is_orphan(self, node_id: str) -> bool:
        return (await self.get_degree(node_id)) == 0

    async def get_orphans(self) -> list[MemoryNode]:
        sql = """SELECT n.* FROM memory_nodes n
                 LEFT JOIN memory_edges e ON n.id = e.source_id OR n.id = e.target_id
                 WHERE e.id IS NULL"""
        async with self.db.execute(sql) as cur:
            rows = await cur.fetchall()
        return [_row_to_node(r, self._embedding_cache.get(r["id"])) for r in rows]

    # -- search ---------------------------------------------------------------

    async def vector_search(
        self, embedding: list[float], k: int = 10, status_filter: list[MemoryStatus] | None = None
    ) -> list[tuple[MemoryNode, float]]:
        if not self._embedding_cache:
            return []
        query_vec = np.array(embedding, dtype=np.float32)
        scores: list[tuple[str, float]] = []
        for nid, vec in self._embedding_cache.items():
            sim = _cosine_sim(query_vec, vec)
            scores.append((nid, float(sim)))
        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for nid, sim in scores[:k * 3]:
            node = await self.get_node(nid)
            if not node:
                continue
            if status_filter and node.status not in status_filter:
                continue
            results.append((node, sim))
            if len(results) >= k:
                break
        return results

    async def keyword_search(self, query: str, entity_refs: list[str] | None = None, k: int = 10) -> list[MemoryNode]:
        terms = query.lower().split()
        if not terms:
            return []
        conditions = []
        params: list[str] = []
        for term in terms:
            conditions.append("(LOWER(content_summary) LIKE ? OR LOWER(COALESCE(content_full,'')) LIKE ?)")
            params.extend([f"%{term}%", f"%{term}%"])
        sql = f"SELECT * FROM memory_nodes WHERE {' OR '.join(conditions)} LIMIT ?"
        params.append(str(k))
        async with self.db.execute(sql, params) as cur:
            rows = await cur.fetchall()
        return [_row_to_node(r, self._embedding_cache.get(r["id"])) for r in rows]

    async def get_stats(self) -> dict:
        stats: dict = {}
        async with self.db.execute("SELECT status, COUNT(*) FROM memory_nodes GROUP BY status") as cur:
            async for row in cur:
                stats[row[0]] = row[1]
        async with self.db.execute("SELECT COUNT(*) FROM memory_edges") as cur:
            row = await cur.fetchone()
            stats["total_edges"] = row[0] if row else 0
        async with self.db.execute("SELECT COUNT(*) FROM memory_nodes") as cur:
            row = await cur.fetchone()
            stats["total_nodes"] = row[0] if row else 0
        return stats


# -- helpers ------------------------------------------------------------------

import json as _json_mod


def _json_dumps(obj) -> str:
    return _json_mod.dumps(obj, default=str)


def _json_loads(s: str | None, default=None):
    if not s:
        return default if default is not None else []
    try:
        return _json_mod.loads(s)
    except _json_mod.JSONDecodeError:
        return default if default is not None else []


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(dot / (na * nb))


def _row_to_node(row, embedding: np.ndarray | None = None) -> MemoryNode:
    from genesys.models.enums import ReactivationPattern
    ts_raw = _json_loads(row["reactivation_timestamps"], [])
    timestamps = []
    for t in ts_raw:
        if isinstance(t, str):
            try:
                timestamps.append(datetime.fromisoformat(t))
            except ValueError:
                pass
        elif isinstance(t, datetime):
            timestamps.append(t)

    return MemoryNode(
        id=uuid.UUID(row["id"]),
        status=MemoryStatus(row["status"]),
        content_summary=row["content_summary"],
        content_full=row["content_full"],
        content_ref=row["content_ref"],
        embedding=embedding.tolist() if embedding is not None else None,
        created_at=datetime.fromisoformat(row["created_at"]),
        last_accessed_at=datetime.fromisoformat(row["last_accessed_at"]),
        last_reactivated_at=datetime.fromisoformat(row["last_reactivated_at"]),
        decay_score=row["decay_score"],
        causal_weight=row["causal_weight"],
        reactivation_count=row["reactivation_count"],
        reactivation_pattern=ReactivationPattern(row["reactivation_pattern"]),
        irrelevance_counter=row["irrelevance_counter"],
        source_agent=row["source_agent"],
        source_session=row["source_session"],
        entity_refs=_json_loads(row["entity_refs"], []),
        category=row["category"],
        stability=row["stability"],
        pinned=bool(row["pinned"]),
        promotion_reason=row["promotion_reason"],
        reactivation_timestamps=timestamps,
    )


def _row_to_edge(row) -> MemoryEdge:
    return MemoryEdge(
        id=uuid.UUID(row["id"]),
        source_id=uuid.UUID(row["source_id"]),
        target_id=uuid.UUID(row["target_id"]),
        type=EdgeType(row["type"]),
        weight=row["weight"],
        created_at=datetime.fromisoformat(row["created_at"]),
        metadata=_json_loads(row["metadata"], None) if row["metadata"] else None,
    )
