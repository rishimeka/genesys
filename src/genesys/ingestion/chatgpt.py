"""ChatGPT conversation history parser."""
from __future__ import annotations

import json
import logging
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from genesys.engine.config import MAX_INGEST_FILE_MB
from genesys.models.edge import MemoryEdge
from genesys.models.enums import EdgeType, MemoryStatus
from genesys.models.node import MemoryNode
from genesys.storage.base import EmbeddingProvider, EventBusProvider, GraphStorageProvider

logger = logging.getLogger("genesys.ingestion")


async def ingest_chatgpt_export(
    source: str | Path,
    graph: GraphStorageProvider,
    embeddings: EmbeddingProvider,
    event_bus: EventBusProvider | None = None,
    since: datetime | None = None,
) -> dict:
    """Parse a ChatGPT export and ingest conversations as memories.

    Args:
        source: Path to conversations.json or zip archive.
        graph: Graph storage provider.
        embeddings: Embedding provider.
        event_bus: Optional event bus for background processing.
        since: Only ingest conversations after this date.

    Returns:
        Dict with counts of memories and edges created.
    """
    path_check = Path(source)
    max_bytes = MAX_INGEST_FILE_MB * 1024 * 1024
    if path_check.exists() and path_check.stat().st_size > max_bytes:
        raise ValueError(f"File exceeds {MAX_INGEST_FILE_MB}MB limit: {path_check.stat().st_size / 1024 / 1024:.1f}MB")

    conversations = _load_conversations(source)

    memories_created = 0
    edges_created = 0

    for conv in conversations:
        conv_title = conv.get("title", "untitled")
        create_time = conv.get("create_time")
        if create_time and since:
            conv_dt = datetime.fromtimestamp(create_time, tz=timezone.utc)
            if conv_dt < since:
                continue

        messages = _extract_messages(conv)
        if not messages:
            continue

        texts = [m["content"] for m in messages]
        embs = await embeddings.embed_batch(texts)

        prev_node_id: str | None = None
        for msg, emb in zip(messages, embs):
            now = datetime.now(timezone.utc)
            node = MemoryNode(
                status=MemoryStatus.ACTIVE,
                content_summary=msg["content"][:200],
                content_full=msg["content"],
                embedding=emb,
                created_at=now,
                last_accessed_at=now,
                last_reactivated_at=now,
                decay_score=1.0,
                source_agent="chatgpt",
                source_session=conv_title,
            )
            node_id = await graph.create_node(node)
            memories_created += 1

            if prev_node_id:
                edge = MemoryEdge(
                    source_id=uuid.UUID(prev_node_id),
                    target_id=node.id,
                    type=EdgeType.TEMPORAL_SEQUENCE,
                    weight=0.5,
                )
                await graph.create_edge(edge)
                edges_created += 1

            if event_bus:
                await event_bus.publish("memory.created", {
                    "node_id": node_id,
                    "content_full": msg["content"],
                })

            prev_node_id = node_id

    return {"memories_created": memories_created, "edges_created": edges_created}


def _load_conversations(source: str | Path) -> list[dict]:
    """Load conversations from ChatGPT export."""
    path = Path(source)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                if ".." in name or name.startswith("/"):
                    logger.warning("Skipping suspicious zip entry: %s", name)
                    continue
                if name.endswith("conversations.json"):
                    with zf.open(name) as f:
                        return json.loads(f.read())
            return []
    else:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]


def _extract_messages(conv: dict) -> list[dict]:
    """Extract user messages from ChatGPT's mapping-based format."""
    messages = []
    mapping = conv.get("mapping", {})
    if not mapping:
        return messages

    # Build ordered list by traversing parent->children
    ordered_ids = _traverse_mapping(mapping)

    for node_id in ordered_ids:
        node_data = mapping[node_id]
        msg = node_data.get("message")
        if not msg:
            continue
        if msg.get("author", {}).get("role") != "user":
            continue
        parts = msg.get("content", {}).get("parts", [])
        content = " ".join(str(p) for p in parts if isinstance(p, str))
        if content.strip():
            messages.append({"role": "user", "content": content.strip()})

    return messages


def _traverse_mapping(mapping: dict) -> list[str]:
    """Traverse the ChatGPT mapping tree in order."""
    # Find root (node with no parent or parent not in mapping)
    roots = [
        nid for nid, data in mapping.items()
        if data.get("parent") is None or data.get("parent") not in mapping
    ]
    if not roots:
        return list(mapping.keys())

    ordered = []
    queue = list(roots)
    while queue:
        nid = queue.pop(0)
        ordered.append(nid)
        children = mapping.get(nid, {}).get("children", [])
        queue.extend(children)
    return ordered
