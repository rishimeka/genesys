"""Claude conversation history parser."""
from __future__ import annotations

import json
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from genesys.models.edge import MemoryEdge
from genesys.models.enums import EdgeType, MemoryStatus
from genesys.models.node import MemoryNode
from genesys.storage.base import EmbeddingProvider, EventBusProvider, GraphStorageProvider


async def ingest_claude_export(
    source: str | Path,
    graph: GraphStorageProvider,
    embeddings: EmbeddingProvider,
    event_bus: EventBusProvider | None = None,
    since: datetime | None = None,
) -> dict:
    """Parse a Claude export (JSON or zip) and ingest conversations as memories.

    Args:
        source: Path to a JSON file or zip archive of Claude conversations.
        graph: Graph storage provider.
        embeddings: Embedding provider.
        event_bus: Optional event bus for background processing.
        since: Only ingest conversations after this date.

    Returns:
        Dict with counts of memories and edges created.
    """
    conversations = _load_conversations(source)

    memories_created = 0
    edges_created = 0

    for conv in conversations:
        conv_title = conv.get("name", conv.get("title", "untitled"))
        conv_created = conv.get("created_at")
        if conv_created and since:
            if isinstance(conv_created, str):
                conv_dt = datetime.fromisoformat(conv_created.replace("Z", "+00:00"))
            else:
                conv_dt = datetime.fromtimestamp(conv_created, tz=timezone.utc)
            if conv_dt < since:
                continue

        # Extract user messages
        messages = _extract_messages(conv)
        if not messages:
            continue

        # Batch embed all messages
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
                source_agent="claude",
                source_session=conv_title,
            )
            node_id = await graph.create_node(node)
            memories_created += 1

            # Create temporal sequence edge to previous message in same conversation
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
    """Load conversations from a JSON file or zip archive."""
    path = Path(source)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            conversations = []
            for name in zf.namelist():
                if name.endswith(".json"):
                    with zf.open(name) as f:
                        data = json.loads(f.read())
                        if isinstance(data, list):
                            conversations.extend(data)
                        else:
                            conversations.append(data)
            return conversations
    else:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]


def _extract_messages(conv: dict) -> list[dict]:
    """Extract user messages from a Claude conversation."""
    messages = []
    # Claude export format: chat_messages list
    chat_messages = conv.get("chat_messages", [])
    for msg in chat_messages:
        if msg.get("sender") == "human":
            content = ""
            # Content can be a string or list of content blocks
            raw = msg.get("text", msg.get("content", ""))
            if isinstance(raw, list):
                content = " ".join(
                    block.get("text", "") for block in raw if isinstance(block, dict)
                )
            else:
                content = str(raw)
            if content.strip():
                messages.append({"role": "user", "content": content.strip()})
    return messages
