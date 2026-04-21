"""Episodic → semantic consolidation."""
from __future__ import annotations

from datetime import datetime, timezone

from genesys_memory.models.edge import MemoryEdge
from genesys_memory.models.enums import EdgeType, MemoryStatus
from genesys_memory.models.node import MemoryNode
from genesys_memory.storage.base import EmbeddingProvider, GraphStorageProvider, LLMProvider


async def check_and_consolidate(
    entity_ref: str,
    graph: GraphStorageProvider,
    llm: LLMProvider,
    embeddings: EmbeddingProvider,
) -> str | None:
    """
    If 3+ episodic memories share entity_ref, consolidate into 1 semantic memory.
    Returns new semantic node ID or None.
    """
    # Find episodic nodes with this entity_ref
    episodic_nodes = await graph.get_nodes_by_status(MemoryStatus.EPISODIC, limit=200)
    matching = [
        n for n in episodic_nodes
        if entity_ref.lower() in [e.lower() for e in n.entity_refs]
    ]

    if len(matching) < 3:
        return None

    # Consolidate
    texts = [n.content_full or n.content_summary for n in matching]
    consolidated_text = await llm.consolidate(texts)
    summary = consolidated_text[:200]

    # Generate embedding
    embedding = await embeddings.embed(consolidated_text)

    now = datetime.now(timezone.utc)
    semantic_node = MemoryNode(
        status=MemoryStatus.SEMANTIC,
        content_summary=summary,
        content_full=consolidated_text,
        embedding=embedding,
        created_at=now,
        last_accessed_at=now,
        last_reactivated_at=now,
        entity_refs=[entity_ref],
    )

    node_id = await graph.create_node(semantic_node)

    # Create DERIVED_FROM edges
    for source_node in matching:
        edge = MemoryEdge(
            source_id=semantic_node.id,
            target_id=source_node.id,
            type=EdgeType.DERIVED_FROM,
            weight=0.7,
        )
        await graph.create_edge(edge)

    return node_id
