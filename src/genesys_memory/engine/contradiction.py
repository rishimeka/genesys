"""Contradiction detection between memories."""
from __future__ import annotations


from genesys_memory.models.edge import MemoryEdge
from genesys_memory.models.enums import EdgeType, MemoryStatus
from genesys_memory.models.node import MemoryNode
from genesys_memory.storage.base import EmbeddingProvider, GraphStorageProvider, LLMProvider


async def detect_contradictions(
    new_node: MemoryNode,
    graph: GraphStorageProvider,
    embeddings: EmbeddingProvider,
    llm: LLMProvider,
) -> list[tuple[str, float]]:
    """
    Check if new_node contradicts existing memories.
    1. Vector search for similarity > 0.85
    2. LLM confirmation
    3. Create CONTRADICTS edges for confirmed contradictions
    Returns list of (contradicted_node_id, confidence).
    """
    if not new_node.embedding:
        return []

    # Find highly similar memories (potential contradictions)
    candidates = await graph.vector_search(new_node.embedding, k=20)
    contradictions: list[tuple[str, float]] = []

    for candidate_node, sim_score in candidates:
        # Skip self
        if str(candidate_node.id) == str(new_node.id):
            continue
        # Only check high-similarity pairs
        # FalkorDB cosine distance: lower = more similar; convert to similarity
        similarity = 1.0 - sim_score if sim_score <= 1.0 else sim_score
        if similarity < 0.85:
            continue

        content_a = new_node.content_full or new_node.content_summary
        content_b = candidate_node.content_full or candidate_node.content_summary
        is_contradiction, confidence, reason = await llm.detect_contradiction(content_a, content_b)

        if is_contradiction and confidence > 0.7:
            # Create CONTRADICTS edge
            edge = MemoryEdge(
                source_id=new_node.id,
                target_id=candidate_node.id,
                type=EdgeType.CONTRADICTS,
                weight=confidence,
                reason=reason,
                created_by="llm_contradiction",
            )
            await graph.create_edge(edge)
            contradictions.append((str(candidate_node.id), confidence))

            # If contradicted memory is core, trigger supersession
            if candidate_node.status == MemoryStatus.CORE:
                supersede_edge = MemoryEdge(
                    source_id=new_node.id,
                    target_id=candidate_node.id,
                    type=EdgeType.SUPERSEDES,
                    weight=confidence,
                    reason=reason,
                    created_by="llm_contradiction",
                )
                await graph.create_edge(supersede_edge)
                await graph.update_node(str(candidate_node.id), {
                    "status": MemoryStatus.EPISODIC,
                })

    return contradictions
