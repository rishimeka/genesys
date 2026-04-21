"""Status transition engine."""
from __future__ import annotations

from datetime import datetime, timezone

from genesys.engine import config
from genesys.engine.scoring import calculate_decay_score
from genesys.models.enums import MemoryStatus
from genesys.storage.base import EmbeddingProvider, GraphStorageProvider, LLMProvider


async def evaluate_transitions(
    graph: GraphStorageProvider,
    embeddings: EmbeddingProvider,
    llm: LLMProvider,
    context_embedding: list[float] | None = None,
    context_entities: list[str] | None = None,
) -> list[dict]:
    """Evaluate non-core, non-pruned nodes for status transitions."""
    transitions: list[dict] = []
    stats = await graph.get_stats()
    max_cw = stats.get("max_causal_weight", 1)

    # Tagged → Active: promote if node has edges (consolidation signal)
    tagged_nodes = await graph.get_nodes_by_status(MemoryStatus.TAGGED)
    for node in tagged_nodes:
        if not await graph.is_orphan(str(node.id)):
            await graph.update_node(str(node.id), {"status": MemoryStatus.ACTIVE})
            transitions.append({
                "node_id": str(node.id),
                "old": "tagged",
                "new": "active",
                "reason": "consolidation signal: edge formed",
            })
        else:
            # Auto-expire tagged memories after 24h with no connections
            age_hours = (datetime.now(timezone.utc) - node.created_at).total_seconds() / 3600
            if age_hours > config.TAGGED_EXPIRE_HOURS:
                await graph.update_node(str(node.id), {"status": MemoryStatus.DORMANT})
                transitions.append({
                    "node_id": str(node.id),
                    "old": "tagged",
                    "new": "dormant",
                    "reason": "no consolidation signal within 24h",
                })

    active_nodes = await graph.get_nodes_by_status(MemoryStatus.ACTIVE)
    episodic_nodes = await graph.get_nodes_by_status(MemoryStatus.EPISODIC)
    semantic_nodes = await graph.get_nodes_by_status(MemoryStatus.SEMANTIC)

    for node in active_nodes + episodic_nodes + semantic_nodes:
        if node.status == MemoryStatus.CORE:
            continue

        score = await calculate_decay_score(
            node, context_embedding, context_entities, graph, embeddings, max_cw
        )
        await graph.update_node(str(node.id), {"decay_score": score})

        # Active → Episodic
        if node.status == MemoryStatus.ACTIVE and score < config.ACTIVE_TO_EPISODIC_THRESHOLD:
            new_counter = node.irrelevance_counter + 1
            await graph.update_node(str(node.id), {"irrelevance_counter": new_counter})
            if new_counter >= config.ACTIVE_TO_EPISODIC_SESSIONS:
                await graph.update_node(str(node.id), {
                    "status": MemoryStatus.EPISODIC,
                    "irrelevance_counter": 0,
                })
                transitions.append({
                    "node_id": str(node.id),
                    "old": "active",
                    "new": "episodic",
                    "reason": f"decay_score {score:.2f} < {config.ACTIVE_TO_EPISODIC_THRESHOLD} for {config.ACTIVE_TO_EPISODIC_SESSIONS}+ sessions",
                })

        # Episodic/Semantic → Dormant
        elif node.status in (MemoryStatus.EPISODIC, MemoryStatus.SEMANTIC) and score < config.DORMANCY_THRESHOLD:
            now = datetime.now(timezone.utc)
            days_since = (now - node.last_reactivated_at).days
            if days_since > config.DORMANCY_DAYS and node.reactivation_count < config.DORMANCY_MAX_REACTIVATIONS:
                await graph.update_node(str(node.id), {"status": MemoryStatus.DORMANT})
                transitions.append({
                    "node_id": str(node.id),
                    "old": node.status.value,
                    "new": "dormant",
                    "reason": f"decay_score {score:.2f} < {config.DORMANCY_THRESHOLD}, inactive {days_since} days",
                })

    return transitions
