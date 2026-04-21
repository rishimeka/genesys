"""Core memory auto-promotion logic.

Uses a composite consolidation score based on ACT-R activation,
graph hub importance, and schema match (neighborhood density).
Category-based promotion is kept as a fast path.
"""
from __future__ import annotations

import math

from genesys.engine import config
from genesys.engine.scoring import base_level_activation
from genesys.models.enums import MemoryStatus
from genesys.models.node import MemoryNode
from genesys.storage.base import GraphStorageProvider


async def consolidation_score(
    node: MemoryNode,
    graph: GraphStorageProvider,
) -> float:
    """Composite score for core promotion.

    0.4 × activation + 0.3 × hub_score + 0.2 × schema_match + 0.1 × stability
    """
    # ACT-R activation (normalized to 0-1 range)
    b_i = base_level_activation(node.reactivation_timestamps, node.created_at)
    activation_norm = min(max(math.exp(b_i), 0.0), 1.0)

    # Hub importance: how connected vs average
    degree = await graph.get_degree(str(node.id))
    stats = await graph.get_stats()
    total_nodes = stats.get("total_nodes", stats.get("nodes", 1))
    total_edges = stats.get("total_edges", stats.get("edges", 0))
    avg_degree = (2 * total_edges / max(total_nodes, 1))  # each edge touches 2 nodes
    hub_score = min(degree / max(avg_degree, 1), config.HUB_SCORE_CAP) / config.HUB_SCORE_CAP

    # Schema match: fraction of neighbors that are themselves well-connected
    schema_match = 0.0
    if degree > 0:
        edges = await graph.get_edges(str(node.id), "both")
        well_connected = 0
        checked = 0
        for edge in edges[:config.SCHEMA_NEIGHBOR_CAP]:
            neighbor_id = str(edge.target_id) if str(edge.source_id) == str(node.id) else str(edge.source_id)
            n_degree = await graph.get_degree(neighbor_id)
            if n_degree > avg_degree:
                well_connected += 1
            checked += 1
        if checked > 0:
            schema_match = well_connected / checked

    # Stability as proxy for importance (grows with retrieval)
    stability_norm = min(node.stability / config.STABILITY_CAP, 1.0)

    return (
        config.CORE_ACTIVATION_WEIGHT * activation_norm +
        config.CORE_HUB_WEIGHT * hub_score +
        config.CORE_SCHEMA_WEIGHT * schema_match +
        config.CORE_STABILITY_WEIGHT * stability_norm
    )


async def evaluate_core_promotion(
    node: MemoryNode,
    graph: GraphStorageProvider,
) -> tuple[bool, str | None]:
    """Check if a node should be promoted to core. Returns (should_promote, reason)."""
    if node.status == MemoryStatus.CORE or node.pinned:
        return False, None

    # Fast path: category-based auto-promote
    if node.category in config.AUTO_PROMOTE_CATEGORIES:
        return True, f"category_default:{node.category}"

    # Composite consolidation score
    score = await consolidation_score(node, graph)
    if score >= config.CORE_THRESHOLD:
        return True, f"consolidation_score:{score:.3f}"

    return False, None


async def promote_to_core(node_id: str, reason: str, graph: GraphStorageProvider) -> None:
    """Promote a node to core status."""
    node = await graph.get_node(node_id)
    updates = {
        "status": MemoryStatus.CORE,
        "promotion_reason": reason,
    }
    # Never overwrite user-set pins
    if node and not node.pinned:
        updates["pinned"] = False
    await graph.update_node(node_id, updates)
