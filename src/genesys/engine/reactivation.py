"""Cascade reactivation — boost causal neighbors when a node is accessed."""
from __future__ import annotations

from datetime import datetime, timezone

from genesys.engine import config
from genesys.models.enums import MemoryStatus
from genesys.storage.base import GraphStorageProvider


async def cascade_reactivate(
    node_id: str,
    graph: GraphStorageProvider,
    depth: int = config.CASCADE_DEPTH,
    decay_factor: float = config.CASCADE_DECAY_FACTOR,
) -> list[str]:
    """
    Traverse up to `depth` hops, apply partial reactivation with decay_factor per hop.
    Returns list of reactivated node IDs.
    """
    reactivated: list[str] = []
    now = datetime.now(timezone.utc)

    # BFS traversal
    visited: set[str] = {node_id}
    current_level = [node_id]
    current_strength = 1.0

    for _hop in range(depth):
        current_strength *= decay_factor
        next_level: list[str] = []

        for nid in current_level:
            neighbors = await graph.traverse(nid, depth=1)
            for neighbor in neighbors:
                nid_str = str(neighbor.id)
                if nid_str in visited:
                    continue
                visited.add(nid_str)
                next_level.append(nid_str)

                updates: dict = {
                    "reactivation_count": neighbor.reactivation_count + 1,
                    "last_reactivated_at": now,
                }

                # Revive dormant nodes if reactivation is strong enough
                if neighbor.status == MemoryStatus.DORMANT and current_strength >= config.DORMANT_REVIVAL_THRESHOLD:
                    updates["status"] = MemoryStatus.EPISODIC

                await graph.update_node(nid_str, updates)
                reactivated.append(nid_str)

        current_level = next_level

    return reactivated
