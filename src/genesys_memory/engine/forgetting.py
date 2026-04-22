"""Active forgetting — prune memories meeting ALL conjunctive criteria."""
from __future__ import annotations

from genesys_memory.engine import config
from genesys_memory.models.enums import MemoryStatus, Visibility
from genesys_memory.storage.base import GraphStorageProvider


async def sweep_for_forgetting(graph: GraphStorageProvider) -> list[str]:
    """
    Prune memories meeting ALL criteria:
    1. decay_score < 0.01
    2. is_orphan (zero supportive edges)
    3. NOT pinned
    4. status != CORE
    5. visibility != ORG (org nodes are exempt from per-user forgetting)

    Returns list of pruned node IDs.
    """
    pruned: list[str] = []
    orphans = await graph.get_orphans()

    for node in orphans:
        if node.visibility == Visibility.ORG:
            continue
        if (
            node.decay_score < config.FORGETTING_THRESHOLD
            and not node.pinned
            and node.status != MemoryStatus.CORE
        ):
            node_id = str(node.id)
            await graph.delete_node(node_id)
            pruned.append(node_id)

    return pruned
