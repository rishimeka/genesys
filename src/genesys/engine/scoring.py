"""Three-force multiplicative scoring engine."""
from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np

from genesys.engine import config
from genesys.models.enums import ReactivationPattern
from genesys.models.node import MemoryNode
from genesys.storage.base import EmbeddingProvider, GraphStorageProvider


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.asarray(a), np.asarray(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def calculate_reactivation_durability(pattern: ReactivationPattern, count: int) -> float:
    """Legacy helper kept for backward compat — not used in new scoring."""
    base = min(count / 30.0, 1.0)
    multipliers = {
        ReactivationPattern.STEADY: 1.0,
        ReactivationPattern.BURST: 0.6,
        ReactivationPattern.SINGLE: 0.2,
    }
    return base * multipliers[pattern]


def base_level_activation(
    timestamps: list[datetime],
    created_at: datetime,
    d: float = config.ACTR_DECAY_EXPONENT,
) -> float:
    """ACT-R base-level activation: B_i = ln(Σ t_j^{-d}).

    Each t_j is the time (in seconds) since the j-th access.
    Uses created_at as fallback if no timestamps are provided.
    """
    now = datetime.now(timezone.utc)
    ts = timestamps if timestamps else [created_at]
    total = 0.0
    for t in ts:
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        elapsed = (now - t).total_seconds()
        if elapsed < 1.0:
            elapsed = 1.0  # clamp to avoid division issues
        total += elapsed ** (-d)
    if total <= 0:
        return -10.0  # effectively zero after exp()
    return math.log(total)


async def calculate_decay_score(
    node: MemoryNode,
    context_embedding: list[float] | None,
    context_entities: list[str] | None,
    graph: GraphStorageProvider,
    embeddings: EmbeddingProvider,
    max_causal_weight: int,
) -> float:
    """decay_score = relevance × connectivity_factor × activation_factor"""

    # Force 1: Relevance
    if context_embedding and node.embedding:
        vector_sim = cosine_similarity(node.embedding, context_embedding)
    else:
        vector_sim = 0.0

    if context_entities and node.entity_refs:
        keyword_overlap = len(set(node.entity_refs) & set(context_entities)) / max(len(node.entity_refs), 1)
    else:
        keyword_overlap = 0.0

    relevance = config.RELEVANCE_VECTOR_WEIGHT * vector_sim + config.RELEVANCE_KEYWORD_WEIGHT * keyword_overlap

    if context_embedding is None:
        now = datetime.now(timezone.utc)
        days_since_access = (now - node.last_accessed_at).days
        relevance = max(0.1, 1.0 - (days_since_access / 365.0))

    # Force 2: Connectivity (unchanged)
    causal_weight = await graph.get_causal_weight(str(node.id))
    if max_causal_weight > 0:
        raw = math.log2(1 + causal_weight) / math.log2(1 + max_causal_weight)
        connectivity_factor = raw ** 2
    else:
        connectivity_factor = 0.0

    if await graph.is_orphan(str(node.id)):
        connectivity_factor = 0.0
    elif connectivity_factor < config.MIN_CONNECTIVITY:
        connectivity_factor = config.MIN_CONNECTIVITY

    # Force 3: ACT-R base-level activation
    b_i = base_level_activation(node.reactivation_timestamps, node.created_at)
    activation_factor = min(max(math.exp(b_i), 0.0), 1.0)

    return relevance * connectivity_factor * activation_factor
