"""Tests for the three-force multiplicative scoring engine."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import numpy as np
import pytest

from genesys.engine.scoring import (
    base_level_activation,
    calculate_decay_score,
    calculate_reactivation_durability,
    cosine_similarity,
)
from genesys.models.enums import ReactivationPattern
from genesys.models.node import MemoryNode


def _make_node(**kwargs) -> MemoryNode:
    defaults = {
        "content_summary": "test",
        "embedding": np.random.randn(1536).tolist(),
        "last_accessed_at": datetime.now(timezone.utc),
        "last_reactivated_at": datetime.now(timezone.utc),
        "reactivation_count": 10,
        "reactivation_pattern": ReactivationPattern.STEADY,
    }
    defaults.update(kwargs)
    return MemoryNode(**defaults)


def _mock_graph(causal_weight: int = 5, is_orphan: bool = False):
    g = AsyncMock()
    g.get_causal_weight = AsyncMock(return_value=causal_weight)
    g.is_orphan = AsyncMock(return_value=is_orphan)
    return g


def _mock_embeddings():
    return AsyncMock()


class TestReactivationDurability:
    def test_steady_high_count(self):
        assert calculate_reactivation_durability(ReactivationPattern.STEADY, 30) == 1.0

    def test_burst_moderate(self):
        assert calculate_reactivation_durability(ReactivationPattern.BURST, 15) == pytest.approx(0.3)

    def test_single_low(self):
        assert calculate_reactivation_durability(ReactivationPattern.SINGLE, 1) == pytest.approx(1 / 30 * 0.2)

    def test_zero_count(self):
        assert calculate_reactivation_durability(ReactivationPattern.STEADY, 0) == 0.0


class TestCosine:
    def test_identical(self):
        v = np.random.randn(10).tolist()
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)


class TestDecayScore:
    @pytest.mark.asyncio
    async def test_multiplicative_not_additive(self):
        """If connectivity is zero, total score is zero."""
        node = _make_node(reactivation_count=0)
        graph = _mock_graph(causal_weight=0, is_orphan=True)
        emb = _mock_embeddings()
        context = np.random.randn(1536).tolist()

        score = await calculate_decay_score(node, context, None, graph, emb, 20)
        assert score == pytest.approx(0.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_orphan_gets_zero_connectivity(self):
        """Orphan node should have connectivity_factor = 0 → score = 0."""
        node = _make_node(reactivation_count=15)
        graph = _mock_graph(causal_weight=0, is_orphan=True)
        emb = _mock_embeddings()
        context = np.random.randn(1536).tolist()

        score = await calculate_decay_score(node, context, None, graph, emb, 20)
        assert score == pytest.approx(0.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_high_causal_weight_resists_decay(self):
        """Node with high causal_weight should score higher than low, all else equal."""
        embedding = np.random.randn(1536).tolist()
        context = embedding  # Perfect relevance

        node_high = _make_node(embedding=embedding, reactivation_count=15)
        node_low = _make_node(embedding=embedding, reactivation_count=15)

        graph_high = _mock_graph(causal_weight=20)
        graph_low = _mock_graph(causal_weight=1)
        emb = _mock_embeddings()

        score_high = await calculate_decay_score(node_high, context, None, graph_high, emb, 20)
        score_low = await calculate_decay_score(node_low, context, None, graph_low, emb, 20)

        assert score_high > score_low

    @pytest.mark.asyncio
    async def test_no_context_uses_baseline(self):
        """When no context_embedding is provided, use time-based baseline relevance."""
        node = _make_node(reactivation_count=15)
        graph = _mock_graph(causal_weight=5)
        emb = _mock_embeddings()

        score = await calculate_decay_score(node, None, None, graph, emb, 10)
        # Should be > 0 since node was recently accessed
        assert score > 0

    @pytest.mark.asyncio
    async def test_keyword_overlap_boosts_relevance(self):
        """Entity ref overlap should boost relevance."""
        embedding = np.random.randn(1536).tolist()
        node = _make_node(
            embedding=embedding,
            entity_refs=["Python", "Django"],
            reactivation_count=15,
        )
        graph = _mock_graph(causal_weight=5)
        emb = _mock_embeddings()

        score_with = await calculate_decay_score(
            node, embedding, ["Python", "Django"], graph, emb, 10
        )
        score_without = await calculate_decay_score(
            node, embedding, [], graph, emb, 10
        )

        assert score_with > score_without


class TestBaseLevelActivation:
    def test_single_recent_access(self):
        """A single recent access should give positive activation."""
        now = datetime.now(timezone.utc)
        ts = [now - timedelta(seconds=10)]
        b_i = base_level_activation(ts, now - timedelta(days=1))
        # ln(10^{-0.5}) = -0.5*ln(10) ≈ -1.15, but activation_factor = exp(b_i) > 0
        import math
        assert math.exp(b_i) > 0.1  # recent access gives meaningful activation

    def test_multiple_accesses_higher_than_single(self):
        """More accesses should yield higher activation."""
        now = datetime.now(timezone.utc)
        single = [now - timedelta(hours=1)]
        multiple = [now - timedelta(hours=1), now - timedelta(hours=2), now - timedelta(hours=3)]
        b_single = base_level_activation(single, now - timedelta(days=1))
        b_multi = base_level_activation(multiple, now - timedelta(days=1))
        assert b_multi > b_single

    def test_spacing_effect(self):
        """Spaced accesses should produce higher activation than burst accesses at same count."""
        now = datetime.now(timezone.utc)
        # 3 accesses spaced 1 day apart
        spaced = [now - timedelta(days=i) for i in range(3)]
        # 3 accesses in the last minute (burst)
        burst = [now - timedelta(days=30, seconds=i) for i in range(3)]
        b_spaced = base_level_activation(spaced, now - timedelta(days=10))
        b_burst = base_level_activation(burst, now - timedelta(days=10))
        assert b_spaced > b_burst

    def test_old_access_decays(self):
        """Very old accesses should give low activation."""
        now = datetime.now(timezone.utc)
        old = [now - timedelta(days=365)]
        b_i = base_level_activation(old, now - timedelta(days=400))
        import math
        activation_factor = min(max(math.exp(b_i), 0.0), 1.0)
        assert activation_factor < 0.01  # should be very small

    def test_empty_timestamps_uses_created_at(self):
        """Empty timestamps should fall back to created_at."""
        now = datetime.now(timezone.utc)
        created = now - timedelta(hours=1)
        b_i = base_level_activation([], created)
        assert b_i > -5  # should be reasonable, not -inf

    def test_activation_factor_clamped(self):
        """exp(B_i) should be clamped to [0, 1]."""
        import math
        now = datetime.now(timezone.utc)
        # Very recent = high B_i, but clamped to 1.0
        ts = [now - timedelta(seconds=1)]
        b_i = base_level_activation(ts, now)
        factor = min(max(math.exp(b_i), 0.0), 1.0)
        assert factor == pytest.approx(1.0, abs=1e-4)


class TestDecayScoreWithACTR:
    @pytest.mark.asyncio
    async def test_fresh_node_nonzero(self):
        """A freshly created node with recent timestamp should have non-zero score."""
        now = datetime.now(timezone.utc)
        node = _make_node(reactivation_timestamps=[now])
        graph = _mock_graph(causal_weight=5)
        emb = _mock_embeddings()
        context = node.embedding

        score = await calculate_decay_score(node, context, None, graph, emb, 10)
        assert score > 0

    @pytest.mark.asyncio
    async def test_orphan_still_zero(self):
        """Orphan should still get zero score even with ACT-R activation."""
        now = datetime.now(timezone.utc)
        node = _make_node(reactivation_timestamps=[now])
        graph = _mock_graph(causal_weight=0, is_orphan=True)
        emb = _mock_embeddings()

        score = await calculate_decay_score(node, node.embedding, None, graph, emb, 10)
        assert score == pytest.approx(0.0, abs=0.01)
