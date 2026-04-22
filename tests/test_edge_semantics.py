"""Tests for edge semantics correctness.

Verifies that CONTRADICTS and SUPERSEDES edges are treated differently
from supportive edges (CAUSED_BY, SUPPORTS, DERIVED_FROM, RELATED_TO)
in orphan detection, core promotion, and retrieval scoring.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from genesys_memory.core_memory.promoter import consolidation_score, evaluate_core_promotion
from genesys_memory.models.edge import MemoryEdge
from genesys_memory.models.enums import EdgeType, SUPPORTIVE_EDGE_TYPES, NEGATIVE_EDGE_TYPES
from genesys_memory.models.node import MemoryNode
from genesys_memory.storage.memory import InMemoryGraphProvider


def _make_node(**kwargs) -> MemoryNode:
    defaults = {
        "content_summary": "test memory",
        "content_full": "test memory content",
    }
    defaults.update(kwargs)
    return MemoryNode(**defaults)


def _make_edge(source_id: uuid.UUID, target_id: uuid.UUID, edge_type: EdgeType, weight: float = 0.7) -> MemoryEdge:
    return MemoryEdge(
        source_id=source_id,
        target_id=target_id,
        type=edge_type,
        weight=weight,
    )


@pytest.fixture
def graph():
    from genesys_memory.context import current_user_id
    token = current_user_id.set("test-user")
    provider = InMemoryGraphProvider()
    provider._user_nodes["test-user"] = {}
    provider._user_edges["test-user"] = []
    yield provider
    current_user_id.reset(token)


class TestSupportiveDegree:
    @pytest.mark.asyncio
    async def test_supportive_edges_counted(self, graph: InMemoryGraphProvider):
        node_a = _make_node()
        node_b = _make_node()
        await graph.create_node(node_a)
        await graph.create_node(node_b)
        await graph.create_edge(_make_edge(node_a.id, node_b.id, EdgeType.CAUSED_BY))

        assert await graph.get_supportive_degree(str(node_a.id)) == 1
        assert await graph.get_supportive_degree(str(node_b.id)) == 1

    @pytest.mark.asyncio
    async def test_contradicts_not_counted_as_supportive(self, graph: InMemoryGraphProvider):
        node_a = _make_node()
        node_b = _make_node()
        await graph.create_node(node_a)
        await graph.create_node(node_b)
        await graph.create_edge(_make_edge(node_a.id, node_b.id, EdgeType.CONTRADICTS))

        assert await graph.get_supportive_degree(str(node_a.id)) == 0
        assert await graph.get_supportive_degree(str(node_b.id)) == 0

    @pytest.mark.asyncio
    async def test_supersedes_not_counted_as_supportive(self, graph: InMemoryGraphProvider):
        node_a = _make_node()
        node_b = _make_node()
        await graph.create_node(node_a)
        await graph.create_node(node_b)
        await graph.create_edge(_make_edge(node_a.id, node_b.id, EdgeType.SUPERSEDES))

        assert await graph.get_supportive_degree(str(node_a.id)) == 0

    @pytest.mark.asyncio
    async def test_mixed_edges_only_counts_supportive(self, graph: InMemoryGraphProvider):
        node_a = _make_node()
        node_b = _make_node()
        node_c = _make_node()
        await graph.create_node(node_a)
        await graph.create_node(node_b)
        await graph.create_node(node_c)
        await graph.create_edge(_make_edge(node_a.id, node_b.id, EdgeType.CONTRADICTS))
        await graph.create_edge(_make_edge(node_a.id, node_c.id, EdgeType.SUPPORTS))

        assert await graph.get_degree(str(node_a.id)) == 2
        assert await graph.get_supportive_degree(str(node_a.id)) == 1

    @pytest.mark.asyncio
    async def test_all_supportive_types_counted(self, graph: InMemoryGraphProvider):
        """Every type in SUPPORTIVE_EDGE_TYPES should be counted."""
        center = _make_node()
        await graph.create_node(center)
        for et in SUPPORTIVE_EDGE_TYPES:
            other = _make_node()
            await graph.create_node(other)
            await graph.create_edge(_make_edge(center.id, other.id, et))

        assert await graph.get_supportive_degree(str(center.id)) == len(SUPPORTIVE_EDGE_TYPES)

    @pytest.mark.asyncio
    async def test_no_negative_types_counted(self, graph: InMemoryGraphProvider):
        """No type in NEGATIVE_EDGE_TYPES should be counted."""
        center = _make_node()
        await graph.create_node(center)
        for et in NEGATIVE_EDGE_TYPES:
            other = _make_node()
            await graph.create_node(other)
            await graph.create_edge(_make_edge(center.id, other.id, et))

        assert await graph.get_supportive_degree(str(center.id)) == 0


class TestIsOrphan:
    @pytest.mark.asyncio
    async def test_node_with_only_contradicts_is_orphan(self, graph: InMemoryGraphProvider):
        """A node whose only edges are CONTRADICTS should be considered an orphan."""
        node_a = _make_node()
        node_b = _make_node()
        await graph.create_node(node_a)
        await graph.create_node(node_b)
        await graph.create_edge(_make_edge(node_a.id, node_b.id, EdgeType.CONTRADICTS))

        assert await graph.is_orphan(str(node_a.id)) is True
        assert await graph.is_orphan(str(node_b.id)) is True

    @pytest.mark.asyncio
    async def test_node_with_only_supersedes_is_orphan(self, graph: InMemoryGraphProvider):
        node_a = _make_node()
        node_b = _make_node()
        await graph.create_node(node_a)
        await graph.create_node(node_b)
        await graph.create_edge(_make_edge(node_a.id, node_b.id, EdgeType.SUPERSEDES))

        assert await graph.is_orphan(str(node_a.id)) is True

    @pytest.mark.asyncio
    async def test_node_with_supportive_edge_not_orphan(self, graph: InMemoryGraphProvider):
        node_a = _make_node()
        node_b = _make_node()
        await graph.create_node(node_a)
        await graph.create_node(node_b)
        await graph.create_edge(_make_edge(node_a.id, node_b.id, EdgeType.CAUSED_BY))

        assert await graph.is_orphan(str(node_a.id)) is False

    @pytest.mark.asyncio
    async def test_node_with_mixed_edges_not_orphan(self, graph: InMemoryGraphProvider):
        """A node with both CONTRADICTS and SUPPORTS edges is not orphan."""
        node_a = _make_node()
        node_b = _make_node()
        node_c = _make_node()
        await graph.create_node(node_a)
        await graph.create_node(node_b)
        await graph.create_node(node_c)
        await graph.create_edge(_make_edge(node_a.id, node_b.id, EdgeType.CONTRADICTS))
        await graph.create_edge(_make_edge(node_a.id, node_c.id, EdgeType.SUPPORTS))

        assert await graph.is_orphan(str(node_a.id)) is False


class TestGetOrphans:
    @pytest.mark.asyncio
    async def test_contradicts_only_node_in_orphan_list(self, graph: InMemoryGraphProvider):
        node_a = _make_node()
        node_b = _make_node()
        await graph.create_node(node_a)
        await graph.create_node(node_b)
        await graph.create_edge(_make_edge(node_a.id, node_b.id, EdgeType.CONTRADICTS))

        orphans = await graph.get_orphans()
        orphan_ids = {str(n.id) for n in orphans}
        assert str(node_a.id) in orphan_ids
        assert str(node_b.id) in orphan_ids

    @pytest.mark.asyncio
    async def test_supported_node_not_in_orphan_list(self, graph: InMemoryGraphProvider):
        node_a = _make_node()
        node_b = _make_node()
        await graph.create_node(node_a)
        await graph.create_node(node_b)
        await graph.create_edge(_make_edge(node_a.id, node_b.id, EdgeType.RELATED_TO))

        orphans = await graph.get_orphans()
        orphan_ids = {str(n.id) for n in orphans}
        assert str(node_a.id) not in orphan_ids
        assert str(node_b.id) not in orphan_ids


class TestCorePromotionWithEdgeSemantics:
    @pytest.mark.asyncio
    async def test_contradicts_only_node_not_promoted(self):
        """A node with only CONTRADICTS edges should not be promoted to core."""
        now = datetime.now(timezone.utc)
        node = _make_node(
            category=None,
            reactivation_timestamps=[now - timedelta(seconds=i * 3600) for i in range(10)],
            stability=3.0,
        )
        graph = AsyncMock()
        graph.get_supportive_degree = AsyncMock(return_value=0)
        graph.get_stats = AsyncMock(return_value={"total_nodes": 100, "total_edges": 200})
        graph.get_edges = AsyncMock(return_value=[])

        score = await consolidation_score(node, graph)
        # Hub score = 0 (supportive degree = 0), so total score should be low
        assert score < 0.55  # Below CORE_THRESHOLD

    @pytest.mark.asyncio
    async def test_supported_node_can_be_promoted(self):
        """A node with high supportive degree should be promotable."""
        now = datetime.now(timezone.utc)
        node = _make_node(
            category=None,
            reactivation_timestamps=[now - timedelta(seconds=i * 3600) for i in range(10)],
            stability=3.0,
        )
        graph = AsyncMock()
        graph.get_supportive_degree = AsyncMock(return_value=20)
        graph.get_stats = AsyncMock(return_value={"total_nodes": 100, "total_edges": 200})
        graph.get_edges = AsyncMock(return_value=[])

        should, reason = await evaluate_core_promotion(node, graph)
        assert should is True

    @pytest.mark.asyncio
    async def test_high_raw_degree_from_contradicts_no_promotion(self):
        """Even if raw degree is high, zero supportive degree means no promotion."""
        now = datetime.now(timezone.utc)
        node = _make_node(
            category=None,
            reactivation_timestamps=[now - timedelta(seconds=60)],
            stability=1.5,
        )
        graph = AsyncMock()
        # Simulates: 20 CONTRADICTS edges but 0 supportive edges
        graph.get_supportive_degree = AsyncMock(return_value=0)
        graph.get_stats = AsyncMock(return_value={"total_nodes": 100, "total_edges": 200})
        graph.get_edges = AsyncMock(return_value=[])

        score = await consolidation_score(node, graph)
        # Hub score component is 0, so score is at most activation + stability
        # 0.4 * activation + 0.0 * hub + 0.0 * schema + 0.1 * stability
        assert score < 0.55


class TestContradictionReasonCapture:
    @pytest.mark.asyncio
    async def test_detect_contradiction_returns_reason(self):
        """detect_contradiction should return (bool, float, str | None)."""
        from genesys_memory.engine.llm_provider import AnthropicLLMProvider

        provider = AnthropicLLMProvider.__new__(AnthropicLLMProvider)
        provider._ask = AsyncMock(return_value='{"contradicts": true, "confidence": 0.9, "reason": "Memory A says X, Memory B says not-X"}')

        result = await provider.detect_contradiction("Memory A", "Memory B")
        assert len(result) == 3
        is_contra, confidence, reason = result
        assert is_contra is True
        assert confidence == 0.9
        assert reason == "Memory A says X, Memory B says not-X"

    @pytest.mark.asyncio
    async def test_detect_contradiction_missing_reason(self):
        """If LLM omits reason, should return None for reason."""
        from genesys_memory.engine.llm_provider import AnthropicLLMProvider

        provider = AnthropicLLMProvider.__new__(AnthropicLLMProvider)
        provider._ask = AsyncMock(return_value='{"contradicts": false, "confidence": 0.2}')

        result = await provider.detect_contradiction("Memory A", "Memory B")
        is_contra, confidence, reason = result
        assert is_contra is False
        assert reason is None

    @pytest.mark.asyncio
    async def test_detect_contradiction_json_error(self):
        """Malformed JSON should return safe defaults."""
        from genesys_memory.engine.llm_provider import AnthropicLLMProvider

        provider = AnthropicLLMProvider.__new__(AnthropicLLMProvider)
        provider._ask = AsyncMock(return_value="not json")

        result = await provider.detect_contradiction("Memory A", "Memory B")
        assert result == (False, 0.0, None)


class TestCausalInferenceReasonCapture:
    @pytest.mark.asyncio
    async def test_infer_causal_edges_returns_reason(self):
        from genesys_memory.engine.llm_provider import AnthropicLLMProvider

        provider = AnthropicLLMProvider.__new__(AnthropicLLMProvider)
        provider._ask = AsyncMock(return_value='[{"target_id": "abc", "edge_type": "caused_by", "confidence": 0.8, "reason": "A led to B"}]')

        results = await provider.infer_causal_edges("new memory", [("abc", "existing memory")])
        assert len(results) == 1
        target_id, edge_type, confidence, reason = results[0]
        assert target_id == "abc"
        assert edge_type == EdgeType.CAUSED_BY
        assert confidence == 0.8
        assert reason == "A led to B"

    @pytest.mark.asyncio
    async def test_infer_causal_edges_missing_reason(self):
        from genesys_memory.engine.llm_provider import AnthropicLLMProvider

        provider = AnthropicLLMProvider.__new__(AnthropicLLMProvider)
        provider._ask = AsyncMock(return_value='[{"target_id": "abc", "edge_type": "supports", "confidence": 0.7}]')

        results = await provider.infer_causal_edges("new memory", [("abc", "existing")])
        assert len(results) == 1
        _, _, _, reason = results[0]
        assert reason is None


class TestEdgeTypeClassification:
    def test_supportive_types_correct(self):
        assert EdgeType.CAUSED_BY in SUPPORTIVE_EDGE_TYPES
        assert EdgeType.SUPPORTS in SUPPORTIVE_EDGE_TYPES
        assert EdgeType.DERIVED_FROM in SUPPORTIVE_EDGE_TYPES
        assert EdgeType.RELATED_TO in SUPPORTIVE_EDGE_TYPES
        assert EdgeType.CONTRADICTS not in SUPPORTIVE_EDGE_TYPES
        assert EdgeType.SUPERSEDES not in SUPPORTIVE_EDGE_TYPES

    def test_negative_types_correct(self):
        assert EdgeType.CONTRADICTS in NEGATIVE_EDGE_TYPES
        assert EdgeType.SUPERSEDES in NEGATIVE_EDGE_TYPES
        assert EdgeType.CAUSED_BY not in NEGATIVE_EDGE_TYPES
        assert EdgeType.RELATED_TO not in NEGATIVE_EDGE_TYPES

    def test_all_types_classified(self):
        """Every EdgeType should be in either SUPPORTIVE or NEGATIVE."""
        for et in EdgeType:
            if et == EdgeType.TEMPORAL_SEQUENCE:
                continue  # Neutral type
            assert et in SUPPORTIVE_EDGE_TYPES or et in NEGATIVE_EDGE_TYPES, f"{et} is unclassified"
