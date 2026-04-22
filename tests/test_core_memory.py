"""Tests for core memory promotion."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from genesys_memory.core_memory.promoter import (
    consolidation_score,
    evaluate_core_promotion,
    promote_to_core,
)
from genesys_memory.models.enums import MemoryStatus
from genesys_memory.models.node import MemoryNode


def _make_node(**kwargs) -> MemoryNode:
    defaults = {"content_summary": "test"}
    defaults.update(kwargs)
    return MemoryNode(**defaults)


def _mock_graph(degree: int = 5, total_nodes: int = 100, total_edges: int = 200):
    """Create a mock graph with configurable degree and stats."""
    g = AsyncMock()
    g.get_supportive_degree = AsyncMock(return_value=degree)
    g.get_stats = AsyncMock(return_value={
        "total_nodes": total_nodes,
        "total_edges": total_edges,
    })
    # Edges for schema match — return empty by default
    g.get_edges = AsyncMock(return_value=[])
    return g


class TestCorePromotion:
    @pytest.mark.asyncio
    async def test_category_auto_promotion(self):
        """Memory classified as 'professional' should be auto-promoted."""
        node = _make_node(category="professional")
        graph = AsyncMock()

        should, reason = await evaluate_core_promotion(node, graph)
        assert should is True
        assert reason == "category_default:professional"

    @pytest.mark.asyncio
    async def test_category_educational(self):
        node = _make_node(category="educational")
        graph = AsyncMock()
        should, reason = await evaluate_core_promotion(node, graph)
        assert should is True

    @pytest.mark.asyncio
    async def test_category_not_promoted_low_score(self):
        """Non-auto-promote category with low consolidation score should not promote."""
        now = datetime.now(timezone.utc)
        node = _make_node(
            category="preference",
            reactivation_timestamps=[now - timedelta(days=365)],
            stability=1.0,
        )
        graph = _mock_graph(degree=0, total_nodes=100, total_edges=200)
        should, _ = await evaluate_core_promotion(node, graph)
        assert should is False

    @pytest.mark.asyncio
    async def test_high_connectivity_promotes(self):
        """Node with high degree + recent access + high stability should promote."""
        now = datetime.now(timezone.utc)
        node = _make_node(
            category=None,
            reactivation_timestamps=[now - timedelta(seconds=i * 3600) for i in range(10)],
            stability=3.0,
        )
        # degree=20, avg_degree=4 → hub_score high
        graph = _mock_graph(degree=20, total_nodes=100, total_edges=200)
        should, reason = await evaluate_core_promotion(node, graph)
        assert should is True
        assert "consolidation_score" in reason

    @pytest.mark.asyncio
    async def test_recent_active_node_with_connections(self):
        """Recently accessed node with moderate connections should have decent score."""
        now = datetime.now(timezone.utc)
        node = _make_node(
            category=None,
            reactivation_timestamps=[now - timedelta(minutes=5)],
            stability=2.0,
        )
        graph = _mock_graph(degree=8, total_nodes=50, total_edges=100)
        score = await consolidation_score(node, graph)
        assert score > 0.25

    @pytest.mark.asyncio
    async def test_old_orphan_low_score(self):
        """Old, disconnected node should score very low."""
        now = datetime.now(timezone.utc)
        node = _make_node(
            category=None,
            reactivation_timestamps=[now - timedelta(days=365)],
            stability=1.0,
        )
        graph = _mock_graph(degree=0, total_nodes=100, total_edges=200)
        score = await consolidation_score(node, graph)
        assert score < 0.15

    @pytest.mark.asyncio
    async def test_already_core_not_promoted(self):
        node = _make_node(status=MemoryStatus.CORE)
        graph = AsyncMock()
        should, _ = await evaluate_core_promotion(node, graph)
        assert should is False

    @pytest.mark.asyncio
    async def test_already_pinned_not_promoted(self):
        node = _make_node(pinned=True)
        graph = AsyncMock()
        should, _ = await evaluate_core_promotion(node, graph)
        assert should is False

    @pytest.mark.asyncio
    async def test_promote_to_core_updates_graph(self):
        graph = AsyncMock()
        await promote_to_core("test-id", "test_reason", graph)
        graph.update_node.assert_called_once()
        call_args = graph.update_node.call_args
        assert call_args[0][0] == "test-id"
        assert call_args[0][1]["status"] == MemoryStatus.CORE
        assert call_args[0][1]["promotion_reason"] == "test_reason"
