"""Tests for active forgetting — safety-critical conjunctive criteria."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from genesys.engine.forgetting import sweep_for_forgetting
from genesys.models.enums import MemoryStatus
from genesys.models.node import MemoryNode


def _make_orphan_node(**kwargs) -> MemoryNode:
    defaults = {
        "content_summary": "orphan",
        "decay_score": 0.0,
        "pinned": False,
        "status": MemoryStatus.ACTIVE,
    }
    defaults.update(kwargs)
    return MemoryNode(**defaults)


class TestForgetting:
    @pytest.mark.asyncio
    async def test_prune_meets_all_criteria(self):
        """Node meeting all 4 criteria should be pruned."""
        node = _make_orphan_node(decay_score=0.0, pinned=False, status=MemoryStatus.DORMANT)
        graph = AsyncMock()
        graph.get_orphans = AsyncMock(return_value=[node])
        graph.delete_node = AsyncMock()

        pruned = await sweep_for_forgetting(graph)
        assert len(pruned) == 1
        graph.delete_node.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_prune_if_has_edges(self):
        """Node with edges won't appear in orphans, so won't be pruned."""
        graph = AsyncMock()
        graph.get_orphans = AsyncMock(return_value=[])  # no orphans
        graph.delete_node = AsyncMock()

        pruned = await sweep_for_forgetting(graph)
        assert len(pruned) == 0
        graph.delete_node.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_prune_if_pinned(self):
        """Pinned orphan with decay_score=0 must NOT be pruned."""
        node = _make_orphan_node(decay_score=0.0, pinned=True)
        graph = AsyncMock()
        graph.get_orphans = AsyncMock(return_value=[node])
        graph.delete_node = AsyncMock()

        pruned = await sweep_for_forgetting(graph)
        assert len(pruned) == 0
        graph.delete_node.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_prune_if_core(self):
        """Core orphan with decay_score=0 must NOT be pruned."""
        node = _make_orphan_node(decay_score=0.0, status=MemoryStatus.CORE)
        graph = AsyncMock()
        graph.get_orphans = AsyncMock(return_value=[node])
        graph.delete_node = AsyncMock()

        pruned = await sweep_for_forgetting(graph)
        assert len(pruned) == 0
        graph.delete_node.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_prune_if_high_decay_score(self):
        """Orphan with decay_score > 0.01 must NOT be pruned."""
        node = _make_orphan_node(decay_score=0.5)
        graph = AsyncMock()
        graph.get_orphans = AsyncMock(return_value=[node])
        graph.delete_node = AsyncMock()

        pruned = await sweep_for_forgetting(graph)
        assert len(pruned) == 0
        graph.delete_node.assert_not_called()

    @pytest.mark.asyncio
    async def test_multiple_orphans_mixed(self):
        """Only orphans meeting ALL criteria are pruned."""
        pruneable = _make_orphan_node(decay_score=0.0, status=MemoryStatus.DORMANT)
        pinned = _make_orphan_node(decay_score=0.0, pinned=True)
        high_score = _make_orphan_node(decay_score=0.5)

        graph = AsyncMock()
        graph.get_orphans = AsyncMock(return_value=[pruneable, pinned, high_score])
        graph.delete_node = AsyncMock()

        pruned = await sweep_for_forgetting(graph)
        assert len(pruned) == 1
        graph.delete_node.assert_called_once()
