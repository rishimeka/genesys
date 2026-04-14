"""Tests for status transition engine."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from genesys.engine.transitions import evaluate_transitions
from genesys.models.enums import MemoryStatus, ReactivationPattern
from genesys.models.node import MemoryNode


def _make_node(**kwargs) -> MemoryNode:
    defaults = {
        "content_summary": "test",
        "last_accessed_at": datetime.now(timezone.utc),
        "last_reactivated_at": datetime.now(timezone.utc),
    }
    defaults.update(kwargs)
    return MemoryNode(**defaults)


class TestTransitions:
    @pytest.mark.asyncio
    async def test_active_to_episodic_after_3_sessions(self):
        """Active node with low score for 3+ sessions transitions to episodic."""
        node = _make_node(
            status=MemoryStatus.ACTIVE,
            irrelevance_counter=2,  # Already at 2, this eval makes 3
        )
        graph = AsyncMock()
        graph.get_stats = AsyncMock(return_value={"max_causal_weight": 1})
        graph.get_nodes_by_status = AsyncMock(side_effect=lambda s, **kw: [node] if s == MemoryStatus.ACTIVE else [])
        graph.update_node = AsyncMock()

        emb = AsyncMock()
        llm = AsyncMock()

        with patch("genesys.engine.transitions.calculate_decay_score", return_value=0.3):
            transitions = await evaluate_transitions(graph, emb, llm)

        assert len(transitions) == 1
        assert transitions[0]["new"] == "episodic"

    @pytest.mark.asyncio
    async def test_active_stays_if_below_threshold_count(self):
        """Active node with low score but counter < 3 stays active."""
        node = _make_node(status=MemoryStatus.ACTIVE, irrelevance_counter=0)
        graph = AsyncMock()
        graph.get_stats = AsyncMock(return_value={"max_causal_weight": 1})
        graph.get_nodes_by_status = AsyncMock(side_effect=lambda s, **kw: [node] if s == MemoryStatus.ACTIVE else [])
        graph.update_node = AsyncMock()

        with patch("genesys.engine.transitions.calculate_decay_score", return_value=0.3):
            transitions = await evaluate_transitions(graph, AsyncMock(), AsyncMock())

        assert len(transitions) == 0

    @pytest.mark.asyncio
    async def test_episodic_to_dormant(self):
        """Episodic node with very low score, inactive 90+ days, low reactivation → dormant."""
        node = _make_node(
            status=MemoryStatus.EPISODIC,
            last_reactivated_at=datetime.now(timezone.utc) - timedelta(days=100),
            reactivation_count=1,
        )
        graph = AsyncMock()
        graph.get_stats = AsyncMock(return_value={"max_causal_weight": 1})
        graph.get_nodes_by_status = AsyncMock(side_effect=lambda s, **kw: [node] if s == MemoryStatus.EPISODIC else [])
        graph.update_node = AsyncMock()

        with patch("genesys.engine.transitions.calculate_decay_score", return_value=0.05):
            transitions = await evaluate_transitions(graph, AsyncMock(), AsyncMock())

        assert len(transitions) == 1
        assert transitions[0]["new"] == "dormant"

    @pytest.mark.asyncio
    async def test_episodic_stays_if_recently_active(self):
        """Episodic node with low score but reactivated recently stays."""
        node = _make_node(
            status=MemoryStatus.EPISODIC,
            last_reactivated_at=datetime.now(timezone.utc) - timedelta(days=10),
            reactivation_count=1,
        )
        graph = AsyncMock()
        graph.get_stats = AsyncMock(return_value={"max_causal_weight": 1})
        graph.get_nodes_by_status = AsyncMock(side_effect=lambda s, **kw: [node] if s == MemoryStatus.EPISODIC else [])
        graph.update_node = AsyncMock()

        with patch("genesys.engine.transitions.calculate_decay_score", return_value=0.05):
            transitions = await evaluate_transitions(graph, AsyncMock(), AsyncMock())

        assert len(transitions) == 0
