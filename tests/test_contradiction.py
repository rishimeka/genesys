"""Tests for contradiction detection."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from genesys_memory.engine.contradiction import detect_contradictions
from genesys_memory.models.enums import MemoryStatus
from genesys_memory.models.node import MemoryNode


def _make_node(**kwargs) -> MemoryNode:
    defaults = {
        "content_summary": "test",
        "content_full": "test content",
        "embedding": [0.1] * 1536,
    }
    defaults.update(kwargs)
    return MemoryNode(**defaults)


class TestContradiction:
    @pytest.mark.asyncio
    async def test_contradiction_detected(self):
        """Two contradicting memories should create a CONTRADICTS edge."""
        new_node = _make_node(content_full="I work at Meta")
        existing = _make_node(content_full="I work at Google")

        graph = AsyncMock()
        # Return existing node with high similarity (low distance)
        graph.vector_search = AsyncMock(return_value=[(existing, 0.05)])  # 0.05 distance = 0.95 similarity
        graph.create_edge = AsyncMock()
        graph.update_node = AsyncMock()

        emb = AsyncMock()
        llm = AsyncMock()
        llm.detect_contradiction = AsyncMock(return_value=(True, 0.95, "Direct contradiction about employer"))

        result = await detect_contradictions(new_node, graph, emb, llm)
        assert len(result) == 1
        assert result[0][1] == 0.95
        # Verify reason is stored on the edge
        edge_arg = graph.create_edge.call_args[0][0]
        assert edge_arg.reason == "Direct contradiction about employer"
        assert edge_arg.created_by == "llm_contradiction"

    @pytest.mark.asyncio
    async def test_no_contradiction_low_confidence(self):
        """LLM says no contradiction → no edge created."""
        new_node = _make_node(content_full="I like Python")
        existing = _make_node(content_full="I like JavaScript")

        graph = AsyncMock()
        graph.vector_search = AsyncMock(return_value=[(existing, 0.1)])
        graph.create_edge = AsyncMock()

        llm = AsyncMock()
        llm.detect_contradiction = AsyncMock(return_value=(False, 0.2, None))

        result = await detect_contradictions(new_node, graph, AsyncMock(), llm)
        assert len(result) == 0
        graph.create_edge.assert_not_called()

    @pytest.mark.asyncio
    async def test_contradiction_supersedes_core(self):
        """Contradicting a core memory should trigger supersession."""
        new_node = _make_node(content_full="I work at Meta")
        core_node = _make_node(
            content_full="I work at Google",
            status=MemoryStatus.CORE,
        )

        graph = AsyncMock()
        graph.vector_search = AsyncMock(return_value=[(core_node, 0.05)])
        graph.create_edge = AsyncMock()
        graph.update_node = AsyncMock()

        llm = AsyncMock()
        llm.detect_contradiction = AsyncMock(return_value=(True, 0.9, "Employment changed"))

        result = await detect_contradictions(new_node, graph, AsyncMock(), llm)
        assert len(result) == 1

        # Should create both CONTRADICTS and SUPERSEDES edges
        assert graph.create_edge.call_count == 2
        # Core node should be demoted to episodic
        graph.update_node.assert_called_once()
        update_args = graph.update_node.call_args[0]
        assert update_args[1]["status"] == MemoryStatus.EPISODIC

    @pytest.mark.asyncio
    async def test_no_contradiction_low_similarity(self):
        """Memories with low similarity shouldn't even be checked by LLM."""
        new_node = _make_node(content_full="I like cooking")
        unrelated = _make_node(content_full="The weather is nice")

        graph = AsyncMock()
        graph.vector_search = AsyncMock(return_value=[(unrelated, 0.5)])  # 0.5 distance = 0.5 similarity
        graph.create_edge = AsyncMock()

        llm = AsyncMock()

        result = await detect_contradictions(new_node, graph, AsyncMock(), llm)
        assert len(result) == 0
        llm.detect_contradiction.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_embedding_returns_empty(self):
        """Node without embedding should return empty."""
        new_node = _make_node(embedding=None)
        graph = AsyncMock()

        result = await detect_contradictions(new_node, graph, AsyncMock(), AsyncMock())
        assert result == []
