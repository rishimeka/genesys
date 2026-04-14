"""Tests for Phase 3 MCP tools."""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from genesys.mcp.tools import MCPToolHandler
from genesys.models.edge import MemoryEdge
from genesys.models.enums import EdgeType, MemoryStatus, ReactivationPattern
from genesys.models.node import MemoryNode


def _make_node(**kwargs) -> MemoryNode:
    defaults = {"content_summary": "test memory"}
    defaults.update(kwargs)
    return MemoryNode(**defaults)


def _make_handler() -> tuple[MCPToolHandler, AsyncMock, AsyncMock, AsyncMock]:
    graph = AsyncMock()
    embeddings = AsyncMock()
    cache = AsyncMock()
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock()
    embeddings.embed = AsyncMock(return_value=[0.1] * 1536)
    handler = MCPToolHandler(graph=graph, embeddings=embeddings, cache=cache)
    return handler, graph, embeddings, cache


class TestAllToolsRegistered:
    @pytest.mark.asyncio
    async def test_all_tools_registered(self):
        """Server should list 11 tools."""
        # Import and check tool listing
        from genesys.server import list_tools
        tool_list = await list_tools()
        assert len(tool_list) == 11
        names = {t.name for t in tool_list}
        expected = {
            "memory_store", "memory_recall", "memory_search", "memory_traverse",
            "memory_explain", "pin_memory", "unpin_memory", "list_core_memories",
            "delete_memory", "memory_stats", "set_core_preferences",
        }
        assert names == expected


class TestPinUnpinRoundtrip:
    @pytest.mark.asyncio
    async def test_pin_sets_core(self):
        handler, graph, _, _ = _make_handler()
        node = _make_node(status=MemoryStatus.ACTIVE)
        graph.get_node = AsyncMock(return_value=node)

        result = await handler.pin_memory(str(node.id))
        assert result["status"] == "pinned"
        assert result["new_status"] == "core"
        graph.update_node.assert_called_once()
        call_updates = graph.update_node.call_args[0][1]
        assert call_updates["pinned"] is True
        assert call_updates["status"] == MemoryStatus.CORE

    @pytest.mark.asyncio
    async def test_unpin_reevaluates(self):
        handler, graph, _, _ = _make_handler()
        node = _make_node(status=MemoryStatus.CORE, pinned=True)

        # After unpin, get_node returns unpinned node
        unpinned_node = _make_node(status=MemoryStatus.CORE, pinned=False, category=None)
        graph.get_node = AsyncMock(side_effect=[node, unpinned_node])
        graph.get_causal_weight = AsyncMock(return_value=0)
        graph.get_causal_chain = AsyncMock(return_value=[])

        result = await handler.unpin_memory(str(node.id))
        assert result["status"] == "unpinned"
        # Should demote since no promotion criteria met
        assert result["new_status"] == "active"

    @pytest.mark.asyncio
    async def test_pin_not_found(self):
        handler, graph, _, _ = _make_handler()
        graph.get_node = AsyncMock(return_value=None)
        result = await handler.pin_memory("nonexistent")
        assert "error" in result


class TestMemoryExplain:
    @pytest.mark.asyncio
    async def test_explain_structure(self):
        handler, graph, _, _ = _make_handler()
        node = _make_node(
            decay_score=0.85,
            reactivation_count=5,
            reactivation_pattern=ReactivationPattern.STEADY,
        )
        graph.get_node = AsyncMock(return_value=node)
        graph.get_causal_weight = AsyncMock(return_value=3)
        graph.is_orphan = AsyncMock(return_value=False)
        graph.get_causal_chain = AsyncMock(return_value=[_make_node()])
        graph.get_edges = AsyncMock(return_value=[])

        result = await handler.memory_explain(str(node.id))

        assert result["node_id"] == str(node.id)
        assert result["summary"] == "test memory"
        assert result["decay_score"] == 0.85
        assert result["causal_weight"] == 3
        assert result["reactivation_count"] == 5
        assert result["reactivation_pattern"] == "steady"
        assert result["pinned"] is False
        assert result["is_orphan"] is False
        assert result["upstream_count"] == 1
        assert result["downstream_count"] == 1
        assert "removal_impact" in result

    @pytest.mark.asyncio
    async def test_explain_not_found(self):
        handler, graph, _, _ = _make_handler()
        graph.get_node = AsyncMock(return_value=None)
        result = await handler.memory_explain("missing")
        assert "error" in result


class TestDeleteMemory:
    @pytest.mark.asyncio
    async def test_delete_removes_node(self):
        handler, graph, _, _ = _make_handler()
        node = _make_node()
        graph.get_node = AsyncMock(return_value=node)

        result = await handler.delete_memory(str(node.id))
        assert result["status"] == "deleted"
        graph.delete_node.assert_called_once_with(str(node.id))

    @pytest.mark.asyncio
    async def test_delete_not_found(self):
        handler, graph, _, _ = _make_handler()
        graph.get_node = AsyncMock(return_value=None)
        result = await handler.delete_memory("missing")
        assert "error" in result


class TestMemorySearch:
    @pytest.mark.asyncio
    async def test_search_with_status_filter(self):
        handler, graph, embeddings, _ = _make_handler()
        node = _make_node(status=MemoryStatus.CORE, category="professional")
        graph.vector_search = AsyncMock(return_value=[(node, 0.9)])

        result = await handler.memory_search("test", filters={"status": ["core"]}, k=5)
        assert result["count"] == 1
        assert result["results"][0]["status"] == "core"

    @pytest.mark.asyncio
    async def test_search_with_category_filter(self):
        handler, graph, _, _ = _make_handler()
        node1 = _make_node(category="work")
        node2 = _make_node(category="personal")
        graph.vector_search = AsyncMock(return_value=[(node1, 0.9), (node2, 0.8)])

        result = await handler.memory_search("test", filters={"category": "work"})
        assert result["count"] == 1
        assert result["results"][0]["category"] == "work"


class TestListCoreMemories:
    @pytest.mark.asyncio
    async def test_list_sorted_by_causal_weight(self):
        handler, graph, _, _ = _make_handler()
        n1 = _make_node(status=MemoryStatus.CORE, causal_weight=5, content_summary="low")
        n2 = _make_node(status=MemoryStatus.CORE, causal_weight=15, content_summary="high")
        graph.get_nodes_by_status = AsyncMock(return_value=[n1, n2])

        result = await handler.list_core_memories()
        assert result["count"] == 2
        assert result["core_memories"][0]["summary"] == "high"
        assert result["core_memories"][1]["summary"] == "low"

    @pytest.mark.asyncio
    async def test_list_filtered_by_category(self):
        handler, graph, _, _ = _make_handler()
        n1 = _make_node(status=MemoryStatus.CORE, category="work")
        n2 = _make_node(status=MemoryStatus.CORE, category="personal")
        graph.get_nodes_by_status = AsyncMock(return_value=[n1, n2])

        result = await handler.list_core_memories(category="work")
        assert result["count"] == 1


class TestMemoryStats:
    @pytest.mark.asyncio
    async def test_stats_returns_dict(self):
        handler, graph, _, _ = _make_handler()
        graph.get_stats = AsyncMock(return_value={
            "node_count": 10,
            "edge_count": 5,
            "node_count_by_status": {"active": 8, "core": 2},
            "edge_count_by_type": {"CAUSED_BY": 3, "SUPPORTS": 2},
            "orphan_count": 1,
        })

        result = await handler.memory_stats()
        assert result["node_count"] == 10
        assert result["edge_count"] == 5


class TestSetCorePreferences:
    @pytest.mark.asyncio
    async def test_set_preferences(self):
        handler, graph, _, cache = _make_handler()
        result = await handler.set_core_preferences(
            auto=["work", "health"],
            excluded=["trivia"],
        )
        assert result["status"] == "updated"
        assert result["preferences"]["auto"] == ["work", "health"]
        assert result["preferences"]["excluded"] == ["trivia"]
