"""Tests for Phase C: edge context, staleness, and co-retrieval validation."""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from genesys_memory.mcp.tools import MCPToolHandler, _is_edge_stale
from genesys_memory.models.edge import MemoryEdge
from genesys_memory.models.enums import EdgeType
from genesys_memory.models.node import MemoryNode
from genesys_memory.storage.memory import InMemoryGraphProvider


def _make_node(**kwargs) -> MemoryNode:
    defaults = {"content_summary": "test memory", "content_full": "test memory content"}
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


class TestIsEdgeStale:
    def test_fresh_edge_not_stale(self):
        edge = MemoryEdge(
            source_id=uuid.uuid4(),
            target_id=uuid.uuid4(),
            type=EdgeType.RELATED_TO,
            last_validated_at=datetime.now(timezone.utc),
        )
        assert _is_edge_stale(edge) is False

    def test_old_edge_is_stale(self):
        edge = MemoryEdge(
            source_id=uuid.uuid4(),
            target_id=uuid.uuid4(),
            type=EdgeType.RELATED_TO,
            last_validated_at=datetime.now(timezone.utc) - timedelta(days=45),
        )
        assert _is_edge_stale(edge) is True

    def test_edge_at_boundary_not_stale(self):
        edge = MemoryEdge(
            source_id=uuid.uuid4(),
            target_id=uuid.uuid4(),
            type=EdgeType.RELATED_TO,
            last_validated_at=datetime.now(timezone.utc) - timedelta(days=30),
        )
        assert _is_edge_stale(edge) is False

    def test_edge_just_past_boundary_is_stale(self):
        edge = MemoryEdge(
            source_id=uuid.uuid4(),
            target_id=uuid.uuid4(),
            type=EdgeType.RELATED_TO,
            last_validated_at=datetime.now(timezone.utc) - timedelta(days=31),
        )
        assert _is_edge_stale(edge) is True

    def test_none_validated_at_not_stale(self):
        edge = MemoryEdge(
            source_id=uuid.uuid4(),
            target_id=uuid.uuid4(),
            type=EdgeType.RELATED_TO,
        )
        edge.last_validated_at = None
        assert _is_edge_stale(edge) is False

    def test_custom_stale_days(self):
        edge = MemoryEdge(
            source_id=uuid.uuid4(),
            target_id=uuid.uuid4(),
            type=EdgeType.RELATED_TO,
            last_validated_at=datetime.now(timezone.utc) - timedelta(days=10),
        )
        with patch("genesys_memory.engine.config.EDGE_STALE_DAYS", 7):
            assert _is_edge_stale(edge) is True


class TestEdgeProvenance:
    def test_edge_has_source_context(self):
        edge = MemoryEdge(
            source_id=uuid.uuid4(),
            target_id=uuid.uuid4(),
            type=EdgeType.CAUSED_BY,
            source_context="user mentioned this during onboarding",
        )
        assert edge.source_context == "user mentioned this during onboarding"

    def test_edge_has_last_validated_at_default(self):
        edge = MemoryEdge(
            source_id=uuid.uuid4(),
            target_id=uuid.uuid4(),
            type=EdgeType.RELATED_TO,
        )
        assert edge.last_validated_at is not None
        assert (datetime.now(timezone.utc) - edge.last_validated_at).seconds < 5

    def test_edge_serializes_new_fields(self):
        edge = MemoryEdge(
            source_id=uuid.uuid4(),
            target_id=uuid.uuid4(),
            type=EdgeType.SUPPORTS,
            reason="high semantic overlap",
            created_by="auto_link",
            source_context="recalled together in session abc",
        )
        data = edge.model_dump(mode="json")
        assert data["reason"] == "high semantic overlap"
        assert data["created_by"] == "auto_link"
        assert data["source_context"] == "recalled together in session abc"
        assert data["last_validated_at"] is not None


class TestCoRetrievalValidation:
    @pytest.mark.asyncio
    async def test_recall_validates_shared_edges(self):
        handler, graph, embeddings, _ = _make_handler()

        node_a = _make_node(embedding=[0.1] * 1536)
        node_b = _make_node(embedding=[0.2] * 1536)

        edge = MemoryEdge(
            source_id=node_a.id,
            target_id=node_b.id,
            type=EdgeType.RELATED_TO,
            last_validated_at=datetime.now(timezone.utc) - timedelta(days=20),
        )

        graph.vector_search = AsyncMock(return_value=[(node_a, 0.9), (node_b, 0.85)])
        graph.keyword_search = AsyncMock(return_value=[])
        graph.get_causal_chain = AsyncMock(return_value=[])
        graph.get_nodes_by_status = AsyncMock(return_value=[])
        graph.get_edges = AsyncMock(return_value=[])
        graph.get_node = AsyncMock(side_effect=lambda nid: node_a if nid == str(node_a.id) else node_b)
        graph.get_all_edges = AsyncMock(return_value=[edge])
        graph.validate_edge = AsyncMock()

        with patch("genesys_memory.mcp.tools.current_org_ids") as mock_org:
            mock_org.get.return_value = []
            result = await handler.memory_recall("test query", k=5)

        assert result["count"] == 2
        graph.validate_edge.assert_called_once_with(str(edge.id))

    @pytest.mark.asyncio
    async def test_recall_skips_validation_in_read_only(self):
        handler, graph, embeddings, _ = _make_handler()

        node_a = _make_node(embedding=[0.1] * 1536)
        node_b = _make_node(embedding=[0.2] * 1536)

        edge = MemoryEdge(
            source_id=node_a.id,
            target_id=node_b.id,
            type=EdgeType.RELATED_TO,
        )

        graph.vector_search = AsyncMock(return_value=[(node_a, 0.9), (node_b, 0.85)])
        graph.keyword_search = AsyncMock(return_value=[])
        graph.get_causal_chain = AsyncMock(return_value=[])
        graph.get_nodes_by_status = AsyncMock(return_value=[])
        graph.get_edges = AsyncMock(return_value=[])
        graph.get_all_edges = AsyncMock(return_value=[edge])

        with patch("genesys_memory.mcp.tools.current_org_ids") as mock_org:
            mock_org.get.return_value = []
            result = await handler.memory_recall("test query", k=5, read_only=True)

        assert result["count"] == 2
        graph.validate_edge.assert_not_called()

    @pytest.mark.asyncio
    async def test_recall_only_validates_edges_between_recalled_nodes(self):
        handler, graph, embeddings, _ = _make_handler()

        node_a = _make_node(embedding=[0.1] * 1536)
        node_b = _make_node(embedding=[0.2] * 1536)
        node_c_id = uuid.uuid4()

        shared_edge = MemoryEdge(
            source_id=node_a.id,
            target_id=node_b.id,
            type=EdgeType.RELATED_TO,
        )
        external_edge = MemoryEdge(
            source_id=node_a.id,
            target_id=node_c_id,
            type=EdgeType.CAUSED_BY,
        )

        graph.vector_search = AsyncMock(return_value=[(node_a, 0.9), (node_b, 0.85)])
        graph.keyword_search = AsyncMock(return_value=[])
        graph.get_causal_chain = AsyncMock(return_value=[])
        graph.get_nodes_by_status = AsyncMock(return_value=[])
        graph.get_edges = AsyncMock(return_value=[])
        graph.get_node = AsyncMock(side_effect=lambda nid: node_a if nid == str(node_a.id) else node_b)
        graph.get_all_edges = AsyncMock(return_value=[shared_edge, external_edge])
        graph.validate_edge = AsyncMock()

        with patch("genesys_memory.mcp.tools.current_org_ids") as mock_org:
            mock_org.get.return_value = []
            await handler.memory_recall("test query", k=5)

        graph.validate_edge.assert_called_once_with(str(shared_edge.id))


class TestExplainShowsStaleness:
    @pytest.mark.asyncio
    async def test_explain_includes_staleness_info(self):
        handler, graph, _, _ = _make_handler()
        node = _make_node(decay_score=0.85, reactivation_count=3)

        stale_edge = MemoryEdge(
            source_id=node.id,
            target_id=uuid.uuid4(),
            type=EdgeType.RELATED_TO,
            reason="cosine similarity 0.812",
            created_by="auto_link",
            last_validated_at=datetime.now(timezone.utc) - timedelta(days=45),
        )
        fresh_edge = MemoryEdge(
            source_id=node.id,
            target_id=uuid.uuid4(),
            type=EdgeType.CAUSED_BY,
            reason="user shared causal link",
            created_by="user_explicit",
            last_validated_at=datetime.now(timezone.utc),
        )

        graph.get_node = AsyncMock(return_value=node)
        graph.get_causal_weight = AsyncMock(return_value=2)
        graph.is_orphan = AsyncMock(return_value=False)
        graph.get_causal_chain = AsyncMock(return_value=[])
        graph.get_edges = AsyncMock(return_value=[stale_edge, fresh_edge])

        result = await handler.memory_explain(str(node.id))

        assert len(result["edges"]) == 2
        edges = result["edges"]
        stale = [e for e in edges if e["stale"]]
        fresh = [e for e in edges if not e["stale"]]
        assert len(stale) == 1
        assert len(fresh) == 1
        assert stale[0]["reason"] == "cosine similarity 0.812"
        assert stale[0]["created_by"] == "auto_link"
        assert stale[0]["last_validated_at"] is not None
        assert fresh[0]["created_by"] == "user_explicit"


class TestInMemoryValidateEdge:
    @pytest.mark.asyncio
    async def test_validate_edge_updates_timestamp(self):
        from genesys_memory.context import current_user_id
        token = current_user_id.set("test-user")
        try:
            provider = InMemoryGraphProvider()
            node_a = _make_node()
            node_b = _make_node()
            await provider.create_node(node_a)
            await provider.create_node(node_b)

            old_time = datetime.now(timezone.utc) - timedelta(days=40)
            edge = MemoryEdge(
                source_id=node_a.id,
                target_id=node_b.id,
                type=EdgeType.RELATED_TO,
                last_validated_at=old_time,
            )
            await provider.create_edge(edge)

            assert _is_edge_stale(edge) is True

            await provider.validate_edge(str(edge.id))

            edges = await provider.get_all_edges()
            validated_edge = edges[0]
            assert (datetime.now(timezone.utc) - validated_edge.last_validated_at).seconds < 5
            assert _is_edge_stale(validated_edge) is False
        finally:
            current_user_id.reset(token)

    @pytest.mark.asyncio
    async def test_validate_nonexistent_edge_noop(self):
        from genesys_memory.context import current_user_id
        token = current_user_id.set("test-user")
        try:
            provider = InMemoryGraphProvider()
            await provider.validate_edge("nonexistent-id")
        finally:
            current_user_id.reset(token)


class TestEdgePersistence:
    @pytest.mark.asyncio
    async def test_new_fields_survive_serialization(self):
        """source_context and last_validated_at survive JSON round-trip via persist."""
        import tempfile
        from genesys_memory.context import current_user_id

        token = current_user_id.set("test-user")
        try:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                path = f.name

            provider = InMemoryGraphProvider(persist_path=path)
            node = _make_node()
            await provider.create_node(node)

            ts = datetime.now(timezone.utc) - timedelta(days=5)
            edge = MemoryEdge(
                source_id=node.id,
                target_id=node.id,
                type=EdgeType.RELATED_TO,
                reason="test reason",
                created_by="auto_link",
                source_context="recalled in session xyz",
                last_validated_at=ts,
            )
            await provider.create_edge(edge)

            provider2 = InMemoryGraphProvider(persist_path=path)
            provider2._load()
            edges = provider2._user_edges.get("test-user", [])
            assert len(edges) == 1
            loaded = edges[0]
            assert loaded.reason == "test reason"
            assert loaded.created_by == "auto_link"
            assert loaded.source_context == "recalled in session xyz"
            assert loaded.last_validated_at is not None
        finally:
            current_user_id.reset(token)
