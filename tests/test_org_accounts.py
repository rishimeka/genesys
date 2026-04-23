"""Tests for organizational account support.

Verifies visibility filtering, promote_to_org, auto-linking boundaries,
org node exemption from forgetting/transitions, and cross-boundary
edge visibility in traversal and causal chains.
"""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock

import pytest

from genesys_memory.context import current_org_ids, current_user_id
from genesys_memory.engine.forgetting import sweep_for_forgetting
from genesys_memory.mcp.tools import MCPToolHandler
from genesys_memory.models.edge import MemoryEdge
from genesys_memory.models.enums import EdgeType, MemoryStatus, Visibility
from genesys_memory.models.node import MemoryNode
from genesys_memory.storage.memory import InMemoryGraphProvider, InMemoryCacheProvider


def _make_node(**kwargs) -> MemoryNode:
    defaults = {
        "content_summary": "test memory",
        "content_full": "test memory content",
    }
    defaults.update(kwargs)
    return MemoryNode(**defaults)


def _make_edge(source_id: uuid.UUID, target_id: uuid.UUID, edge_type: EdgeType = EdgeType.RELATED_TO, weight: float = 0.7) -> MemoryEdge:
    return MemoryEdge(source_id=source_id, target_id=target_id, type=edge_type, weight=weight)


@pytest.fixture
def setup_two_users():
    """Set up two users: user-a (active) and user-b, sharing org-1."""
    graph = InMemoryGraphProvider()
    graph._user_nodes["user-a"] = {}
    graph._user_edges["user-a"] = []
    graph._user_nodes["user-b"] = {}
    graph._user_edges["user-b"] = []

    token_uid = current_user_id.set("user-a")
    token_org = current_org_ids.set(["org-1"])

    yield graph

    current_user_id.reset(token_uid)
    current_org_ids.reset(token_org)


class TestVisibilityFiltering:
    @pytest.mark.asyncio
    async def test_private_nodes_invisible_to_other_users(self, setup_two_users):
        graph = setup_two_users
        # User B creates a private node
        token = current_user_id.set("user-b")
        private_node = _make_node(content_summary="user-b secret")
        await graph.create_node(private_node)
        current_user_id.reset(token)

        # User A should not see it
        result = await graph.get_node(str(private_node.id))
        assert result is None

    @pytest.mark.asyncio
    async def test_org_nodes_visible_to_org_members(self, setup_two_users):
        graph = setup_two_users
        # User B creates an org node
        token = current_user_id.set("user-b")
        org_node = _make_node(
            content_summary="shared knowledge",
            visibility=Visibility.ORG,
            org_id="org-1",
        )
        await graph.create_node(org_node)
        current_user_id.reset(token)

        # User A (member of org-1) should see it
        result = await graph.get_node(str(org_node.id))
        assert result is not None
        assert result.content_summary == "shared knowledge"

    @pytest.mark.asyncio
    async def test_org_node_invisible_to_non_members(self, setup_two_users):
        graph = setup_two_users
        # User B creates a node in org-2
        token = current_user_id.set("user-b")
        org_node = _make_node(visibility=Visibility.ORG, org_id="org-2")
        await graph.create_node(org_node)
        current_user_id.reset(token)

        # User A (only in org-1) should not see it
        result = await graph.get_node(str(org_node.id))
        assert result is None

    @pytest.mark.asyncio
    async def test_vector_search_includes_org_nodes(self, setup_two_users):
        graph = setup_two_users
        embedding = [1.0] * 10

        # User B creates an org node with embedding
        token = current_user_id.set("user-b")
        org_node = _make_node(
            content_summary="org memory",
            embedding=embedding,
            visibility=Visibility.ORG,
            org_id="org-1",
        )
        await graph.create_node(org_node)
        current_user_id.reset(token)

        # User A's vector search should find it
        results = await graph.vector_search(embedding, k=10, org_ids=["org-1"])
        found_ids = {str(n.id) for n, _ in results}
        assert str(org_node.id) in found_ids

    @pytest.mark.asyncio
    async def test_vector_search_excludes_other_private(self, setup_two_users):
        graph = setup_two_users
        embedding = [1.0] * 10

        # User B creates a private node with embedding
        token = current_user_id.set("user-b")
        private_node = _make_node(embedding=embedding)
        await graph.create_node(private_node)
        current_user_id.reset(token)

        # User A should not find it
        results = await graph.vector_search(embedding, k=10, org_ids=["org-1"])
        found_ids = {str(n.id) for n, _ in results}
        assert str(private_node.id) not in found_ids

    @pytest.mark.asyncio
    async def test_keyword_search_includes_org_nodes(self, setup_two_users):
        graph = setup_two_users

        # User B creates an org node
        token = current_user_id.set("user-b")
        org_node = _make_node(
            content_summary="quarterly revenue report",
            content_full="quarterly revenue report details",
            visibility=Visibility.ORG,
            org_id="org-1",
        )
        await graph.create_node(org_node)
        current_user_id.reset(token)

        # User A should find it via keyword search
        results = await graph.keyword_search("revenue", k=10, org_ids=["org-1"])
        found_ids = {str(n.id) for n in results}
        assert str(org_node.id) in found_ids

    @pytest.mark.asyncio
    async def test_get_nodes_by_status_includes_org(self, setup_two_users):
        graph = setup_two_users

        # User B creates an org CORE node
        token = current_user_id.set("user-b")
        org_node = _make_node(
            status=MemoryStatus.CORE,
            visibility=Visibility.ORG,
            org_id="org-1",
        )
        await graph.create_node(org_node)
        current_user_id.reset(token)

        # User A should see it in CORE list
        cores = await graph.get_nodes_by_status(MemoryStatus.CORE)
        found_ids = {str(n.id) for n in cores}
        assert str(org_node.id) in found_ids


class TestTraversalVisibility:
    @pytest.mark.asyncio
    async def test_traverse_stops_at_private_boundary(self, setup_two_users):
        graph = setup_two_users

        # User B: org node -> private node chain
        token = current_user_id.set("user-b")
        org_node = _make_node(
            content_summary="org start",
            visibility=Visibility.ORG,
            org_id="org-1",
        )
        private_node = _make_node(content_summary="b secret")
        await graph.create_node(org_node)
        await graph.create_node(private_node)
        await graph.create_edge(_make_edge(org_node.id, private_node.id))
        current_user_id.reset(token)

        # User A traverses from org_node — should NOT reach private_node
        result = await graph.traverse(str(org_node.id), depth=3, org_ids=["org-1"])
        result_ids = {str(n.id) for n in result}
        assert str(org_node.id) in result_ids
        assert str(private_node.id) not in result_ids

    @pytest.mark.asyncio
    async def test_traverse_follows_org_to_org(self, setup_two_users):
        graph = setup_two_users

        # User B: two org nodes linked
        token = current_user_id.set("user-b")
        org_a = _make_node(content_summary="org a", visibility=Visibility.ORG, org_id="org-1")
        org_b = _make_node(content_summary="org b", visibility=Visibility.ORG, org_id="org-1")
        await graph.create_node(org_a)
        await graph.create_node(org_b)
        await graph.create_edge(_make_edge(org_a.id, org_b.id))
        current_user_id.reset(token)

        # User A can traverse org_a -> org_b
        result = await graph.traverse(str(org_a.id), depth=2, org_ids=["org-1"])
        result_ids = {str(n.id) for n in result}
        assert str(org_b.id) in result_ids

    @pytest.mark.asyncio
    async def test_causal_chain_respects_visibility(self, setup_two_users):
        graph = setup_two_users

        # User B: org node caused_by private node
        token = current_user_id.set("user-b")
        org_node = _make_node(content_summary="org effect", visibility=Visibility.ORG, org_id="org-1")
        private_cause = _make_node(content_summary="private cause")
        await graph.create_node(org_node)
        await graph.create_node(private_cause)
        await graph.create_edge(_make_edge(org_node.id, private_cause.id, EdgeType.CAUSED_BY))
        current_user_id.reset(token)

        # User A: causal chain upstream from org_node should NOT include private_cause
        chain = await graph.get_causal_chain(str(org_node.id), "upstream", org_ids=["org-1"])
        chain_ids = {str(n.id) for n in chain}
        assert str(private_cause.id) not in chain_ids


class TestPromoteToOrg:
    @pytest.mark.asyncio
    async def test_promote_sets_visibility_and_org_id(self, setup_two_users):
        graph = setup_two_users
        node = _make_node(content_summary="my insight")
        await graph.create_node(node)

        await graph.promote_to_org(str(node.id), "org-1")

        promoted = await graph.get_node(str(node.id))
        assert promoted is not None
        assert promoted.visibility == Visibility.ORG
        assert promoted.org_id == "org-1"
        assert promoted.original_user_id == "user-a"

    @pytest.mark.asyncio
    async def test_promote_via_tool_keep_private(self, setup_two_users):
        graph = setup_two_users
        cache = InMemoryCacheProvider()
        handler = MCPToolHandler(graph=graph, embeddings=None, cache=cache)

        # Create a node with a private linked node
        node_a = _make_node(content_summary="to promote")
        node_b = _make_node(content_summary="private linked")
        await graph.create_node(node_a)
        await graph.create_node(node_b)
        await graph.create_edge(_make_edge(node_a.id, node_b.id))

        result = await handler.promote_to_org(str(node_a.id), "org-1", action="keep_private")
        assert result["status"] == "promoted"
        assert len(result["edges_preserved"]) == 1

        # node_a is org, node_b stays private
        promoted = await graph.get_node(str(node_a.id))
        assert promoted.visibility == Visibility.ORG
        linked = await graph.get_node(str(node_b.id))
        assert linked.visibility == Visibility.PRIVATE

    @pytest.mark.asyncio
    async def test_promote_via_tool_delete_links(self, setup_two_users):
        graph = setup_two_users
        cache = InMemoryCacheProvider()
        handler = MCPToolHandler(graph=graph, embeddings=None, cache=cache)

        node_a = _make_node(content_summary="to promote")
        node_b = _make_node(content_summary="private linked")
        await graph.create_node(node_a)
        await graph.create_node(node_b)
        await graph.create_edge(_make_edge(node_a.id, node_b.id))

        result = await handler.promote_to_org(str(node_a.id), "org-1", action="delete_links")
        assert result["status"] == "promoted"

        # Edge should be gone
        edges = await graph.get_edges(str(node_a.id), "both")
        assert len(edges) == 0

    @pytest.mark.asyncio
    async def test_promote_via_tool_promote_all(self, setup_two_users):
        graph = setup_two_users
        cache = InMemoryCacheProvider()
        handler = MCPToolHandler(graph=graph, embeddings=None, cache=cache)

        node_a = _make_node(content_summary="to promote")
        node_b = _make_node(content_summary="linked, also promotable")
        await graph.create_node(node_a)
        await graph.create_node(node_b)
        await graph.create_edge(_make_edge(node_a.id, node_b.id))

        result = await handler.promote_to_org(str(node_a.id), "org-1", action="promote_all")
        assert result["status"] == "promoted"

        # Both should be org now
        a = await graph.get_node(str(node_a.id))
        b = await graph.get_node(str(node_b.id))
        assert a.visibility == Visibility.ORG
        assert b.visibility == Visibility.ORG


class TestOrgNodeForgetting:
    @pytest.mark.asyncio
    async def test_org_node_exempt_from_forgetting(self, setup_two_users):
        graph = setup_two_users
        org_node = _make_node(
            content_summary="org memory",
            visibility=Visibility.ORG,
            org_id="org-1",
            decay_score=0.0,
        )
        await graph.create_node(org_node)

        pruned = await sweep_for_forgetting(graph)
        assert str(org_node.id) not in pruned

        # Node should still exist
        still_there = await graph.get_node(str(org_node.id))
        assert still_there is not None

    @pytest.mark.asyncio
    async def test_private_orphan_still_pruned(self, setup_two_users):
        graph = setup_two_users
        private_node = _make_node(
            content_summary="private orphan",
            decay_score=0.0,
        )
        await graph.create_node(private_node)

        pruned = await sweep_for_forgetting(graph)
        assert str(private_node.id) in pruned


class TestAutoLinkingBoundary:
    @pytest.mark.asyncio
    async def test_org_store_only_links_to_same_org(self, setup_two_users):
        graph = setup_two_users
        cache = InMemoryCacheProvider()

        embedding = [1.0] * 10
        embeddings = AsyncMock()
        embeddings.embed = AsyncMock(return_value=embedding)

        handler = MCPToolHandler(graph=graph, embeddings=embeddings, cache=cache)

        # Create a private node with same embedding
        private_node = _make_node(content_summary="private similar", embedding=embedding)
        await graph.create_node(private_node)

        # Create an org node in org-1 with same embedding
        org_node = _make_node(
            content_summary="org similar",
            embedding=embedding,
            visibility=Visibility.ORG,
            org_id="org-1",
        )
        await graph.create_node(org_node)

        # Store a new org node — should only auto-link to the existing org node, not private
        result = await handler.memory_store(
            content="new org memory",
            visibility="org",
            org_id="org-1",
        )
        new_id = result["node_id"]

        edges = await graph.get_edges(new_id, "both")
        linked_ids = set()
        for e in edges:
            other = str(e.target_id) if str(e.source_id) == new_id else str(e.source_id)
            linked_ids.add(other)

        assert str(org_node.id) in linked_ids
        assert str(private_node.id) not in linked_ids


class TestPromoteDryRun:
    @pytest.mark.asyncio
    async def test_dry_run_keep_private_shows_preserved(self, setup_two_users):
        graph = setup_two_users
        cache = InMemoryCacheProvider()
        handler = MCPToolHandler(graph=graph, embeddings=None, cache=cache)

        node_a = _make_node(content_summary="to promote")
        node_b = _make_node(content_summary="private linked")
        await graph.create_node(node_a)
        await graph.create_node(node_b)
        await graph.create_edge(_make_edge(node_a.id, node_b.id))

        result = await handler.promote_to_org(str(node_a.id), "org-1", action="keep_private", dry_run=True)
        assert result["dry_run"] is True
        assert len(result["nodes_promoted"]) == 1
        assert result["nodes_promoted"][0]["id"] == str(node_a.id)
        assert len(result["edges_preserved"]) == 1
        assert result["edges_preserved"][0]["linked_summary"] == "private linked"
        assert len(result["edges_deleted"]) == 0

        # Nothing should have changed
        node = await graph.get_node(str(node_a.id))
        assert node.visibility == Visibility.PRIVATE

    @pytest.mark.asyncio
    async def test_dry_run_delete_links_shows_deleted(self, setup_two_users):
        graph = setup_two_users
        cache = InMemoryCacheProvider()
        handler = MCPToolHandler(graph=graph, embeddings=None, cache=cache)

        node_a = _make_node(content_summary="to promote")
        node_b = _make_node(content_summary="will lose edge")
        await graph.create_node(node_a)
        await graph.create_node(node_b)
        await graph.create_edge(_make_edge(node_a.id, node_b.id))

        result = await handler.promote_to_org(str(node_a.id), "org-1", action="delete_links", dry_run=True)
        assert result["dry_run"] is True
        assert len(result["edges_deleted"]) == 1
        assert result["edges_deleted"][0]["linked_summary"] == "will lose edge"

        # Edge should still exist
        edges = await graph.get_edges(str(node_a.id), "both")
        assert len(edges) == 1

    @pytest.mark.asyncio
    async def test_dry_run_promote_all_shows_cascade(self, setup_two_users):
        graph = setup_two_users
        cache = InMemoryCacheProvider()
        handler = MCPToolHandler(graph=graph, embeddings=None, cache=cache)

        node_a = _make_node(content_summary="root")
        node_b = _make_node(content_summary="will cascade")
        await graph.create_node(node_a)
        await graph.create_node(node_b)
        await graph.create_edge(_make_edge(node_a.id, node_b.id))

        result = await handler.promote_to_org(str(node_a.id), "org-1", action="promote_all", dry_run=True)
        assert result["dry_run"] is True
        assert len(result["nodes_promoted"]) == 2
        promoted_ids = {n["id"] for n in result["nodes_promoted"]}
        assert str(node_a.id) in promoted_ids
        assert str(node_b.id) in promoted_ids

        # Both should still be private
        a = await graph.get_node(str(node_a.id))
        b = await graph.get_node(str(node_b.id))
        assert a.visibility == Visibility.PRIVATE
        assert b.visibility == Visibility.PRIVATE

    @pytest.mark.asyncio
    async def test_dry_run_promote_all_shows_skipped(self, setup_two_users):
        graph = setup_two_users
        cache = InMemoryCacheProvider()
        handler = MCPToolHandler(graph=graph, embeddings=None, cache=cache)

        node_a = _make_node(content_summary="root")
        node_b = _make_node(content_summary="has deeper links")
        node_c = _make_node(content_summary="deep private")
        await graph.create_node(node_a)
        await graph.create_node(node_b)
        await graph.create_node(node_c)
        await graph.create_edge(_make_edge(node_a.id, node_b.id))
        await graph.create_edge(_make_edge(node_b.id, node_c.id))

        result = await handler.promote_to_org(str(node_a.id), "org-1", action="promote_all", dry_run=True)
        assert result["dry_run"] is True
        assert len(result["nodes_skipped"]) == 1
        assert result["nodes_skipped"][0]["reason"] == "has further private links"


class TestMemoryNodeOrgFields:
    def test_default_visibility_is_private(self):
        node = _make_node()
        assert node.visibility == Visibility.PRIVATE
        assert node.org_id is None
        assert node.original_user_id is None

    def test_org_fields_settable(self):
        node = _make_node(visibility=Visibility.ORG, org_id="org-1", original_user_id="user-a")
        assert node.visibility == Visibility.ORG
        assert node.org_id == "org-1"
        assert node.original_user_id == "user-a"
