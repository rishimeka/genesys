"""Phase 1 security tests: authorization, ownership, input bounds.

Tests org_id authorization, node ownership checks, related_to visibility,
MCP server dispatch, parameter capping, fail-closed ownership, and admin role override.
"""
from __future__ import annotations

import logging
import uuid
from unittest.mock import AsyncMock

import pytest

from genesys_memory.context import current_org_ids, current_user_id, current_user_role
from genesys_memory.mcp.tools import MCPToolHandler, _caller_owns_node
from genesys_memory.models.edge import MemoryEdge
from genesys_memory.models.enums import EdgeType, MemoryStatus, Visibility
from genesys_memory.models.node import MemoryNode
from genesys_memory.storage.memory import InMemoryCacheProvider, InMemoryGraphProvider


def _make_node(**kwargs) -> MemoryNode:
    defaults = {
        "content_summary": "test memory",
        "content_full": "test memory content",
    }
    defaults.update(kwargs)
    return MemoryNode(**defaults)


def _make_edge(
    source_id: uuid.UUID,
    target_id: uuid.UUID,
    edge_type: EdgeType = EdgeType.RELATED_TO,
    weight: float = 0.7,
) -> MemoryEdge:
    return MemoryEdge(source_id=source_id, target_id=target_id, type=edge_type, weight=weight)


@pytest.fixture
def two_user_setup():
    """Two users (user-a in org-1, user-b in org-1). user-a is active caller."""
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


@pytest.fixture
def handler_with_graph(two_user_setup):
    """MCPToolHandler wired to the two-user graph with a mock embedder."""
    graph = two_user_setup
    cache = InMemoryCacheProvider()
    embeddings = AsyncMock()
    embeddings.embed = AsyncMock(return_value=[1.0] * 10)
    handler = MCPToolHandler(graph=graph, embeddings=embeddings, cache=cache)
    return handler, graph


# ────────────────────────────────────────────────────────────────────
# 1.1 — Org authorization on memory_store and promote_to_org
# ────────────────────────────────────────────────────────────────────

class TestMemoryStoreOrgAuthorization:
    @pytest.mark.asyncio
    async def test_memory_store_rejects_foreign_org_id(self, handler_with_graph):
        """User in org-1 cannot store a memory in org-2."""
        handler, graph = handler_with_graph
        result = await handler.memory_store(
            content="should be rejected",
            visibility="org",
            org_id="org-2",
        )
        assert "error" in result
        assert "org memberships" in result["error"]

    @pytest.mark.asyncio
    async def test_memory_store_accepts_own_org_id(self, handler_with_graph):
        """User in org-1 can store a memory in org-1."""
        handler, graph = handler_with_graph
        result = await handler.memory_store(
            content="org memory",
            visibility="org",
            org_id="org-1",
        )
        assert "error" not in result
        assert result["status"] == "stored"
        assert result["visibility"] == "org"

    @pytest.mark.asyncio
    async def test_memory_store_private_does_not_check_org(self, handler_with_graph):
        """Private memories don't need org_id validation."""
        handler, graph = handler_with_graph
        result = await handler.memory_store(
            content="private memory",
            visibility="private",
        )
        assert "error" not in result
        assert result["status"] == "stored"


class TestPromoteToOrgAuthorization:
    @pytest.mark.asyncio
    async def test_promote_to_org_rejects_foreign_org_id(self, handler_with_graph):
        """User in org-1 cannot promote to org-2."""
        handler, graph = handler_with_graph
        node = _make_node(content_summary="my insight")
        await graph.create_node(node)

        result = await handler.promote_to_org(str(node.id), "org-2")
        assert "error" in result
        assert "org memberships" in result["error"]

    @pytest.mark.asyncio
    async def test_promote_to_org_rejects_non_owner(self, handler_with_graph):
        """User A cannot promote user B's private node even in shared org."""
        handler, graph = handler_with_graph

        # User B creates a private node
        token = current_user_id.set("user-b")
        node_b = _make_node(content_summary="user-b private")
        await graph.create_node(node_b)
        # Promote it to org-1 so user-a can see it
        await graph.promote_to_org(str(node_b.id), "org-1")
        current_user_id.reset(token)

        # Now make it private again to test the ownership check path
        # Actually the node is already org-visible so promote_to_org will say "Already org-visible"
        # Instead test: user-a creates a node, user-b tries to promote it
        token_b = current_user_id.set("user-b")
        token_org_b = current_org_ids.set(["org-1"])

        # user-a created a private node
        current_user_id.reset(token_b)
        current_org_ids.reset(token_org_b)

        # As user-a, create a private node
        node_a = _make_node(content_summary="user-a private")
        await graph.create_node(node_a)

        # Now switch to user-b context and try to promote user-a's node
        token_b2 = current_user_id.set("user-b")
        token_org_b2 = current_org_ids.set(["org-1"])

        # user-b can see user-a's node only if it's org-visible; private nodes
        # are invisible. So user-b trying to promote user-a's private node
        # will get "Node not found" (can't even see it). That's the correct behavior —
        # you can't promote what you can't see.
        result = await handler.promote_to_org(str(node_a.id), "org-1")
        assert "error" in result

        current_user_id.reset(token_b2)
        current_org_ids.reset(token_org_b2)

    @pytest.mark.asyncio
    async def test_promote_to_org_succeeds_for_owner_in_org(self, handler_with_graph):
        """Node owner in the target org can promote."""
        handler, graph = handler_with_graph
        node = _make_node(content_summary="promotable")
        await graph.create_node(node)

        result = await handler.promote_to_org(str(node.id), "org-1")
        assert result.get("status") == "promoted"

    @pytest.mark.asyncio
    async def test_promote_rejects_non_owner_of_org_visible_node(self, handler_with_graph):
        """User A created an org node; user B cannot re-promote it (already org-visible check)."""
        handler, graph = handler_with_graph

        # User A creates a node and promotes it
        node = _make_node(content_summary="already promoted")
        await graph.create_node(node)
        await graph.promote_to_org(str(node.id), "org-1")

        # Switch to user B
        token_b = current_user_id.set("user-b")
        token_org_b = current_org_ids.set(["org-1"])

        # User B can see the org node but shouldn't be able to promote it elsewhere
        result = await handler.promote_to_org(str(node.id), "org-1")
        assert "error" in result
        assert "Already org-visible" in result["error"]

        current_user_id.reset(token_b)
        current_org_ids.reset(token_org_b)


# ────────────────────────────────────────────────────────────────────
# 1.2 — Ownership checks on node mutations
# ────────────────────────────────────────────────────────────────────

class TestOwnershipChecks:
    @pytest.mark.asyncio
    async def test_pin_memory_rejects_non_owner(self, handler_with_graph):
        """User B cannot pin user A's org-visible node."""
        handler, graph = handler_with_graph

        # User A creates and promotes a node
        node = _make_node(content_summary="org knowledge")
        await graph.create_node(node)
        await graph.promote_to_org(str(node.id), "org-1")

        # Switch to user B
        token_b = current_user_id.set("user-b")
        token_org_b = current_org_ids.set(["org-1"])

        result = await handler.pin_memory(str(node.id))
        assert "error" in result
        assert "owner" in result["error"].lower()

        current_user_id.reset(token_b)
        current_org_ids.reset(token_org_b)

    @pytest.mark.asyncio
    async def test_delete_memory_rejects_non_owner_in_org(self, handler_with_graph):
        """User B cannot delete user A's org-visible node."""
        handler, graph = handler_with_graph

        node = _make_node(content_summary="org knowledge to protect")
        await graph.create_node(node)
        await graph.promote_to_org(str(node.id), "org-1")

        # Switch to user B
        token_b = current_user_id.set("user-b")
        token_org_b = current_org_ids.set(["org-1"])

        result = await handler.delete_memory(str(node.id))
        assert "error" in result
        assert "owner" in result["error"].lower()

        # Verify node still exists
        current_user_id.reset(token_b)
        current_org_ids.reset(token_org_b)

        still_there = await graph.get_node(str(node.id))
        assert still_there is not None

    @pytest.mark.asyncio
    async def test_unpin_memory_rejects_non_owner(self, handler_with_graph):
        """User B cannot unpin user A's pinned org node."""
        handler, graph = handler_with_graph

        node = _make_node(content_summary="pinned org node", pinned=True, status=MemoryStatus.CORE)
        await graph.create_node(node)
        await graph.promote_to_org(str(node.id), "org-1")

        # Switch to user B
        token_b = current_user_id.set("user-b")
        token_org_b = current_org_ids.set(["org-1"])

        result = await handler.unpin_memory(str(node.id))
        assert "error" in result
        assert "owner" in result["error"].lower()

        current_user_id.reset(token_b)
        current_org_ids.reset(token_org_b)

    @pytest.mark.asyncio
    async def test_owner_can_delete_own_node(self, handler_with_graph):
        """Node owner can delete their own node."""
        handler, graph = handler_with_graph
        node = _make_node(content_summary="deletable")
        await graph.create_node(node)

        result = await handler.delete_memory(str(node.id))
        assert result["status"] == "deleted"

    @pytest.mark.asyncio
    async def test_owner_can_pin_own_node(self, handler_with_graph):
        """Node owner can pin their own node."""
        handler, graph = handler_with_graph
        node = _make_node(content_summary="pinnable")
        await graph.create_node(node)

        result = await handler.pin_memory(str(node.id))
        assert result["status"] == "pinned"

    @pytest.mark.asyncio
    async def test_explain_does_not_require_ownership(self, handler_with_graph):
        """memory_explain is read-only and should work for any visible node."""
        handler, graph = handler_with_graph

        node = _make_node(content_summary="org knowledge")
        await graph.create_node(node)
        await graph.promote_to_org(str(node.id), "org-1")

        # Switch to user B
        token_b = current_user_id.set("user-b")
        token_org_b = current_org_ids.set(["org-1"])

        result = await handler.memory_explain(str(node.id))
        assert "error" not in result
        assert result["node_id"] == str(node.id)

        current_user_id.reset(token_b)
        current_org_ids.reset(token_org_b)


# ────────────────────────────────────────────────────────────────────
# 1.3 — Visibility check on related_to edges
# ────────────────────────────────────────────────────────────────────

class TestRelatedToVisibility:
    @pytest.mark.asyncio
    async def test_related_to_skips_invisible_target(self, handler_with_graph, caplog):
        """related_to with a node_id not visible to caller creates no edge and logs."""
        handler, graph = handler_with_graph

        # User B creates a private node
        token_b = current_user_id.set("user-b")
        private_node = _make_node(content_summary="user-b secret")
        await graph.create_node(private_node)
        current_user_id.reset(token_b)

        # User A stores a memory with related_to pointing at user B's private node
        with caplog.at_level(logging.WARNING, logger="genesys_memory.mcp.tools"):
            result = await handler.memory_store(
                content="should not link to b's node",
                related_to=[str(private_node.id)],
            )

        # Node should be created
        assert "error" not in result
        new_id = result["node_id"]

        # But no edge to the private node
        edges = await graph.get_edges(new_id, "both")
        linked_ids = {
            str(e.target_id) if str(e.source_id) == new_id else str(e.source_id)
            for e in edges
            if e.created_by == "user_explicit"
        }
        assert str(private_node.id) not in linked_ids

        # Warning should have been logged
        assert "not visible to caller" in caplog.text

    @pytest.mark.asyncio
    async def test_related_to_succeeds_for_visible_target(self, handler_with_graph):
        """related_to with a visible node_id creates the edge normally."""
        handler, graph = handler_with_graph

        target = _make_node(content_summary="visible target")
        await graph.create_node(target)

        result = await handler.memory_store(
            content="linked memory",
            related_to=[str(target.id)],
        )
        assert "error" not in result
        new_id = result["node_id"]

        edges = await graph.get_edges(new_id, "both")
        linked_ids = {
            str(e.target_id) if str(e.source_id) == new_id else str(e.source_id)
            for e in edges
            if e.created_by == "user_explicit"
        }
        assert str(target.id) in linked_ids


# ────────────────────────────────────────────────────────────────────
# 1.4 — MCP server dispatch includes promote_to_org and visibility
# ────────────────────────────────────────────────────────────────────

class TestMCPServerDispatch:
    @pytest.mark.asyncio
    async def test_mcp_server_promote_to_org_exposed(self):
        """promote_to_org should be in the tool list."""
        from genesys_memory.server import list_tools
        tool_list = await list_tools()
        names = {t.name for t in tool_list}
        assert "promote_to_org" in names

    @pytest.mark.asyncio
    async def test_mcp_server_memory_store_accepts_visibility(self):
        """memory_store schema should include visibility and org_id."""
        from genesys_memory.server import list_tools
        tool_list = await list_tools()
        store_tool = next(t for t in tool_list if t.name == "memory_store")
        props = store_tool.inputSchema["properties"]
        assert "visibility" in props
        assert "org_id" in props

    @pytest.mark.asyncio
    async def test_mcp_server_promote_schema_has_required_fields(self):
        """promote_to_org schema should have node_id and org_id as required."""
        from genesys_memory.server import list_tools
        tool_list = await list_tools()
        promote_tool = next(t for t in tool_list if t.name == "promote_to_org")
        assert "node_id" in promote_tool.inputSchema["required"]
        assert "org_id" in promote_tool.inputSchema["required"]
        props = promote_tool.inputSchema["properties"]
        assert "action" in props
        assert "dry_run" in props

    @pytest.mark.asyncio
    async def test_mcp_dispatch_passes_visibility_through(self):
        """The dispatch wiring should actually pass visibility to the handler."""
        from genesys_memory.server import _TOOL_DISPATCH
        _, required, optional = _TOOL_DISPATCH["memory_store"]
        assert "visibility" in optional
        assert "org_id" in optional


# ────────────────────────────────────────────────────────────────────
# 1.7 — Cap k and depth parameters
# ────────────────────────────────────────────────────────────────────

class TestParameterCapping:
    @pytest.mark.asyncio
    async def test_memory_recall_caps_k_at_100(self, handler_with_graph, caplog):
        """k > 100 should be capped to 100."""
        handler, graph = handler_with_graph

        with caplog.at_level(logging.INFO, logger="genesys_memory.mcp.tools"):
            result = await handler.memory_recall(query="test", k=500)

        assert "capped" in caplog.text
        # The result should still work (just capped)
        assert "results" in result

    @pytest.mark.asyncio
    async def test_memory_recall_normal_k_not_capped(self, handler_with_graph, caplog):
        """k <= 100 should not be capped or logged."""
        handler, graph = handler_with_graph

        with caplog.at_level(logging.INFO, logger="genesys_memory.mcp.tools"):
            result = await handler.memory_recall(query="test", k=10)

        assert "capped" not in caplog.text
        assert "results" in result

    @pytest.mark.asyncio
    async def test_memory_traverse_caps_depth_at_10(self, handler_with_graph, caplog):
        """depth > 10 should be capped to 10."""
        handler, graph = handler_with_graph

        node = _make_node(content_summary="start node")
        await graph.create_node(node)

        with caplog.at_level(logging.INFO, logger="genesys_memory.mcp.tools"):
            result = await handler.memory_traverse(node_id=str(node.id), depth=50)

        assert "capped" in caplog.text
        assert "nodes" in result

    @pytest.mark.asyncio
    async def test_memory_traverse_normal_depth_not_capped(self, handler_with_graph, caplog):
        """depth <= 10 should not be capped or logged."""
        handler, graph = handler_with_graph

        node = _make_node(content_summary="start node")
        await graph.create_node(node)

        with caplog.at_level(logging.INFO, logger="genesys_memory.mcp.tools"):
            result = await handler.memory_traverse(node_id=str(node.id), depth=3)

        assert "capped" not in caplog.text
        assert "nodes" in result


# ────────────────────────────────────────────────────────────────────
# 1.2a — Fail-closed ownership: no user context raises PermissionError
# ───────────────────���────────────────────────────────────────────────

class TestFailClosedOwnership:
    def test_no_user_context_raises_permission_error(self):
        """_caller_owns_node must raise PermissionError when current_user_id is unset."""
        node = _make_node(content_summary="any", original_user_id="someone")
        with pytest.raises(PermissionError, match="current_user_id not set"):
            _caller_owns_node(node)

    def test_with_user_context_does_not_raise(self):
        """_caller_owns_node with user context returns a bool, not raises."""
        token = current_user_id.set("user-a")
        try:
            node = _make_node(content_summary="any", original_user_id="user-a")
            assert _caller_owns_node(node) is True
        finally:
            current_user_id.reset(token)


# ────────────────────────���───────────────────────────────────────────
# 1.2b — Admin role override: admin can mutate nodes within their org
# ─────────────────────────────────────────────────────────────���──────

class TestAdminRoleOverride:
    def test_admin_can_access_other_users_org_node(self):
        """Admin in the same org can operate on another user's org-visible node."""
        token_uid = current_user_id.set("admin-user")
        token_org = current_org_ids.set(["org-1"])
        token_role = current_user_role.set("admin")
        try:
            node = _make_node(
                content_summary="user-b's node",
                original_user_id="user-b",
                visibility=Visibility.ORG,
                org_id="org-1",
            )
            assert _caller_owns_node(node) is True
        finally:
            current_user_id.reset(token_uid)
            current_org_ids.reset(token_org)
            current_user_role.reset(token_role)

    def test_admin_cannot_access_node_in_different_org(self):
        """Admin in org-1 cannot touch nodes in org-2."""
        token_uid = current_user_id.set("admin-user")
        token_org = current_org_ids.set(["org-1"])
        token_role = current_user_role.set("admin")
        try:
            node = _make_node(
                content_summary="org-2 node",
                original_user_id="user-b",
                visibility=Visibility.ORG,
                org_id="org-2",
            )
            assert _caller_owns_node(node) is False
        finally:
            current_user_id.reset(token_uid)
            current_org_ids.reset(token_org)
            current_user_role.reset(token_role)

    def test_admin_cannot_access_private_node(self):
        """Admin role does not grant access to another user's private node."""
        token_uid = current_user_id.set("admin-user")
        token_org = current_org_ids.set(["org-1"])
        token_role = current_user_role.set("admin")
        try:
            node = _make_node(
                content_summary="user-b private",
                original_user_id="user-b",
                visibility=Visibility.PRIVATE,
            )
            assert _caller_owns_node(node) is False
        finally:
            current_user_id.reset(token_uid)
            current_org_ids.reset(token_org)
            current_user_role.reset(token_role)

    def test_non_admin_in_same_org_cannot_access(self):
        """Non-admin user in the same org cannot access other's node."""
        token_uid = current_user_id.set("regular-user")
        token_org = current_org_ids.set(["org-1"])
        token_role = current_user_role.set("member")
        try:
            node = _make_node(
                content_summary="user-b's node",
                original_user_id="user-b",
                visibility=Visibility.ORG,
                org_id="org-1",
            )
            assert _caller_owns_node(node) is False
        finally:
            current_user_id.reset(token_uid)
            current_org_ids.reset(token_org)
            current_user_role.reset(token_role)
