"""Tests for GDPR user erasure (item 3.11).

Tests erase_user on InMemoryGraphProvider with both keep_promoted_nodes
modes, and verifies behavioral parity with the Postgres provider.
"""
from __future__ import annotations

import pytest

from genesys_memory.context import current_org_ids, current_user_id, current_user_role
from genesys_memory.mcp.tools import MCPToolHandler
from genesys_memory.models.edge import MemoryEdge
from genesys_memory.models.enums import EdgeType, MemoryStatus, Visibility
from genesys_memory.models.node import MemoryNode
from genesys_memory.storage.memory import InMemoryCacheProvider, InMemoryGraphProvider


def _make_node(**kwargs) -> MemoryNode:
    defaults = {"content_summary": "test memory", "status": MemoryStatus.ACTIVE}
    defaults.update(kwargs)
    return MemoryNode(**defaults)


async def _run_delete_all_scenario(graph):
    """Scenario: erase with keep_promoted_nodes=False (delete everything)."""
    tok_uid = current_user_id.set("user-a")
    tok_org = current_org_ids.set(["org-1"])
    try:
        n1 = _make_node(content_summary="a-private-1")
        n2 = _make_node(content_summary="a-private-2")
        n3 = _make_node(content_summary="a-org", visibility=Visibility.ORG, org_id="org-1")
        id1 = await graph.create_node(n1)
        id2 = await graph.create_node(n2)
        id3 = await graph.create_node(n3)
        e1 = MemoryEdge(source_id=n1.id, target_id=n2.id, type=EdgeType.CAUSED_BY, weight=0.7)
        await graph.create_edge(e1)
    finally:
        current_user_id.reset(tok_uid)
        current_org_ids.reset(tok_org)

    tok_uid = current_user_id.set("user-b")
    tok_org = current_org_ids.set(["org-1"])
    try:
        nb = _make_node(content_summary="b-node")
        id_b = await graph.create_node(nb)
        eb = MemoryEdge(source_id=nb.id, target_id=n3.id, type=EdgeType.RELATED_TO, weight=0.5)
        await graph.create_edge(eb)
    finally:
        current_user_id.reset(tok_uid)
        current_org_ids.reset(tok_org)

    manifest = await graph.erase_user("user-a", keep_promoted_nodes=False)

    tok_uid = current_user_id.set("user-b")
    tok_org = current_org_ids.set(["org-1"])
    try:
        a1 = await graph.get_node(id1)
        a2 = await graph.get_node(id2)
        a3 = await graph.get_node(id3)
        b = await graph.get_node(id_b)
        b_edges = await graph.get_edges(id_b, "both")
    finally:
        current_user_id.reset(tok_uid)
        current_org_ids.reset(tok_org)

    return {
        "manifest": manifest,
        "a_nodes_gone": a1 is None and a2 is None and a3 is None,
        "b_node_exists": b is not None,
        "b_edges_count": len(b_edges),
    }


async def _run_anonymize_scenario(graph):
    """Scenario: erase with keep_promoted_nodes=True (default, anonymize org nodes)."""
    tok_uid = current_user_id.set("user-a")
    tok_org = current_org_ids.set(["org-1"])
    try:
        n_priv = _make_node(content_summary="a-private", original_user_id="user-a")
        n_org = _make_node(
            content_summary="a-org-insight",
            visibility=Visibility.ORG,
            org_id="org-1",
            original_user_id="user-a",
        )
        id_priv = await graph.create_node(n_priv)
        id_org = await graph.create_node(n_org)
        # Edge with PII in reason
        e_pii = MemoryEdge(
            source_id=n_priv.id,
            target_id=n_org.id,
            type=EdgeType.CAUSED_BY,
            weight=0.7,
            reason="Created by john.doe@example.com during John Smith's session",
        )
        await graph.create_edge(e_pii)
    finally:
        current_user_id.reset(tok_uid)
        current_org_ids.reset(tok_org)

    tok_uid = current_user_id.set("user-b")
    tok_org = current_org_ids.set(["org-1"])
    try:
        nb = _make_node(content_summary="b-node")
        id_b = await graph.create_node(nb)
        # Edge from user-b to user-a's org node (with PII in reason)
        eb = MemoryEdge(
            source_id=nb.id,
            target_id=n_org.id,
            type=EdgeType.RELATED_TO,
            weight=0.5,
            reason="Linked by Alice Brown for context",
        )
        await graph.create_edge(eb)
    finally:
        current_user_id.reset(tok_uid)
        current_org_ids.reset(tok_org)

    manifest = await graph.erase_user("user-a", keep_promoted_nodes=True)

    tok_uid = current_user_id.set("user-b")
    tok_org = current_org_ids.set(["org-1"])
    try:
        priv = await graph.get_node(id_priv)
        org_node = await graph.get_node(id_org)
        b_node = await graph.get_node(id_b)
        # Get edges connected to the org node
        org_edges = await graph.get_edges(id_org, "both") if org_node else []
    finally:
        current_user_id.reset(tok_uid)
        current_org_ids.reset(tok_org)

    return {
        "manifest": manifest,
        "private_deleted": priv is None,
        "org_node_exists": org_node is not None,
        "org_node_owner": org_node.original_user_id if org_node else None,
        "b_node_exists": b_node is not None,
        "org_edge_reasons": [e.reason for e in org_edges],
    }


class TestInMemoryErasure:
    @pytest.mark.asyncio
    async def test_erase_delete_all(self):
        graph = InMemoryGraphProvider()
        tok = current_user_id.set("user-a")
        await graph.initialize("user-a")
        current_user_id.reset(tok)
        tok = current_user_id.set("user-b")
        await graph.initialize("user-b")
        current_user_id.reset(tok)

        result = await _run_delete_all_scenario(graph)
        assert result["manifest"]["nodes_deleted"] == 3
        assert result["manifest"]["edges_deleted"] >= 1
        assert result["a_nodes_gone"] is True
        assert result["b_node_exists"] is True
        assert result["b_edges_count"] == 0

    @pytest.mark.asyncio
    async def test_erase_anonymize_promoted(self):
        graph = InMemoryGraphProvider()
        tok = current_user_id.set("user-a")
        await graph.initialize("user-a")
        current_user_id.reset(tok)
        tok = current_user_id.set("user-b")
        await graph.initialize("user-b")
        current_user_id.reset(tok)

        result = await _run_anonymize_scenario(graph)
        assert result["manifest"]["nodes_deleted"] == 1
        assert result["manifest"]["nodes_anonymized"] == 1
        assert result["private_deleted"] is True
        assert result["org_node_exists"] is True
        assert result["org_node_owner"] == "erased_user"
        assert result["b_node_exists"] is True
        # PII should be scrubbed from edge reasons
        for reason in result["org_edge_reasons"]:
            if reason:
                assert "john.doe@example.com" not in reason
                assert "John Smith" not in reason
                assert "Alice Brown" not in reason

    @pytest.mark.asyncio
    async def test_erase_nonexistent_user_returns_zeros(self):
        graph = InMemoryGraphProvider()
        manifest = await graph.erase_user("nobody")
        assert manifest["nodes_deleted"] == 0
        assert manifest["edges_deleted"] == 0
        assert manifest["nodes_anonymized"] == 0
        assert manifest["edges_scrubbed"] == 0

    @pytest.mark.asyncio
    async def test_erase_cleans_node_locks(self):
        graph = InMemoryGraphProvider()
        tok = current_user_id.set("lock-user")
        tok_org = current_org_ids.set([])
        try:
            await graph.initialize("lock-user")
            node = _make_node()
            node_id = await graph.create_node(node)
            graph._get_node_lock(node_id)
            assert node_id in graph._node_locks
        finally:
            current_user_id.reset(tok)
            current_org_ids.reset(tok_org)

        await graph.erase_user("lock-user", keep_promoted_nodes=False)
        assert node_id not in graph._node_locks


class TestErasureToolHandler:
    @pytest.mark.asyncio
    async def test_erase_own_data(self):
        graph = InMemoryGraphProvider()
        cache = InMemoryCacheProvider()
        tok = current_user_id.set("doomed-user")
        tok_org = current_org_ids.set([])
        try:
            await graph.initialize("doomed-user")
            handler = MCPToolHandler(graph=graph, embeddings=None, cache=cache)
            node = _make_node(original_user_id="doomed-user")
            await graph.create_node(node)
            result = await handler.erase_user("doomed-user", keep_promoted_nodes=False)
            assert result["status"] == "erased"
            assert result["nodes_deleted"] == 1
        finally:
            current_user_id.reset(tok)
            current_org_ids.reset(tok_org)

    @pytest.mark.asyncio
    async def test_erase_other_user_blocked_for_non_admin(self):
        graph = InMemoryGraphProvider()
        cache = InMemoryCacheProvider()
        tok = current_user_id.set("regular-user")
        tok_org = current_org_ids.set([])
        try:
            handler = MCPToolHandler(graph=graph, embeddings=None, cache=cache)
            result = await handler.erase_user("other-user")
            assert "error" in result
        finally:
            current_user_id.reset(tok)
            current_org_ids.reset(tok_org)

    @pytest.mark.asyncio
    async def test_erase_other_user_allowed_for_admin(self):
        graph = InMemoryGraphProvider()
        cache = InMemoryCacheProvider()
        tok = current_user_id.set("admin-user")
        tok_org = current_org_ids.set([])
        tok_role = current_user_role.set("admin")
        try:
            await graph.initialize("target-user")
            tok2 = current_user_id.set("target-user")
            node = _make_node(original_user_id="target-user")
            await graph.create_node(node)
            current_user_id.reset(tok2)

            tok3 = current_user_id.set("admin-user")
            handler = MCPToolHandler(graph=graph, embeddings=None, cache=cache)
            result = await handler.erase_user("target-user", keep_promoted_nodes=False)
            assert result["status"] == "erased"
            current_user_id.reset(tok3)
        finally:
            current_user_id.reset(tok)
            current_org_ids.reset(tok_org)
            current_user_role.reset(tok_role)

    @pytest.mark.asyncio
    async def test_erase_no_user_context_raises(self):
        graph = InMemoryGraphProvider()
        cache = InMemoryCacheProvider()
        handler = MCPToolHandler(graph=graph, embeddings=None, cache=cache)
        with pytest.raises(PermissionError):
            await handler.erase_user("any-user")

    @pytest.mark.asyncio
    async def test_erase_with_keep_promoted_returns_anonymized_count(self):
        graph = InMemoryGraphProvider()
        cache = InMemoryCacheProvider()
        tok = current_user_id.set("user-a")
        tok_org = current_org_ids.set(["org-1"])
        try:
            await graph.initialize("user-a")
            handler = MCPToolHandler(graph=graph, embeddings=None, cache=cache)
            n_priv = _make_node(content_summary="private", original_user_id="user-a")
            n_org = _make_node(
                content_summary="org",
                visibility=Visibility.ORG,
                org_id="org-1",
                original_user_id="user-a",
            )
            await graph.create_node(n_priv)
            await graph.create_node(n_org)

            result = await handler.erase_user("user-a", keep_promoted_nodes=True)
            assert result["status"] == "erased"
            assert result["nodes_deleted"] == 1
            assert result["nodes_anonymized"] == 1
        finally:
            current_user_id.reset(tok)
            current_org_ids.reset(tok_org)


class TestProviderEquivalence:
    """Behavioral parity: same scenario, same assertions, both providers."""

    @pytest.mark.asyncio
    async def test_inmemory_delete_all(self):
        graph = InMemoryGraphProvider()
        tok = current_user_id.set("user-a")
        await graph.initialize("user-a")
        current_user_id.reset(tok)
        tok = current_user_id.set("user-b")
        await graph.initialize("user-b")
        current_user_id.reset(tok)

        result = await _run_delete_all_scenario(graph)
        assert result["manifest"]["nodes_deleted"] == 3
        assert result["manifest"]["edges_deleted"] >= 1
        assert result["a_nodes_gone"] is True
        assert result["b_node_exists"] is True
        assert result["b_edges_count"] == 0

    @pytest.mark.asyncio
    async def test_inmemory_anonymize(self):
        graph = InMemoryGraphProvider()
        tok = current_user_id.set("user-a")
        await graph.initialize("user-a")
        current_user_id.reset(tok)
        tok = current_user_id.set("user-b")
        await graph.initialize("user-b")
        current_user_id.reset(tok)

        result = await _run_anonymize_scenario(graph)
        assert result["manifest"]["nodes_anonymized"] == 1
        assert result["org_node_owner"] == "erased_user"
        assert result["manifest"]["edges_scrubbed"] >= 1

    @pytest.mark.asyncio
    async def test_postgres_delete_all(self):
        pytest.importorskip("asyncpg")
        try:
            from genesys_server.storage.postgres import PostgresGraphProvider
            from genesys_server.storage.db import close_pool
        except ImportError:
            pytest.skip("genesys-server not installed")

        import asyncpg
        import os
        db_url = os.environ.get("DATABASE_URL", "postgresql://genesys:genesys@localhost:5432/genesys")
        try:
            conn = await asyncpg.connect(db_url, timeout=3)
            await conn.close()
        except Exception:
            pytest.skip("Postgres not available")

        os.environ["DATABASE_URL"] = db_url
        graph = PostgresGraphProvider()
        await graph.initialize("user-a")
        await graph.initialize("user-b")
        try:
            result = await _run_delete_all_scenario(graph)
            assert result["manifest"]["nodes_deleted"] == 3
            assert result["manifest"]["edges_deleted"] >= 1
            assert result["a_nodes_gone"] is True
            assert result["b_node_exists"] is True
            assert result["b_edges_count"] == 0
        finally:
            await graph.destroy("user-b")
            await close_pool()

    @pytest.mark.asyncio
    async def test_postgres_anonymize(self):
        pytest.importorskip("asyncpg")
        try:
            from genesys_server.storage.postgres import PostgresGraphProvider
            from genesys_server.storage.db import close_pool
        except ImportError:
            pytest.skip("genesys-server not installed")

        import asyncpg
        import os
        db_url = os.environ.get("DATABASE_URL", "postgresql://genesys:genesys@localhost:5432/genesys")
        try:
            conn = await asyncpg.connect(db_url, timeout=3)
            await conn.close()
        except Exception:
            pytest.skip("Postgres not available")

        os.environ["DATABASE_URL"] = db_url
        graph = PostgresGraphProvider()
        await graph.initialize("user-a")
        await graph.initialize("user-b")
        try:
            result = await _run_anonymize_scenario(graph)
            assert result["manifest"]["nodes_anonymized"] == 1
            assert result["org_node_owner"] == "erased_user"
            assert result["manifest"]["edges_scrubbed"] >= 1
        finally:
            # Clean up anonymized node
            await graph.erase_user("user-a", keep_promoted_nodes=False)
            await graph.destroy("user-b")
            await close_pool()
