"""Cross-org attack path tests (task 2.10).

Verifies that multi-tenant isolation holds: users in org-A cannot
read, link to, or promote memories in org-B.
"""
from __future__ import annotations

import pytest

from genesys_memory.context import current_org_ids, current_user_id
from genesys_memory.mcp.tools import MCPToolHandler
from genesys_memory.storage.memory import InMemoryCacheProvider, InMemoryGraphProvider


class FakeEmbedder:
    dimension = 16

    async def embed(self, text: str) -> list[float]:
        h = hash(text) & 0xFFFFFFFF
        vec = [(((h >> (i * 2)) & 3) - 1.5) / 1.5 for i in range(self.dimension)]
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


@pytest.fixture()
def shared_graph():
    return InMemoryGraphProvider()


class TestCrossOrgAttackPaths:
    @pytest.mark.asyncio
    async def test_user_b_cannot_recall_org_a_memories(self, shared_graph):
        """User-B (org-B only) must not see org-A memories in recall."""
        cache = InMemoryCacheProvider()
        embedder = FakeEmbedder()

        # User-A stores an org-A memory
        tok_uid = current_user_id.set("user-a")
        tok_org = current_org_ids.set(["org-a"])
        try:
            handler_a = MCPToolHandler(graph=shared_graph, embeddings=embedder, cache=cache)
            r = await handler_a.memory_store(
                "Org-A secret strategy document",
                visibility="org",
                org_id="org-a",
            )
            org_a_node_id = r["node_id"]
        finally:
            current_user_id.reset(tok_uid)
            current_org_ids.reset(tok_org)

        # User-B (org-B) tries to recall
        tok_uid = current_user_id.set("user-b")
        tok_org = current_org_ids.set(["org-b"])
        try:
            handler_b = MCPToolHandler(graph=shared_graph, embeddings=embedder, cache=cache)
            result = await handler_b.memory_recall("secret strategy", k=10, read_only=True)
            result_ids = [m.get("id") for m in result["results"]]
            assert org_a_node_id not in result_ids
        finally:
            current_user_id.reset(tok_uid)
            current_org_ids.reset(tok_org)

    @pytest.mark.asyncio
    async def test_user_b_cannot_promote_to_org_a(self, shared_graph):
        """User-B must not be able to promote their node into org-A."""
        cache = InMemoryCacheProvider()
        embedder = FakeEmbedder()

        # User-B stores a private memory
        tok_uid = current_user_id.set("user-b")
        tok_org = current_org_ids.set(["org-b"])
        try:
            handler_b = MCPToolHandler(graph=shared_graph, embeddings=embedder, cache=cache)
            r = await handler_b.memory_store("My private note")
            node_id = r["node_id"]

            # Try to promote into org-A (not in user-b's org list)
            result = await handler_b.promote_to_org(node_id, "org-a")
            assert "error" in result
            assert "org_id not in caller" in result["error"]
        finally:
            current_user_id.reset(tok_uid)
            current_org_ids.reset(tok_org)

    @pytest.mark.asyncio
    async def test_user_b_cannot_store_into_org_a(self, shared_graph):
        """User-B must not be able to store directly into org-A."""
        cache = InMemoryCacheProvider()
        embedder = FakeEmbedder()

        tok_uid = current_user_id.set("user-b")
        tok_org = current_org_ids.set(["org-b"])
        try:
            handler_b = MCPToolHandler(graph=shared_graph, embeddings=embedder, cache=cache)
            result = await handler_b.memory_store(
                "Trying to inject into org-A",
                visibility="org",
                org_id="org-a",
            )
            assert "error" in result
            assert "org_id not in caller" in result["error"]
        finally:
            current_user_id.reset(tok_uid)
            current_org_ids.reset(tok_org)

    @pytest.mark.asyncio
    async def test_user_a_cannot_delete_user_b_org_memory(self, shared_graph):
        """User-A must not be able to delete a memory owned by User-B, even in the same org."""
        cache = InMemoryCacheProvider()
        embedder = FakeEmbedder()

        # User-B creates an org-shared memory
        tok_uid = current_user_id.set("user-b")
        tok_org = current_org_ids.set(["org-shared"])
        try:
            handler_b = MCPToolHandler(graph=shared_graph, embeddings=embedder, cache=cache)
            r = await handler_b.memory_store(
                "User-B's shared note",
                visibility="org",
                org_id="org-shared",
            )
            node_id = r["node_id"]
        finally:
            current_user_id.reset(tok_uid)
            current_org_ids.reset(tok_org)

        # User-A (same org) tries to delete
        tok_uid = current_user_id.set("user-a")
        tok_org = current_org_ids.set(["org-shared"])
        try:
            handler_a = MCPToolHandler(graph=shared_graph, embeddings=embedder, cache=cache)
            result = await handler_a.delete_memory(node_id)
            assert "error" in result
            assert "owner" in result["error"].lower()
        finally:
            current_user_id.reset(tok_uid)
            current_org_ids.reset(tok_org)
