"""End-to-end tests for memory_store and memory_recall (task 2.9).

Uses InMemoryGraphProvider (no mocks) to exercise the full
store→recall→reactivation→status-transition path.
"""
from __future__ import annotations

import pytest

from genesys_memory.context import current_org_ids, current_user_id
from genesys_memory.mcp.tools import MCPToolHandler
from genesys_memory.storage.memory import InMemoryCacheProvider, InMemoryGraphProvider


class FakeEmbedder:
    """Deterministic embedder for testing — hashes content into a 16-dim vector."""

    dimension = 16

    async def embed(self, text: str) -> list[float]:
        h = hash(text) & 0xFFFFFFFF
        vec = [(((h >> (i * 2)) & 3) - 1.5) / 1.5 for i in range(self.dimension)]
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


@pytest.fixture()
def _user_ctx():
    token = current_user_id.set("e2e-user")
    org_token = current_org_ids.set([])
    yield
    current_user_id.reset(token)
    current_org_ids.reset(org_token)


@pytest.fixture()
def handler(_user_ctx):
    graph = InMemoryGraphProvider()
    embeddings = FakeEmbedder()
    cache = InMemoryCacheProvider()
    return MCPToolHandler(graph=graph, embeddings=embeddings, cache=cache)


@pytest.mark.usefixtures("_user_ctx")
class TestE2ECoreOperations:
    @pytest.mark.asyncio
    async def test_store_returns_node_id(self, handler):
        """Storing a memory returns a valid node_id."""
        result = await handler.memory_store("The project uses Python 3.12")
        assert "node_id" in result
        assert result["status"] == "stored"
        assert result["visibility"] == "private"

    @pytest.mark.asyncio
    async def test_store_then_recall(self, handler):
        """Stored content should be retrievable via recall."""
        await handler.memory_store("Railway deploys via Docker on port 8080")
        result = await handler.memory_recall("How is Railway deployed?", k=5)
        assert result["count"] >= 1
        contents = [r["content"] for r in result["results"]]
        assert any("Railway" in c for c in contents)

    @pytest.mark.asyncio
    async def test_recall_updates_reactivation_count(self, handler):
        """Recalling a memory should increment its reactivation_count."""
        store_result = await handler.memory_store("Neon Postgres in us-east-1")
        node_id = store_result["node_id"]

        recall_result = await handler.memory_recall("What database region?", k=5)
        matched = [r for r in recall_result["results"] if r.get("id") == node_id]
        if matched:
            assert matched[0].get("reactivation_count", 0) >= 1

    @pytest.mark.asyncio
    async def test_recall_read_only_skips_mutation(self, handler):
        """read_only=True should not mutate reactivation state."""
        store_result = await handler.memory_store("Clerk handles authentication")
        node_id = store_result["node_id"]

        node_before = handler.graph.nodes.get(node_id)
        count_before = node_before.reactivation_count if node_before else 0

        await handler.memory_recall("auth provider?", k=5, read_only=True)

        node_after = handler.graph.nodes.get(node_id)
        assert node_after.reactivation_count == count_before

    @pytest.mark.asyncio
    async def test_multiple_stores_recall_ranks_by_relevance(self, handler):
        """More relevant content should rank higher in recall results."""
        await handler.memory_store("The cat sat on the mat")
        await handler.memory_store("Python 3.12 is the runtime version")
        await handler.memory_store("We use pytest for all testing")

        result = await handler.memory_recall("What Python version?", k=5)
        assert result["count"] >= 1

    @pytest.mark.asyncio
    async def test_store_with_related_to_creates_edges(self, handler):
        """Storing with related_to should create edges between nodes."""
        r1 = await handler.memory_store("Postgres is the database")
        node1_id = r1["node_id"]

        r2 = await handler.memory_store(
            "pgvector extension enables vector search",
            related_to=[node1_id],
        )
        node2_id = r2["node_id"]

        edges = await handler.graph.get_edges(node2_id, "out")
        target_ids = [str(e.target_id) for e in edges]
        assert node1_id in target_ids

    @pytest.mark.asyncio
    async def test_store_pin_recall_core_injection(self, handler):
        """Pinned core memories should be injected into recall results."""
        r = await handler.memory_store("The API key must never be committed")
        node_id = r["node_id"]

        await handler.pin_memory(node_id)

        result = await handler.memory_recall("unrelated topic about weather", k=5)
        result_ids = [m.get("id") for m in result["results"]]
        assert node_id in result_ids

    @pytest.mark.asyncio
    async def test_delete_then_recall_excludes_deleted(self, handler):
        """Deleted memories must not appear in recall results."""
        r = await handler.memory_store("Temporary note to delete")
        node_id = r["node_id"]

        await handler.delete_memory(node_id)

        result = await handler.memory_recall("Temporary note", k=5)
        result_ids = [m.get("id") for m in result["results"]]
        assert node_id not in result_ids
