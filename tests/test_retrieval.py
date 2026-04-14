"""Phase 1 retrieval tests.

These tests require FalkorDB running (docker-compose up) and OPENAI_API_KEY set.
"""
from __future__ import annotations

import os
import uuid

import pytest
from dotenv import load_dotenv

load_dotenv()

def _can_run():
    if not os.getenv("OPENAI_API_KEY"):
        return False
    try:
        import falkordb
        db = falkordb.FalkorDB(host=os.getenv("FALKORDB_HOST", "localhost"),
                                port=int(os.getenv("FALKORDB_PORT", "6379")))
        db.connection.ping()
        return True
    except Exception:
        return False

pytestmark = pytest.mark.skipif(
    not _can_run(),
    reason="Requires OPENAI_API_KEY and FalkorDB",
)

from genesys.models.enums import EdgeType, MemoryStatus
from genesys.models.node import MemoryNode
from genesys.models.edge import MemoryEdge
from genesys.mcp.tools import MCPToolHandler
from genesys.retrieval.embedding import OpenAIEmbeddingProvider
from genesys.storage.cache import RedisCacheProvider
from genesys.storage.falkordb import FalkorDBProvider

FALKORDB_HOST = os.getenv("FALKORDB_HOST", "localhost")
FALKORDB_PORT = int(os.getenv("FALKORDB_PORT", "6379"))


@pytest.fixture
async def provider():
    """Create an isolated FalkorDB provider for each test."""
    user_id = f"test_{uuid.uuid4().hex[:8]}"
    p = FalkorDBProvider(host=FALKORDB_HOST, port=FALKORDB_PORT)
    await p.initialize(user_id)
    yield p
    await p.destroy(user_id)


@pytest.fixture
def cache():
    return RedisCacheProvider(host=FALKORDB_HOST, port=FALKORDB_PORT)


@pytest.fixture
def embeddings(cache):
    api_key = os.getenv("OPENAI_API_KEY", "")
    return OpenAIEmbeddingProvider(api_key=api_key, cache=cache)


@pytest.fixture
async def handler(provider, embeddings, cache):
    return MCPToolHandler(graph=provider, embeddings=embeddings, cache=cache)


async def test_store_and_recall_basic(handler):
    """Store a memory, recall it by semantic similarity."""
    result = await handler.memory_store(content="Python is my favorite programming language")
    assert "node_id" in result

    recall = await handler.memory_recall(query="What programming language do I prefer?", k=5)
    assert recall["count"] >= 1
    assert "Python" in recall["results"][0]["content"]


async def test_store_with_causal_edge(handler, provider):
    """Store memory A, then store memory B related_to A. Verify edge exists."""
    result_a = await handler.memory_store(content="I started a new job at Acme Corp")
    node_a_id = result_a["node_id"]

    result_b = await handler.memory_store(
        content="My manager at Acme Corp is Sarah",
        related_to=[node_a_id],
    )
    node_b_id = result_b["node_id"]

    exists = await provider.edge_exists(node_b_id, node_a_id, EdgeType.CAUSED_BY)
    assert exists


async def test_recall_includes_causal_context(handler):
    """Store A, store B caused_by A. Recall B. Response should include A as causal basis."""
    result_a = await handler.memory_store(content="I have a severe allergy to peanuts")
    node_a_id = result_a["node_id"]

    await handler.memory_store(
        content="I always carry an EpiPen because of my peanut allergy",
        related_to=[node_a_id],
    )

    recall = await handler.memory_recall(query="Why do I carry an EpiPen?", k=5)
    # Find the EpiPen memory in results
    epipen_result = None
    for r in recall["results"]:
        if "EpiPen" in r["content"]:
            epipen_result = r
            break

    assert epipen_result is not None
    assert len(epipen_result["causal_basis"]) > 0
    assert "peanut" in epipen_result["causal_basis"][0]["summary"].lower()


async def test_recall_filters_pruned(handler, provider):
    """Store a memory, set status to PRUNED. Recall should not return it."""
    result = await handler.memory_store(content="This is a temporary memory about goldfish")
    node_id = result["node_id"]

    await provider.update_node(node_id, {"status": MemoryStatus.PRUNED.value})

    recall = await handler.memory_recall(query="goldfish", k=5)
    for r in recall["results"]:
        assert r["id"] != node_id


async def test_vector_search_accuracy(handler):
    """Store diverse memories. Query with a specific topic. Top result should be the most relevant."""
    topics = [
        "I enjoy hiking in the Rocky Mountains every summer",
        "My favorite book is Dune by Frank Herbert",
        "I am learning to play the piano",
        "My cat's name is Whiskers and she is 3 years old",
        "I work as a software engineer at a startup",
    ]
    for t in topics:
        await handler.memory_store(content=t)

    recall = await handler.memory_recall(query="What pet do I have?", k=5)
    assert recall["count"] >= 1
    assert "cat" in recall["results"][0]["content"].lower() or "Whiskers" in recall["results"][0]["content"]


async def test_per_user_isolation():
    """Store memories for user_1 and user_2. Recall for user_1 should not return user_2 memories."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    cache = RedisCacheProvider(host=FALKORDB_HOST, port=FALKORDB_PORT)
    emb = OpenAIEmbeddingProvider(api_key=api_key, cache=cache)

    p1 = FalkorDBProvider(host=FALKORDB_HOST, port=FALKORDB_PORT)
    p2 = FalkorDBProvider(host=FALKORDB_HOST, port=FALKORDB_PORT)
    uid1 = f"test_iso_{uuid.uuid4().hex[:8]}"
    uid2 = f"test_iso_{uuid.uuid4().hex[:8]}"

    await p1.initialize(uid1)
    await p2.initialize(uid2)

    try:
        h1 = MCPToolHandler(graph=p1, embeddings=emb, cache=cache)
        h2 = MCPToolHandler(graph=p2, embeddings=emb, cache=cache)

        await h1.memory_store(content="User 1 secret: the password is hunter2")
        await h2.memory_store(content="User 2 likes vanilla ice cream")

        recall1 = await h1.memory_recall(query="ice cream", k=5)
        for r in recall1["results"]:
            assert "vanilla" not in r["content"].lower()

        recall2 = await h2.memory_recall(query="password", k=5)
        for r in recall2["results"]:
            assert "hunter2" not in r["content"].lower()
    finally:
        await p1.destroy(uid1)
        await p2.destroy(uid2)
