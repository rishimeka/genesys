"""Integration tests using FalkorDB with mock embeddings.

Requires FalkorDB running on localhost:6379.
No external API keys needed.
"""
from __future__ import annotations

import hashlib
import time
import uuid

import numpy as np
import pytest

def _falkordb_available():
    try:
        import falkordb
        db = falkordb.FalkorDB(host="localhost", port=6379)
        db.connection.ping()
        return True
    except Exception:
        return False

pytestmark = pytest.mark.skipif(
    not _falkordb_available(),
    reason="Requires FalkorDB running on localhost:6379",
)

from genesys.models.edge import MemoryEdge
from genesys.models.enums import EdgeType, MemoryStatus
from genesys.models.node import MemoryNode
from genesys.mcp.tools import MCPToolHandler
from genesys.storage.falkordb import FalkorDBProvider


# --- Mock providers ---

class MockEmbeddingProvider:
    """Deterministic embeddings based on text hash. Dimension=1536."""

    @property
    def dimension(self) -> int:
        return 1536

    async def embed(self, text: str) -> list[float]:
        # Use hash to generate a deterministic pseudo-embedding
        h = hashlib.sha256(text.encode()).digest()
        rng = np.random.RandomState(int.from_bytes(h[:4], "big"))
        vec = rng.randn(1536).astype(float)
        # Normalize to unit vector for cosine similarity
        vec = vec / np.linalg.norm(vec)
        return vec.tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


class MockCacheProvider:
    """In-memory cache for testing."""

    def __init__(self):
        self._store: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self._store.get(key)

    async def set(self, key: str, value: str, ttl_seconds: int = 300) -> None:
        self._store[key] = value

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

    async def exists(self, key: str) -> bool:
        return key in self._store


class SemanticMockEmbedding:
    """Embeddings where similar topics produce similar vectors.

    Uses keyword overlap to create vectors that have high cosine similarity
    for related content.
    """

    TOPICS = {
        "programming": [0, 1, 2, 3],
        "python": [0, 1, 2, 4],
        "java": [0, 1, 5, 6],
        "pet": [10, 11, 12, 13],
        "cat": [10, 11, 12, 14],
        "dog": [10, 11, 12, 15],
        "retriever": [10, 11, 12, 16],
        "hiking": [20, 21, 22, 23],
        "mountain": [20, 21, 24, 25],
        "book": [30, 31, 32, 33],
        "piano": [40, 41, 42, 43],
        "music": [40, 41, 44, 45],
        "job": [50, 51, 52, 53],
        "work": [50, 51, 54, 55],
        "software": [50, 51, 0, 1],
        "allergy": [60, 61, 62, 63],
        "allergies": [60, 61, 62, 63],
        "allergic": [60, 61, 62, 64],
        "peanut": [60, 61, 64, 65],
        "epipen": [60, 61, 66, 67],
        "shellfish": [60, 61, 68, 69],
        "password": [70, 71, 72, 73],
        "secret": [70, 71, 74, 75],
        "ice cream": [80, 81, 82, 83],
        "vanilla": [80, 81, 84, 85],
        "goldfish": [90, 91, 92, 93],
        "favorite": [100, 101, 102, 103],
        "language": [0, 1, 104, 105],
        "prefer": [100, 101, 106, 107],
        "manager": [50, 51, 108, 109],
        "acme": [50, 51, 110, 111],
        "scientist": [50, 51, 0, 112],
        "google": [50, 51, 113, 114],
        "restaurant": [80, 81, 115, 116],
        "italian": [80, 81, 117, 118],
        "run": [120, 121, 122, 123],
        "morning": [120, 121, 124, 125],
        "marathon": [120, 121, 126, 127],
        "carry": [60, 61, 112, 113],
        "whiskers": [10, 11, 114, 115],
    }

    @property
    def dimension(self) -> int:
        return 1536

    async def embed(self, text: str) -> list[float]:
        # Use deterministic seed per keyword cluster to produce dense sub-vectors
        # This gives reliable cosine similarity in HNSW indexes
        vec = np.zeros(1536)
        text_lower = text.lower()
        for keyword, dims in self.TOPICS.items():
            if keyword in text_lower:
                # Spread signal across a wider band per keyword for HNSW compatibility
                rng = np.random.RandomState(dims[0])
                band = rng.randn(32)  # 32 dense dims per keyword cluster
                for i, d in enumerate(dims):
                    start = (d * 7) % (1536 - 32)
                    vec[start:start + 32] += band
        # Add some noise based on full text hash
        h = hashlib.sha256(text.encode()).digest()
        rng = np.random.RandomState(int.from_bytes(h[:4], "big"))
        vec += rng.randn(1536) * 0.05
        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


# --- Fixtures ---

@pytest.fixture
async def db():
    """Create isolated FalkorDB graph."""
    uid = f"test_{uuid.uuid4().hex[:8]}"
    p = FalkorDBProvider(host="localhost", port=6379)
    await p.initialize(uid)
    yield p
    await p.destroy(uid)


@pytest.fixture
def emb():
    return SemanticMockEmbedding()


@pytest.fixture
def cache():
    return MockCacheProvider()


@pytest.fixture
async def handler(db, emb, cache):
    return MCPToolHandler(graph=db, embeddings=emb, cache=cache)


# --- Tests ---

class TestNodeCRUD:
    """Test basic node create/read/update/delete against FalkorDB."""

    async def test_create_and_get_node(self, db, emb):
        vec = await emb.embed("test content")
        node = MemoryNode(
            content_summary="test content",
            content_full="test content full",
            embedding=vec,
            source_session="test",
        )
        node_id = await db.create_node(node)
        assert node_id == str(node.id)

        fetched = await db.get_node(node_id)
        assert fetched is not None
        assert fetched.content_summary == "test content"
        assert fetched.status == MemoryStatus.ACTIVE

    async def test_update_node(self, db, emb):
        vec = await emb.embed("update me")
        node = MemoryNode(content_summary="update me", embedding=vec, source_session="t")
        node_id = await db.create_node(node)

        await db.update_node(node_id, {"status": "dormant", "decay_score": 0.5})

        fetched = await db.get_node(node_id)
        assert fetched.status == MemoryStatus.DORMANT
        assert fetched.decay_score == 0.5

    async def test_delete_node(self, db, emb):
        vec = await emb.embed("delete me")
        node = MemoryNode(content_summary="delete me", embedding=vec, source_session="t")
        node_id = await db.create_node(node)

        await db.delete_node(node_id)
        assert await db.get_node(node_id) is None

    async def test_get_nodes_by_status(self, db, emb):
        for i in range(3):
            vec = await emb.embed(f"active node {i}")
            node = MemoryNode(content_summary=f"active node {i}", embedding=vec, source_session="t")
            await db.create_node(node)

        vec = await emb.embed("dormant node")
        dorm = MemoryNode(content_summary="dormant node", embedding=vec, status=MemoryStatus.DORMANT, source_session="t")
        await db.create_node(dorm)

        active = await db.get_nodes_by_status(MemoryStatus.ACTIVE)
        assert len(active) == 3
        dormant = await db.get_nodes_by_status(MemoryStatus.DORMANT)
        assert len(dormant) == 1


class TestEdgeOperations:
    """Test edge CRUD and traversal."""

    async def test_create_edge_and_check_exists(self, db, emb):
        vec1 = await emb.embed("cause")
        vec2 = await emb.embed("effect")
        n1 = MemoryNode(content_summary="cause", embedding=vec1, source_session="t")
        n2 = MemoryNode(content_summary="effect", embedding=vec2, source_session="t")
        await db.create_node(n1)
        await db.create_node(n2)

        edge = MemoryEdge(source_id=n2.id, target_id=n1.id, type=EdgeType.CAUSED_BY)
        await db.create_edge(edge)

        assert await db.edge_exists(str(n2.id), str(n1.id), EdgeType.CAUSED_BY)
        assert not await db.edge_exists(str(n1.id), str(n2.id), EdgeType.CAUSED_BY)

    async def test_is_orphan(self, db, emb):
        vec = await emb.embed("lonely node")
        node = MemoryNode(content_summary="lonely node", embedding=vec, source_session="t")
        await db.create_node(node)
        assert await db.is_orphan(str(node.id))

        vec2 = await emb.embed("friend")
        friend = MemoryNode(content_summary="friend", embedding=vec2, source_session="t")
        await db.create_node(friend)
        edge = MemoryEdge(source_id=node.id, target_id=friend.id, type=EdgeType.RELATED_TO)
        await db.create_edge(edge)

        assert not await db.is_orphan(str(node.id))

    async def test_causal_chain(self, db, emb):
        # Create A -> B -> C chain (C caused_by B, B caused_by A)
        nodes = []
        for label in ["root cause", "intermediate effect", "final effect"]:
            vec = await emb.embed(label)
            n = MemoryNode(content_summary=label, embedding=vec, source_session="t")
            await db.create_node(n)
            nodes.append(n)

        # B caused_by A
        await db.create_edge(MemoryEdge(source_id=nodes[1].id, target_id=nodes[0].id, type=EdgeType.CAUSED_BY))
        # C caused_by B
        await db.create_edge(MemoryEdge(source_id=nodes[2].id, target_id=nodes[1].id, type=EdgeType.CAUSED_BY))

        # Upstream from C should give B and A
        upstream = await db.get_causal_chain(str(nodes[2].id), "upstream")
        upstream_ids = {str(n.id) for n in upstream}
        assert str(nodes[1].id) in upstream_ids
        assert str(nodes[0].id) in upstream_ids

        # Downstream from A should give B and C
        downstream = await db.get_causal_chain(str(nodes[0].id), "downstream")
        downstream_ids = {str(n.id) for n in downstream}
        assert str(nodes[1].id) in downstream_ids
        assert str(nodes[2].id) in downstream_ids

    async def test_causal_weight(self, db, emb):
        # A is the root cause, B and C depend on it
        nodes = []
        for label in ["root", "dep1", "dep2"]:
            vec = await emb.embed(label)
            n = MemoryNode(content_summary=label, embedding=vec, source_session="t")
            await db.create_node(n)
            nodes.append(n)

        await db.create_edge(MemoryEdge(source_id=nodes[1].id, target_id=nodes[0].id, type=EdgeType.CAUSED_BY))
        await db.create_edge(MemoryEdge(source_id=nodes[2].id, target_id=nodes[0].id, type=EdgeType.CAUSED_BY))

        weight = await db.get_causal_weight(str(nodes[0].id))
        assert weight == 2


class TestVectorSearch:
    """Test vector similarity search."""

    async def test_basic_vector_search(self, db, emb):
        contents = [
            "Python is my favorite programming language",
            "I enjoy hiking in the Rocky Mountains",
            "My cat Whiskers is 3 years old",
        ]
        for c in contents:
            vec = await emb.embed(c)
            node = MemoryNode(content_summary=c, embedding=vec, source_session="t")
            await db.create_node(node)

        query_vec = await emb.embed("What pet do I have?")
        results = await db.vector_search(query_vec, k=3)
        assert len(results) > 0
        # The cat memory should rank highest for a pet query
        top_summary = results[0][0].content_summary.lower()
        assert "cat" in top_summary or "whiskers" in top_summary

    async def test_vector_search_excludes_pruned(self, db, emb):
        vec = await emb.embed("pruned memory about goldfish")
        node = MemoryNode(
            content_summary="pruned memory about goldfish",
            embedding=vec,
            status=MemoryStatus.PRUNED,
            source_session="t",
        )
        await db.create_node(node)

        query_vec = await emb.embed("goldfish")
        results = await db.vector_search(query_vec, k=5)
        for n, _ in results:
            assert n.status != MemoryStatus.PRUNED


class TestMCPTools:
    """End-to-end tests via the MCPToolHandler."""

    async def test_store_and_recall(self, handler):
        result = await handler.memory_store(content="Python is my favorite programming language")
        assert "node_id" in result

        recall = await handler.memory_recall(query="What programming language do I prefer?", k=5)
        assert recall["count"] >= 1
        found = any("Python" in r["content"] for r in recall["results"])
        assert found

    async def test_store_with_related_to(self, handler, db):
        r1 = await handler.memory_store(content="I started a new job at Acme Corp")
        node_a_id = r1["node_id"]

        r2 = await handler.memory_store(
            content="My manager at Acme Corp is Sarah",
            related_to=[node_a_id],
        )
        node_b_id = r2["node_id"]

        assert await db.edge_exists(node_b_id, node_a_id, EdgeType.CAUSED_BY)

    async def test_recall_with_causal_context(self, handler):
        r1 = await handler.memory_store(content="I have a severe allergy to peanuts")
        r2 = await handler.memory_store(
            content="I always carry an EpiPen because of my peanut allergy",
            related_to=[r1["node_id"]],
        )

        recall = await handler.memory_recall(query="Why do I carry an EpiPen?", k=5)
        epipen_result = None
        for r in recall["results"]:
            if "EpiPen" in r["content"]:
                epipen_result = r
                break

        assert epipen_result is not None
        assert len(epipen_result["causal_basis"]) > 0
        assert "peanut" in epipen_result["causal_basis"][0]["summary"].lower()

    async def test_recall_filters_pruned(self, handler, db):
        result = await handler.memory_store(content="Temporary memory about goldfish")
        node_id = result["node_id"]

        await db.update_node(node_id, {"status": MemoryStatus.PRUNED.value})

        recall = await handler.memory_recall(query="goldfish", k=5)
        for r in recall["results"]:
            assert r["id"] != node_id

    async def test_per_user_isolation(self, emb, cache):
        p1 = FalkorDBProvider(host="localhost", port=6379)
        p2 = FalkorDBProvider(host="localhost", port=6379)
        uid1 = f"iso_{uuid.uuid4().hex[:8]}"
        uid2 = f"iso_{uuid.uuid4().hex[:8]}"
        await p1.initialize(uid1)
        await p2.initialize(uid2)

        try:
            h1 = MCPToolHandler(graph=p1, embeddings=emb, cache=cache)
            h2 = MCPToolHandler(graph=p2, embeddings=emb, cache=cache)

            await h1.memory_store(content="User 1 secret password is hunter2")
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


class TestStats:
    """Test graph statistics."""

    async def test_get_stats(self, db, emb):
        for i in range(3):
            vec = await emb.embed(f"stat node {i}")
            n = MemoryNode(content_summary=f"stat node {i}", embedding=vec, source_session="t")
            await db.create_node(n)

        stats = await db.get_stats()
        assert stats["node_count"] == 3
        assert stats["edge_count"] == 0


class TestBenchmark:
    """Performance benchmarks for store and recall operations."""

    async def test_store_latency(self, handler):
        """Measure average store latency over 10 operations."""
        latencies = []
        for i in range(10):
            start = time.perf_counter()
            await handler.memory_store(content=f"Benchmark memory number {i} about various topics")
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)

        avg = sum(latencies) / len(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        print(f"\n[BENCH] memory_store: avg={avg*1000:.1f}ms, p95={p95*1000:.1f}ms (n=10)")
        # Just ensure it completes — no hard threshold
        assert avg < 5.0  # sanity: under 5 seconds per store

    async def test_recall_latency(self, handler):
        """Store 20 memories, then measure recall latency."""
        topics = [
            "Python programming and web development",
            "Machine learning with PyTorch",
            "Hiking in Yosemite National Park",
            "Cooking Italian pasta recipes",
            "Reading science fiction novels",
            "Playing guitar and music theory",
            "My cat's veterinary appointments",
            "Home renovation and painting",
            "Investing in stock market ETFs",
            "Learning Japanese language",
            "Running a half marathon",
            "Docker and Kubernetes deployment",
            "Family vacation to Hawaii",
            "Meditation and mindfulness practice",
            "Photography with mirrorless cameras",
            "Electric vehicle comparison research",
            "Sourdough bread baking techniques",
            "Chess strategy and openings",
            "Gardening and growing tomatoes",
            "Podcast recommendations for tech",
        ]
        for t in topics:
            await handler.memory_store(content=t)

        queries = [
            "What programming languages do I use?",
            "What are my hobbies?",
            "Tell me about my pets",
            "What sports do I do?",
            "What food do I cook?",
        ]

        latencies = []
        for q in queries:
            start = time.perf_counter()
            result = await handler.memory_recall(query=q, k=5)
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)
            print(f"\n[BENCH] recall '{q}': {elapsed*1000:.1f}ms, {result['count']} results")

        avg = sum(latencies) / len(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        print(f"\n[BENCH] memory_recall: avg={avg*1000:.1f}ms, p95={p95*1000:.1f}ms (n={len(latencies)})")
        assert avg < 5.0

    async def test_recall_accuracy_ranking(self, handler):
        """Store categorized memories and verify recall ranks the right one first."""
        memories = [
            ("I have a golden retriever named Max", "pet"),
            ("I work as a data scientist at Google", "work"),
            ("My favorite restaurant is the Italian place on Main Street", "food"),
            ("I run 5 miles every morning before work", "fitness"),
            ("I'm allergic to shellfish", "health"),
        ]
        ids = {}
        for content, tag in memories:
            r = await handler.memory_store(content=content)
            ids[tag] = r["node_id"]

        # Test that targeted queries rank the right memory first
        test_cases = [
            ("What pet do I have?", "pet", ["golden retriever", "max"]),
            ("Where do I work?", "work", ["data scientist", "google"]),
            ("What food allergies do I have?", "health", ["shellfish", "allergic"]),
        ]

        correct = 0
        for query, expected_tag, keywords in test_cases:
            recall = await handler.memory_recall(query=query, k=3)
            top = recall["results"][0]["content"].lower()
            hit = any(kw in top for kw in keywords)
            status = "PASS" if hit else "FAIL"
            print(f"\n[ACCURACY] '{query}' -> top='{recall['results'][0]['content'][:60]}' [{status}]")
            if hit:
                correct += 1

        accuracy = correct / len(test_cases)
        print(f"\n[ACCURACY] Overall: {correct}/{len(test_cases)} = {accuracy*100:.0f}%")
        # With semantic mock embeddings, we should get at least 2/3 right
        assert correct >= 2
