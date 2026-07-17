"""Tests for Phase 3 MCP tools."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from genesys_memory.context import current_user_id
from genesys_memory.mcp.tools import MCPToolHandler
from genesys_memory.models.enums import MemoryStatus, ReactivationPattern
from genesys_memory.models.node import MemoryNode


@pytest.fixture(autouse=True)
def _user_ctx():
    token = current_user_id.set("test-user")
    yield
    current_user_id.reset(token)


def _make_node(**kwargs) -> MemoryNode:
    defaults = {"content_summary": "test memory", "original_user_id": "test-user"}
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
        """Server should list 13 tools (11 original + promote_to_org + memory_amend)."""
        # Import and check tool listing
        from genesys_memory.server import list_tools
        tool_list = await list_tools()
        assert len(tool_list) == 13
        names = {t.name for t in tool_list}
        expected = {
            "memory_store", "memory_recall", "memory_search", "memory_traverse",
            "memory_explain", "pin_memory", "unpin_memory", "list_core_memories",
            "delete_memory", "memory_stats", "set_core_preferences",
            "promote_to_org", "memory_amend",
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
        graph.get_stats = AsyncMock(return_value={"max_causal_weight": 5})

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


# ---------------------------------------------------------------------------
# Field-feedback additions (F1–F8) — integration tests over the real
# in-memory provider, following the same pytest/asyncio patterns above.
# ---------------------------------------------------------------------------

import importlib

from genesys_memory.storage.cache import NullCacheProvider
from genesys_memory.storage.memory import InMemoryGraphProvider


class _StubEmbedder:
    """Deterministic embedder: identical text → identical vector (cosine 1.0).

    ``recommended_autolink_min_similarity`` can be set to exercise the
    embedder-recommendation precedence branch.
    """

    def __init__(self, recommended_autolink=None, distinct=False):
        self._recommended = recommended_autolink
        self._distinct = distinct
        self._i = 0

    @property
    def dimension(self):
        return 8

    async def embed(self, text):
        # Same text → same vector. Distinct mode gives every call its own axis
        # so unrelated stores don't auto-link.
        import hashlib
        seed = int(hashlib.sha256(text.encode()).hexdigest(), 16)
        vec = [0.0] * 8
        vec[seed % 8] = 1.0
        return vec

    async def embed_batch(self, texts):
        return [await self.embed(t) for t in texts]

    def __getattr__(self, name):
        if name == "recommended_autolink_min_similarity" and self._recommended is not None:
            return self._recommended
        raise AttributeError(name)


class _ConstEmbedder:
    """Returns the same vector for any text, so any pair is an auto-link candidate."""

    @property
    def dimension(self):
        return 8

    async def embed(self, text):
        return [1.0] + [0.0] * 7

    async def embed_batch(self, texts):
        return [await self.embed(t) for t in texts]


async def _real_handler(embedder=None):
    graph = InMemoryGraphProvider()
    await graph.initialize("test-user")
    return MCPToolHandler(graph=graph, embeddings=embedder, cache=NullCacheProvider())


class TestScoreLegibilityF1:
    @pytest.mark.asyncio
    async def test_explain_includes_score_model_and_activation_alias(self):
        h = await _real_handler()
        stored = await h.memory_store("A durable fact about the project")
        ex = await h.memory_explain(stored["node_id"])
        assert ex["activation"] == ex["decay_score"]
        forces = ex["score_model"]["forces"]
        assert 0.0 <= forces["connectivity_factor"] <= 1.0
        assert 0.0 <= forces["activation_factor"] <= 1.0
        assert "formula" in ex["score_model"] and "staleness_note" in ex["score_model"]

    @pytest.mark.asyncio
    async def test_recall_hit_has_activation_alias_equal_to_decay_score(self):
        h = await _real_handler()
        await h.memory_store("Genesys stores causal memories")
        rec = await h.memory_recall("causal memories")
        assert rec["results"], "expected a keyword hit"
        for hit in rec["results"]:
            assert hit["activation"] == hit["decay_score"]


class TestAutolinkF2F6:
    @pytest.mark.asyncio
    async def test_autolink_respects_env_override(self, monkeypatch):
        monkeypatch.setenv("GENESYS_AUTOLINK_MIN_SIMILARITY", "1.01")
        from genesys_memory.engine import config
        importlib.reload(config)
        try:
            h = await _real_handler(_StubEmbedder())
            a = await h.memory_store("identical content")
            b = await h.memory_store("identical content")
            edges = await h.graph.get_edges(b["node_id"], "both")
            assert not any(e.created_by == "auto_link" for e in edges)
        finally:
            monkeypatch.delenv("GENESYS_AUTOLINK_MIN_SIMILARITY", raising=False)
            importlib.reload(config)

    @pytest.mark.asyncio
    async def test_autolink_uses_embedder_recommendation(self):
        # cosine of same text is 1.0; recommendation 1.01 blocks the link.
        h = await _real_handler(_StubEmbedder(recommended_autolink=1.01))
        await h.memory_store("same topic sentence")
        b = await h.memory_store("same topic sentence")
        edges = await h.graph.get_edges(b["node_id"], "both")
        assert not any(e.created_by == "auto_link" for e in edges)

    @pytest.mark.asyncio
    async def test_autolink_capped_at_max_edges(self):
        from genesys_memory.engine import config
        h = await _real_handler(_StubEmbedder())
        for _ in range(6):
            await h.memory_store("cap test content")
        last = await h.memory_store("cap test content")
        edges = await h.graph.get_edges(last["node_id"], "both")
        auto = [e for e in edges if e.created_by == "auto_link"]
        assert len(auto) <= config.AUTOLINK_MAX_EDGES

    @pytest.mark.asyncio
    async def test_autolink_skips_pair_with_existing_explicit_edge(self):
        h = await _real_handler(_StubEmbedder())
        a = await h.memory_store("shared subject")
        b = await h.memory_store("shared subject", related_to=[a["node_id"]])
        edges = await h.graph.get_all_edges([a["node_id"], b["node_id"]])
        pair = [e for e in edges
                if {str(e.source_id), str(e.target_id)} == {a["node_id"], b["node_id"]}]
        assert len(pair) == 1
        assert pair[0].type.value == "caused_by"
        assert not any(e.type.value == "related_to" for e in pair)


class TestTraverseF3:
    @pytest.mark.asyncio
    async def test_traverse_returns_edges_between_returned_nodes(self):
        h = await _real_handler()
        a = await h.memory_store("node A")
        b = await h.memory_store("node B", related=[{"id": a["node_id"], "type": "caused_by"}])
        c = await h.memory_store("node C", related=[{"id": b["node_id"], "type": "caused_by"}])
        tr = await h.memory_traverse(a["node_id"], depth=2)
        assert "edges" in tr and tr["edge_count"] >= 1
        for e in tr["edges"]:
            assert {"source", "target", "type", "weight", "created_by"} <= set(e.keys())

    @pytest.mark.asyncio
    async def test_traverse_edge_types_filter_applies_to_nodes_and_edges(self):
        h = await _real_handler()
        a = await h.memory_store("center")
        b = await h.memory_store("caused", related=[{"id": a["node_id"], "type": "caused_by"}])
        c = await h.memory_store("related", related=[{"id": a["node_id"], "type": "related_to"}])
        tr = await h.memory_traverse(a["node_id"], depth=2, edge_types=["caused_by"])
        node_ids = {n["id"] for n in tr["nodes"]}
        assert c["node_id"] not in node_ids
        assert all(e["type"] == "caused_by" for e in tr["edges"])


class TestWriteSideF4:
    @pytest.mark.asyncio
    async def test_related_param_creates_typed_edge(self):
        h = await _real_handler()
        a = await h.memory_store("original figure")
        b = await h.memory_store("new figure", related=[{"id": a["node_id"], "type": "supersedes"}])
        edges = await h.graph.get_edges(b["node_id"], "out", None)
        sup = [e for e in edges if e.type.value == "supersedes"]
        assert len(sup) == 1
        assert str(sup[0].source_id) == b["node_id"]
        assert str(sup[0].target_id) == a["node_id"]

    @pytest.mark.asyncio
    async def test_related_invalid_type_errors_before_node_creation(self):
        h = await _real_handler()
        a = await h.memory_store("anchor")
        before = len(h.graph.nodes)
        r = await h.memory_store("bad", related=[{"id": a["node_id"], "type": "nope"}])
        assert "error" in r and "valid_types" in r
        assert len(h.graph.nodes) == before

    @pytest.mark.asyncio
    async def test_related_to_still_creates_caused_by(self):
        h = await _real_handler()
        a = await h.memory_store("cause")
        b = await h.memory_store("effect", related_to=[a["node_id"]])
        edges = await h.graph.get_edges(b["node_id"], "out")
        assert any(e.type.value == "caused_by" for e in edges)

    @pytest.mark.asyncio
    async def test_memory_amend_creates_supersedes_and_preserves_old_status(self):
        h = await _real_handler()
        a = await h.memory_store("Revenue is 100")
        old_status = (await h.graph.get_node(a["node_id"])).status
        am = await h.memory_amend(a["node_id"], "Revenue is 200", reason="restated")
        assert am["status"] == "amended" and am["supersedes"] == a["node_id"]
        sup = [e for e in await h.graph.get_edges(am["node_id"], "out")
               if e.type.value == "supersedes" and str(e.target_id) == a["node_id"]]
        assert len(sup) == 1 and sup[0].weight == 1.0 and sup[0].reason == "restated"
        assert (await h.graph.get_node(a["node_id"])).status == old_status

    @pytest.mark.asyncio
    async def test_recall_marks_and_decays_superseded_node(self):
        # ConstEmbedder gives both nodes a nonzero (equal) vector score so the
        # multiplicative SUPERSEDED_DECAY actually reorders them; with no
        # embedder the scores are 0 and the penalty is a no-op.
        h = await _real_handler(_ConstEmbedder())
        a = await h.memory_store("Revenue is 100")
        am = await h.memory_amend(a["node_id"], "Revenue is 200", reason="restated")
        rec = await h.memory_recall("revenue")
        by_id = {hit["id"]: hit for hit in rec["results"]}
        assert a["node_id"] in by_id and am["node_id"] in by_id, "expected both nodes recalled"
        # Old node is tagged with the superseding node id...
        assert by_id[a["node_id"]]["superseded_by"] == am["node_id"]
        # ...and the SUPERSEDED_DECAY penalty ranks it below the new node.
        old_idx = next(i for i, hit in enumerate(rec["results"]) if hit["id"] == a["node_id"])
        new_idx = next(i for i, hit in enumerate(rec["results"]) if hit["id"] == am["node_id"])
        assert new_idx < old_idx
        # The superseding node itself carries no superseded_by marker.
        assert "superseded_by" not in by_id[am["node_id"]]

    @pytest.mark.asyncio
    async def test_amend_requires_ownership(self):
        h = await _real_handler()
        a = await h.memory_store("owned by someone else")
        node = await h.graph.get_node(a["node_id"])
        await h.graph.update_node(a["node_id"], {"original_user_id": "other-user"})
        r = await h.memory_amend(a["node_id"], "hijack")
        assert "error" in r

    @pytest.mark.asyncio
    async def test_store_surfaces_possible_conflicts(self):
        h = await _real_handler(_ConstEmbedder())
        await h.memory_store("Budget is 50")
        second = await h.memory_store("Budget is 75")
        assert "possible_conflicts" in second
        assert any(c["signal"] == "numeric_mismatch" for c in second["possible_conflicts"])
        edges = await h.graph.get_all_edges()
        assert not any(e.type.value == "contradicts" for e in edges)


class TestHeuristicConflictSignal:
    def test_numeric(self):
        from genesys_memory.engine.contradiction import heuristic_conflict_signal
        assert heuristic_conflict_signal("it costs 50", "it costs 75") == "numeric_mismatch"

    def test_negation(self):
        from genesys_memory.engine.contradiction import heuristic_conflict_signal
        assert heuristic_conflict_signal("the deal is on", "the deal is not on") == "negation"

    def test_none(self):
        from genesys_memory.engine.contradiction import heuristic_conflict_signal
        assert heuristic_conflict_signal("blue sky", "green grass") is None


class TestCategoryF5:
    @pytest.mark.asyncio
    async def test_store_with_category_persists_and_filters(self):
        h = await _real_handler(_StubEmbedder())
        stored = await h.memory_store("work note", category="professional")
        node = await h.graph.get_node(stored["node_id"])
        assert node.category == "professional"
        await h.pin_memory(stored["node_id"])
        core = await h.list_core_memories(category="professional")
        assert any(m["id"] == stored["node_id"] for m in core["core_memories"])


class TestConciseRecallF7:
    def test_truncate_summary_word_boundary(self):
        from genesys_memory.mcp.tools import _truncate_summary
        content = "word " * 60  # 300 chars
        s = _truncate_summary(content, 200)
        assert len(s) <= 200 and s.endswith("…")
        assert not s[:-1].endswith("wor")
        assert _truncate_summary("short text", 200) == "short text"

    @pytest.mark.asyncio
    async def test_recall_concise_shape(self):
        h = await _real_handler()
        await h.memory_store("concise recall subject matter")
        rec = await h.memory_recall("subject matter", verbosity="concise")
        allowed = {"id", "summary", "status", "score", "activation", "is_core", "superseded_by"}
        for hit in rec["results"]:
            assert set(hit.keys()) <= allowed
            assert "content" not in hit and "causal_basis" not in hit

    @pytest.mark.asyncio
    async def test_recall_full_unchanged_default(self):
        h = await _real_handler()
        await h.memory_store("full recall subject matter")
        rec = await h.memory_recall("subject matter")
        assert rec["results"]
        assert "content" in rec["results"][0]


class TestStorePositionalCompat:
    @pytest.mark.asyncio
    async def test_created_at_still_fourth_positional_arg(self):
        """Published-API compat: memory_store(content, session, related_to,
        created_at, ...) must keep its original positional order — new params
        (`related`, `category`) are appended at the end."""
        from datetime import datetime, timezone
        h = await _real_handler()
        stored = await h.memory_store("a dated fact", "sess-1", None, "2026-01-01T00:00:00")
        node = await h.graph.get_node(stored["node_id"])
        assert node.created_at == datetime(2026, 1, 1, tzinfo=timezone.utc)
        assert node.category is None  # nothing bled into the new params


class TestTraverseProviderCompatF3:
    @pytest.mark.asyncio
    async def test_traverse_degrades_without_get_connecting_edges(self):
        """External providers (genesys-server backends) may not implement the
        new get_connecting_edges Protocol method — traverse must degrade to
        empty edges, not raise AttributeError."""
        h = await _real_handler()
        a = await h.memory_store("node A")
        b = await h.memory_store("node B", related=[{"id": a["node_id"], "type": "caused_by"}])

        class _LegacyGraph:
            """Delegates to the real provider but hides get_connecting_edges."""

            def __init__(self, inner):
                self._inner = inner

            def __getattr__(self, name):
                if name == "get_connecting_edges":
                    raise AttributeError(name)
                return getattr(self._inner, name)

        h.graph = _LegacyGraph(h.graph)
        tr = await h.memory_traverse(a["node_id"], depth=2)
        assert tr["count"] == 2
        assert tr["edges"] == []
        assert tr["edge_count"] == 0


class TestAutolinkNodeDegreeCapF2:
    @pytest.mark.asyncio
    async def test_incoming_autolink_accumulation_is_bounded(self, monkeypatch):
        from genesys_memory.engine import config
        monkeypatch.setattr(config, "AUTOLINK_MAX_NODE_DEGREE", 2)
        h = await _real_handler(_ConstEmbedder())
        first = await h.memory_store("hub content 0")
        for i in range(1, 8):
            await h.memory_store(f"hub content {i}")
        # Without the cap the first node would accrete one edge per later store.
        first_auto = [
            e for e in await h.graph.get_edges(first["node_id"], "both")
            if e.created_by == "auto_link"
        ]
        assert len(first_auto) <= 2
        # No node exceeds max(per-store fan-out, per-node cap).
        bound = max(config.AUTOLINK_MAX_EDGES, 2)
        for nid in list(h.graph.nodes):
            auto = [e for e in await h.graph.get_edges(nid, "both") if e.created_by == "auto_link"]
            assert len(auto) <= bound


class _MidSimEmbedder:
    """Pairwise cosine ~0.5 between the two test texts: above the recall/
    conflict floor (0.3) but below the auto-link floor (0.9)."""

    recommended_autolink_min_similarity = 0.9
    recommended_min_similarity = 0.3

    @property
    def dimension(self):
        return 8

    async def embed(self, text):
        import math
        s = math.sqrt(0.5)
        vec = [0.0] * 8
        vec[0] = s
        vec[2 if "50" in text else 3] = s
        return vec

    async def embed_batch(self, texts):
        return [await self.embed(t) for t in texts]


class TestConflictScanDecoupledF4:
    @pytest.mark.asyncio
    async def test_conflict_hint_fires_below_autolink_floor(self):
        """A changed figure whose similarity lands between the conflict floor
        and the (stricter) auto-link floor must still produce a hint — and no
        auto-link."""
        h = await _real_handler(_MidSimEmbedder())
        first = await h.memory_store("Budget is 50")
        second = await h.memory_store("Budget is 75")
        assert "possible_conflicts" in second
        assert any(
            c["id"] == first["node_id"] and c["signal"] == "numeric_mismatch"
            for c in second["possible_conflicts"]
        )
        edges = await h.graph.get_all_edges()
        assert not any(e.created_by == "auto_link" for e in edges)


class TestNumericMismatchContext:
    def test_unrelated_numbers_do_not_fire(self):
        from genesys_memory.engine.contradiction import heuristic_conflict_signal
        # A date in one text and an ID in the other share no numeric context.
        assert heuristic_conflict_signal("meeting on 2026", "invoice 12345") is None

    def test_same_context_differing_numbers_fire(self):
        from genesys_memory.engine.contradiction import heuristic_conflict_signal
        assert heuristic_conflict_signal("costs 50 in 2026", "costs 75 in 2026") == "numeric_mismatch"

    def test_identical_numbers_do_not_fire(self):
        from genesys_memory.engine.contradiction import heuristic_conflict_signal
        assert heuristic_conflict_signal("costs 50", "it costs 50") is None


class TestSearchEnumerationF8:
    @pytest.mark.asyncio
    async def test_empty_query_enumerates_without_embedder(self):
        from datetime import datetime, timezone
        h = await _real_handler()  # no embedder configured
        a = await h.memory_store("older change")
        b = await h.memory_store("newer change")
        await h.graph.update_node(a["node_id"], {"last_reactivated_at": datetime(2020, 1, 1, tzinfo=timezone.utc)})
        await h.graph.update_node(b["node_id"], {"last_reactivated_at": datetime(2030, 1, 1, tzinfo=timezone.utc)})

        res = await h.memory_search("", k=10)
        ids = [m["id"] for m in res["results"]]
        assert ids[:2] == [b["node_id"], a["node_id"]]  # recency desc
        assert all("last_reactivated_at" in m and "source_session" in m for m in res["results"])

        # active_since narrows to what changed after the cursor — the
        # "what's new since I last looked" path, no seed query needed.
        res = await h.memory_search("", filters={"active_since": "2025-01-01"}, k=10)
        ids = {m["id"] for m in res["results"]}
        assert b["node_id"] in ids and a["node_id"] not in ids

    @pytest.mark.asyncio
    async def test_enumeration_honors_status_filter(self):
        h = await _real_handler()
        a = await h.memory_store("will be core")
        await h.memory_store("stays active")
        await h.pin_memory(a["node_id"])
        res = await h.memory_search("", filters={"status": ["core"]}, k=10)
        assert {m["id"] for m in res["results"]} == {a["node_id"]}


class TestSearchTzNaiveFilters:
    @pytest.mark.asyncio
    async def test_naive_iso_dates_do_not_raise(self):
        h = await _real_handler(_StubEmbedder())
        await h.memory_store("alpha subject")
        # Tz-naive ISO input previously raised TypeError against tz-aware
        # node timestamps; it must be treated as UTC.
        res = await h.memory_search("alpha", filters={"active_since": "2050-01-01"}, k=10)
        assert res["results"] == []
        res = await h.memory_search("alpha", filters={"since": "2000-01-01"}, k=10)
        assert res["count"] >= 1


class TestServerCallToolErrorsF9:
    @pytest.mark.asyncio
    async def test_missing_required_argument_is_structured(self):
        import json
        from genesys_memory.server import call_tool
        res = await call_tool("memory_explain", {})
        payload = json.loads(res[0].text)
        assert "missing required argument" in payload["error"]
        assert payload["retryable"] is False

    @pytest.mark.asyncio
    async def test_unknown_tool_is_structured(self):
        import json
        from genesys_memory.server import call_tool
        res = await call_tool("not_a_tool", {})
        payload = json.loads(res[0].text)
        assert "error" in payload and payload["retryable"] is False

    @pytest.mark.asyncio
    async def test_tool_exception_returns_error_payload_with_retryable_flag(self, monkeypatch):
        import json
        import genesys_memory.server as srv

        async def boom(**kwargs):
            raise RuntimeError("boom")

        # Read tool → retryable
        monkeypatch.setitem(srv._TOOL_DISPATCH, "memory_recall", (boom, ["query"], {}))
        res = await srv.call_tool("memory_recall", {"query": "x"})
        payload = json.loads(res[0].text)
        assert "boom" in payload["error"] and payload["retryable"] is True

        # Write tool → not retryable (may have partially applied)
        monkeypatch.setitem(srv._TOOL_DISPATCH, "memory_store", (boom, ["content"], {}))
        res = await srv.call_tool("memory_store", {"content": "x"})
        payload = json.loads(res[0].text)
        assert "boom" in payload["error"] and payload["retryable"] is False


class TestSearchActiveSinceF8:
    @pytest.mark.asyncio
    async def test_search_active_since_filters_by_reactivation(self):
        from datetime import datetime, timezone
        h = await _real_handler(_StubEmbedder())
        old = await h.memory_store("alpha subject")
        fresh = await h.memory_store("alpha subject two")
        # Reactivate only `fresh`.
        marker = datetime.now(timezone.utc).isoformat()
        await h.memory_recall("alpha")  # reactivates both hits...
        # Instead assert the filter mechanics directly: set fresh's reactivation
        # far in the future and old's in the past.
        past = datetime(2000, 1, 1, tzinfo=timezone.utc)
        future = datetime(2100, 1, 1, tzinfo=timezone.utc)
        await h.graph.update_node(old["node_id"], {"last_reactivated_at": past})
        await h.graph.update_node(fresh["node_id"], {"last_reactivated_at": future})
        res = await h.memory_search("alpha", filters={"active_since": "2050-01-01T00:00:00+00:00"}, k=10)
        ids = {m["id"] for m in res["results"]}
        assert fresh["node_id"] in ids
        assert old["node_id"] not in ids


class TestRetestRound2Fixes:
    """Fixes from the 0.4.0 field retest: conflict precision, supersede
    direction/visibility, and the latent recall superseder down-ranking."""

    def test_conflict_ignores_cross_quantity_numbers(self):
        from genesys_memory.engine.contradiction import heuristic_conflict_signal
        # 200ms latency vs $50,000 budget: different quantities, no conflict.
        assert heuristic_conflict_signal(
            "API latency is 200ms after the caching fix",
            "Project budget is $50,000 for Q3",
        ) is None

    def test_conflict_ignores_different_units_same_anchor(self):
        from genesys_memory.engine.contradiction import heuristic_conflict_signal
        assert heuristic_conflict_signal("the migration took 6 weeks", "the migration took 8 months") is None

    def test_conflict_fires_same_anchor_same_unit(self):
        from genesys_memory.engine.contradiction import heuristic_conflict_signal
        assert heuristic_conflict_signal("the budget is $50,000", "the budget is $80,000") == "numeric_mismatch"

    async def test_explain_edge_direction_disambiguates_supersedes(self):
        h = await _real_handler(_ConstEmbedder())
        old = await h.memory_store("Rate is $100")
        amended = await h.memory_amend(old["node_id"], "Rate is now $150", reason="rate change")
        explained = await h.memory_explain(old["node_id"])
        sup = [e for e in explained["edges"] if e["type"] == "supersedes"]
        assert sup and sup[0]["direction"] == "incoming"  # the OTHER node supersedes this one
        explained_new = await h.memory_explain(amended["node_id"])
        sup_new = [e for e in explained_new["edges"] if e["type"] == "supersedes"]
        assert sup_new and sup_new[0]["direction"] == "outgoing"

    async def test_search_enumeration_tags_superseded(self):
        h = await _real_handler(_ConstEmbedder())
        old = await h.memory_store("Deadline is June")
        new = await h.memory_amend(old["node_id"], "Deadline is October")
        result = await h.memory_search("")
        by_id = {m["id"]: m for m in result["results"]}
        assert by_id[old["node_id"]].get("superseded_by") == new["node_id"]
        assert "superseded_by" not in by_id[new["node_id"]]

    async def test_recall_never_downranks_the_superseder(self):
        h = await _real_handler(_ConstEmbedder())
        old = await h.memory_store("Server count is 5")
        new = await h.memory_amend(old["node_id"], "Server count is 9")
        # Cap results so the superseded old node falls OUT of the result set:
        # the old elif then marked the SUPERSEDER as superseded_by the old.
        result = await h.memory_recall("server count", max_results=1)
        top = result["results"][0]
        assert top["id"] == new["node_id"]
        assert "superseded_by" not in top


class TestTraverseIncludesStartNode:
    """F3 round-3: the induced subgraph must include the start node, so its
    incident edges (incl. an edge_types-filtered match) are not dropped."""

    async def test_traverse_returns_start_incident_edges(self):
        h = await _real_handler(_ConstEmbedder())
        a = await h.memory_store("Anchor node A")
        b = await h.memory_store("Node B", related=[{"id": a["node_id"], "type": "supports"}])
        c = await h.memory_store("Node C", related=[{"id": a["node_id"], "type": "supersedes"}])
        # Traverse from A: A's own edges (A->B supports via B's edge, C->A supersedes) must appear.
        res = await h.memory_traverse(a["node_id"], depth=2)
        node_ids = {n["id"] for n in res["nodes"]}
        assert a["node_id"] in node_ids  # start node present
        edge_pairs = {(e["source"], e["target"], e["type"]) for e in res["edges"]}
        # B was stored with a supports edge B->A; that edge is start-incident.
        assert any(a["node_id"] in (s, t) for s, t, _ in edge_pairs), "start-incident edges missing"

    async def test_traverse_edge_type_filter_returns_start_incident_match(self):
        h = await _real_handler(_ConstEmbedder())
        a = await h.memory_store("Fact v1")
        # Amend creates a new node with a supersedes edge -> A (start-incident).
        amended = await h.memory_amend(a["node_id"], "Fact v2")
        res = await h.memory_traverse(a["node_id"], depth=2, edge_types=["supersedes"])
        assert res["edge_count"] >= 1, "edge_types filter dropped the only (start-incident) supersedes edge"
        assert all(e["type"] == "supersedes" for e in res["edges"])
