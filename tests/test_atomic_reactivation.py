"""Tests for atomic reactivation updates (task 2.14).

Verifies that concurrent reactivation updates don't lose data
and that timestamps are preserved in order.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta

import pytest

from genesys_memory.context import current_user_id
from genesys_memory.models.enums import MemoryStatus
from genesys_memory.models.node import MemoryNode
from genesys_memory.storage.memory import InMemoryGraphProvider


def _make_node(**kwargs) -> MemoryNode:
    defaults = {"content_summary": "test memory", "status": MemoryStatus.ACTIVE}
    defaults.update(kwargs)
    return MemoryNode(**defaults)


@pytest.fixture()
def _user_ctx():
    token = current_user_id.set("test-user")
    yield
    current_user_id.reset(token)


@pytest.mark.usefixtures("_user_ctx")
class TestAtomicReactivationUpdate:
    @pytest.mark.asyncio
    async def test_concurrent_reactivations_do_not_lose_updates(self):
        """100 concurrent atomic updates must all be reflected."""
        graph = InMemoryGraphProvider()
        await graph.initialize("test-user")
        node = _make_node(stability=1.0, reactivation_count=0, reactivation_timestamps=[])
        node_id = await graph.create_node(node)

        now = datetime.now(timezone.utc)

        async def bump(i: int) -> None:
            ts = now + timedelta(milliseconds=i)
            await graph.atomic_reactivation_update(node_id, ts, 0.1)

        await asyncio.gather(*(bump(i) for i in range(100)))

        updated = graph.nodes[node_id]
        assert updated.reactivation_count == 100
        assert len(updated.reactivation_timestamps) == 100
        assert updated.stability == pytest.approx(1.0 + 100 * 0.1, abs=1e-9)

    @pytest.mark.asyncio
    async def test_reactivation_timestamps_preserves_order(self):
        """Timestamps appended sequentially must remain in insertion order."""
        graph = InMemoryGraphProvider()
        await graph.initialize("test-user")
        node = _make_node(stability=1.0, reactivation_count=0, reactivation_timestamps=[])
        node_id = await graph.create_node(node)

        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for i in range(10):
            ts = base + timedelta(hours=i)
            await graph.atomic_reactivation_update(node_id, ts, 0.05)

        updated = graph.nodes[node_id]
        assert updated.reactivation_count == 10
        assert len(updated.reactivation_timestamps) == 10
        for i in range(9):
            assert updated.reactivation_timestamps[i] < updated.reactivation_timestamps[i + 1]
        assert updated.last_reactivated_at == base + timedelta(hours=9)

    @pytest.mark.asyncio
    async def test_atomic_update_nonexistent_node_is_noop(self):
        """Updating a missing node_id must not raise."""
        graph = InMemoryGraphProvider()
        await graph.initialize("test-user")
        now = datetime.now(timezone.utc)
        await graph.atomic_reactivation_update("nonexistent", now, 0.1)

    @pytest.mark.asyncio
    async def test_atomic_update_with_empty_timestamps(self):
        """Starting from empty timestamps list, a single update works correctly."""
        graph = InMemoryGraphProvider()
        await graph.initialize("test-user")
        node = _make_node(stability=1.0, reactivation_count=0, reactivation_timestamps=[])
        node_id = await graph.create_node(node)

        now = datetime.now(timezone.utc)
        await graph.atomic_reactivation_update(node_id, now, 0.1)

        updated = graph.nodes[node_id]
        assert updated.reactivation_count == 1
        assert updated.reactivation_timestamps == [now]
        assert updated.stability == pytest.approx(1.1, abs=1e-9)
