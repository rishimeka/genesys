"""Tests for logging behavior (task 2.6).

Verifies that swallowed exceptions are logged and that
configure_logging() is idempotent.
"""
from __future__ import annotations

import logging
from unittest.mock import AsyncMock

import pytest

from genesys_memory.context import current_org_ids, current_user_id
from genesys_memory.mcp.tools import MCPToolHandler
from genesys_memory.models.node import MemoryNode


def _make_node(**kwargs) -> MemoryNode:
    defaults = {"content_summary": "test memory"}
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


@pytest.fixture()
def _user_ctx():
    token = current_user_id.set("test-user")
    org_token = current_org_ids.set([])
    yield
    current_user_id.reset(token)
    current_org_ids.reset(org_token)


class TestAutoLinkFailureIsLogged:
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_user_ctx")
    async def test_auto_link_exception_is_logged(self, caplog):
        """When auto-linking raises, the warning is logged with exc_info."""
        handler, graph, embeddings, _ = _make_handler()
        graph.create_node = AsyncMock(return_value="node-1")
        graph.vector_search = AsyncMock(side_effect=RuntimeError("boom"))
        graph.is_orphan = AsyncMock(return_value=True)

        with caplog.at_level(logging.WARNING, logger="genesys_memory.mcp.tools"):
            result = await handler.memory_store("test content")

        assert result["status"] == "stored"
        assert any("Auto-linking failed" in r.message for r in caplog.records)


class TestLoggerDoesNotDoubleConfigure:
    def test_configure_logging_idempotent(self):
        """Calling configure_logging twice must not add a second handler."""
        from genesys_memory import configure_logging

        pkg_logger = logging.getLogger("genesys_memory")
        original_count = len(pkg_logger.handlers)

        configure_logging(logging.DEBUG)
        after_first = len(pkg_logger.handlers)

        configure_logging(logging.DEBUG)
        after_second = len(pkg_logger.handlers)

        assert after_first <= original_count + 1
        assert after_second == after_first
