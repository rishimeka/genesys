"""Tests for conversation ingestion parsers."""
from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from genesys.ingestion.claude import ingest_claude_export
from genesys.ingestion.chatgpt import ingest_chatgpt_export


def _make_claude_export(messages: list[dict]) -> dict:
    return {
        "name": "Test Conversation",
        "created_at": "2025-01-15T10:00:00+00:00",
        "chat_messages": [
            {"sender": "human", "text": m["content"]}
            for m in messages
        ],
    }


def _make_chatgpt_export(messages: list[dict]) -> dict:
    mapping = {}
    parent_id = None
    for i, m in enumerate(messages):
        node_id = f"node-{i}"
        mapping[node_id] = {
            "parent": parent_id,
            "children": [f"node-{i+1}"] if i < len(messages) - 1 else [],
            "message": {
                "author": {"role": "user"},
                "content": {"parts": [m["content"]]},
            },
        }
        parent_id = node_id
    return {
        "title": "Test Chat",
        "create_time": 1705312800,  # 2024-01-15
        "mapping": mapping,
    }


class TestClaudeIngestion:
    @pytest.mark.asyncio
    async def test_claude_export_ingestion(self):
        messages = [
            {"content": "I work at Acme Corp"},
            {"content": "My role is engineering manager"},
            {"content": "I manage a team of 5"},
        ]
        export = _make_claude_export(messages)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([export], f)
            path = f.name

        graph = AsyncMock()
        graph.create_node = AsyncMock(side_effect=lambda n: str(n.id))
        graph.create_edge = AsyncMock(return_value="edge-id")

        embeddings = AsyncMock()
        embeddings.embed_batch = AsyncMock(return_value=[[0.1] * 1536] * 3)

        result = await ingest_claude_export(path, graph, embeddings)

        assert result["memories_created"] == 3
        assert result["edges_created"] == 2  # 2 temporal edges for 3 messages
        assert graph.create_node.call_count == 3
        assert graph.create_edge.call_count == 2

        Path(path).unlink()

    @pytest.mark.asyncio
    async def test_ingestion_respects_date_limit(self):
        messages = [{"content": "old message"}]
        export = _make_claude_export(messages)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([export], f)
            path = f.name

        graph = AsyncMock()
        embeddings = AsyncMock()

        # Set since to after the conversation date
        since = datetime(2026, 1, 1, tzinfo=timezone.utc)
        result = await ingest_claude_export(path, graph, embeddings, since=since)

        assert result["memories_created"] == 0
        assert result["edges_created"] == 0

        Path(path).unlink()


class TestChatGPTIngestion:
    @pytest.mark.asyncio
    async def test_chatgpt_export_ingestion(self):
        messages = [
            {"content": "Hello AI"},
            {"content": "Tell me about Python"},
        ]
        export = _make_chatgpt_export(messages)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([export], f)
            path = f.name

        graph = AsyncMock()
        graph.create_node = AsyncMock(side_effect=lambda n: str(n.id))
        graph.create_edge = AsyncMock(return_value="edge-id")

        embeddings = AsyncMock()
        embeddings.embed_batch = AsyncMock(return_value=[[0.1] * 1536] * 2)

        result = await ingest_chatgpt_export(path, graph, embeddings)

        assert result["memories_created"] == 2
        assert result["edges_created"] == 1
        assert graph.create_node.call_count == 2

        Path(path).unlink()

    @pytest.mark.asyncio
    async def test_chatgpt_respects_date_limit(self):
        messages = [{"content": "old chat"}]
        export = _make_chatgpt_export(messages)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([export], f)
            path = f.name

        graph = AsyncMock()
        embeddings = AsyncMock()

        since = datetime(2026, 1, 1, tzinfo=timezone.utc)
        result = await ingest_chatgpt_export(path, graph, embeddings, since=since)

        assert result["memories_created"] == 0

        Path(path).unlink()
