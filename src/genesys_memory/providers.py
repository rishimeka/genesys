"""Simplified provider singleton for standalone genesys-memory use.

Supports only the in-memory backend with optional local/OpenAI embeddings.
For production backends (Postgres, FalkorDB, etc.), use genesys-server.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from genesys_memory.mcp.tools import MCPToolHandler
from genesys_memory.storage.base import CacheProvider, EmbeddingProvider, EventBusProvider, GraphStorageProvider
from genesys_memory.storage.memory import InMemoryCacheProvider, InMemoryGraphProvider

load_dotenv()


@dataclass
class Providers:
    graph: GraphStorageProvider
    cache: CacheProvider
    embeddings: EmbeddingProvider
    llm: object | None
    event_bus: EventBusProvider | None
    tools: MCPToolHandler
    user_id: str


_instance: Providers | None = None


def _make_embedder() -> EmbeddingProvider:
    """Create embedding provider based on GENESYS_EMBEDDER env var."""
    embedder = os.getenv("GENESYS_EMBEDDER", "openai")
    if embedder == "local":
        from genesys_memory.retrieval.embedding import LocalEmbeddingProvider
        return LocalEmbeddingProvider()
    else:
        from genesys_memory.retrieval.embedding import OpenAIEmbeddingProvider
        return OpenAIEmbeddingProvider(api_key=os.getenv("OPENAI_API_KEY", ""))


def get_providers() -> Providers:
    """Return the shared provider singleton, creating it on first call."""
    global _instance
    if _instance is not None:
        return _instance

    user_id = os.getenv("GENESYS_USER_ID", "default_user")

    default_persist = str(Path(__file__).parent.parent.parent / "data" / "memories.json")
    graph = InMemoryGraphProvider(persist_path=os.getenv("GENESYS_PERSIST_PATH", default_persist))
    cache = InMemoryCacheProvider()
    embeddings = _make_embedder()

    llm_provider = None
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    if anthropic_key:
        from genesys_memory.engine.llm_provider import AnthropicLLMProvider
        llm_provider = AnthropicLLMProvider(api_key=anthropic_key)

    tools = MCPToolHandler(graph=graph, embeddings=embeddings, cache=cache)

    _instance = Providers(
        graph=graph,
        cache=cache,
        embeddings=embeddings,
        llm=llm_provider,
        event_bus=None,
        tools=tools,
        user_id=user_id,
    )
    return _instance
