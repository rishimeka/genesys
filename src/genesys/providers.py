"""Shared provider singleton for both MCP and FastAPI servers."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

from genesys.mcp.tools import MCPToolHandler
from genesys.retrieval.embedding import OpenAIEmbeddingProvider
from genesys.storage.base import CacheProvider, EmbeddingProvider, EventBusProvider, GraphStorageProvider

load_dotenv()


@dataclass
class Providers:
    graph: GraphStorageProvider
    cache: CacheProvider
    embeddings: EmbeddingProvider
    llm: object | None
    event_bus: EventBusProvider | None
    tools: MCPToolHandler
    user_id: str  # Default user_id for backwards compat; context var overrides at runtime


_instance: Providers | None = None


def get_providers() -> Providers:
    """Return the shared provider singleton, creating it on first call."""
    global _instance
    if _instance is not None:
        return _instance

    backend = os.getenv("GENESYS_BACKEND", "memory")
    user_id = os.getenv("GENESYS_USER_ID", "default_user")

    if backend == "postgres":
        from genesys.storage.memory import InMemoryCacheProvider
        from genesys.storage.postgres import PostgresGraphProvider

        graph = PostgresGraphProvider()
        cache = InMemoryCacheProvider()  # TODO: swap to Redis if needed
        embeddings = OpenAIEmbeddingProvider(api_key=os.getenv("OPENAI_API_KEY", ""))
    elif backend == "memory":
        from genesys.storage.memory import InMemoryCacheProvider, InMemoryGraphProvider

        default_persist = os.path.join(os.path.dirname(__file__), "..", "..", "data", "memories.json")
        graph = InMemoryGraphProvider(persist_path=os.getenv("GENESYS_PERSIST_PATH", default_persist))
        cache = InMemoryCacheProvider()
        embeddings = OpenAIEmbeddingProvider(api_key=os.getenv("OPENAI_API_KEY", ""))
    else:
        from genesys.storage.cache import RedisCacheProvider
        from genesys.storage.falkordb import FalkorDBProvider

        host = os.getenv("FALKORDB_HOST", "localhost")
        port = int(os.getenv("FALKORDB_PORT", "6379"))
        graph = FalkorDBProvider(host=host, port=port)
        cache = RedisCacheProvider(host=host, port=port)
        embeddings = OpenAIEmbeddingProvider(api_key=os.getenv("OPENAI_API_KEY", ""), cache=cache)

    llm_provider = None
    event_bus = None

    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    if anthropic_key:
        from genesys.engine.llm_provider import AnthropicLLMProvider
        llm_provider = AnthropicLLMProvider(api_key=anthropic_key)

        redis_url = os.getenv("REDIS_URL", "")
        if backend != "memory" and redis_url:
            from genesys.background.workers import RedisEventBus
            host = os.getenv("FALKORDB_HOST", "localhost")
            port = int(os.getenv("FALKORDB_PORT", "6379"))
            event_bus = RedisEventBus(host=host, port=port)

    tools = MCPToolHandler(graph=graph, embeddings=embeddings, cache=cache, event_bus=event_bus)

    _instance = Providers(
        graph=graph,
        cache=cache,
        embeddings=embeddings,
        llm=llm_provider,
        event_bus=event_bus,
        tools=tools,
        user_id=user_id,
    )
    return _instance
