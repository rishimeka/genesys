from __future__ import annotations

import hashlib

from openai import AsyncOpenAI

from genesys.storage.cache import RedisCacheProvider


class OpenAIEmbeddingProvider:
    MODEL = "text-embedding-3-small"
    DIMENSION = 1536

    def __init__(self, api_key: str, cache: RedisCacheProvider | None = None):
        self._client = AsyncOpenAI(api_key=api_key)
        self._cache = cache

    @property
    def dimension(self) -> int:
        return self.DIMENSION

    def _cache_key(self, text: str) -> str:
        return f"embed:{hashlib.sha256(text.encode()).hexdigest()}"

    async def embed(self, text: str) -> list[float]:
        if self._cache:
            import json
            cached = await self._cache.get(self._cache_key(text))
            if cached:
                return json.loads(cached)

        # text-embedding-3-small has 8192 token limit; code can be ~2 chars/token
        if len(text) > 8000:
            text = text[:8000]
        response = await self._client.embeddings.create(input=[text], model=self.MODEL)
        vec = response.data[0].embedding

        if self._cache:
            import json
            await self._cache.set(self._cache_key(text), json.dumps(vec), ttl_seconds=86400)

        return vec

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = await self._client.embeddings.create(input=texts, model=self.MODEL)
        return [d.embedding for d in sorted(response.data, key=lambda d: d.index)]
