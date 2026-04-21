from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from genesys_memory.storage.base import CacheProvider


class LocalEmbeddingProvider:
    """Local embedding provider using sentence-transformers (all-MiniLM-L6-v2).

    No API key required. Model is lazy-loaded on first embed() call.
    """

    DIMENSION = 384

    def __init__(self):
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")

    @property
    def dimension(self) -> int:
        return self.DIMENSION

    async def embed(self, text: str) -> list[float]:
        self._load_model()
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self._load_model()
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vecs]


class OpenAIEmbeddingProvider:
    MODEL = "text-embedding-3-small"
    DIMENSION = 1536

    def __init__(self, api_key: str, cache: CacheProvider | None = None):
        from openai import AsyncOpenAI
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
