"""Flat vector memory baseline for benchmark comparison."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from genesys_memory.engine.scoring import cosine_similarity
from genesys_memory.storage.base import EmbeddingProvider


@dataclass
class FlatMemoryEntry:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    embedding: list[float] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class FlatVectorMemory:
    """Simple in-memory vector store with cosine similarity retrieval.

    Uses the same EmbeddingProvider as Genesys for fair comparison.
    No causal edges, no scoring, no status transitions — just vectors.
    """

    def __init__(self, embeddings: EmbeddingProvider):
        self.embeddings = embeddings
        self.memories: list[FlatMemoryEntry] = []

    async def store(self, content: str) -> str:
        """Store a memory with its embedding. Returns the entry ID."""
        embedding = await self.embeddings.embed(content)
        entry = FlatMemoryEntry(content=content, embedding=embedding)
        self.memories.append(entry)
        return entry.id

    async def recall(self, query: str, k: int = 10) -> list[dict]:
        """Retrieve top-k memories by cosine similarity."""
        if not self.memories:
            return []

        query_embedding = await self.embeddings.embed(query)

        scored = []
        for entry in self.memories:
            if not entry.embedding:
                continue
            sim = cosine_similarity(query_embedding, entry.embedding)
            scored.append((entry, sim))

        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for entry, score in scored[:k]:
            results.append({
                "id": entry.id,
                "content": entry.content,
                "score": round(score, 4),
                "created_at": entry.created_at.isoformat(),
            })

        return results

    def clear(self) -> None:
        """Clear all stored memories."""
        self.memories.clear()
