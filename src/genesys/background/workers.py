"""Background workers using Redis Streams event bus."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Callable

import redis.asyncio as aioredis

from genesys.core_memory.promoter import evaluate_core_promotion, promote_to_core
from genesys.engine.contradiction import detect_contradictions
from genesys.engine.reactivation import cascade_reactivate
from genesys.models.enums import MemoryStatus
from genesys.models.edge import MemoryEdge
from genesys.storage.base import EmbeddingProvider, GraphStorageProvider, LLMProvider

logger = logging.getLogger(__name__)


class RedisEventBus:
    """EventBusProvider implementation using Redis Streams."""

    def __init__(self, host: str = "localhost", port: int = 6379):
        self._redis = aioredis.Redis(host=host, port=port, decode_responses=True)
        self._handlers: dict[str, list[Callable]] = {}
        self._running = False
        self._consumer_group = "genesys_workers"
        self._consumer_name = "worker_0"

    async def publish(self, channel: str, payload: dict) -> None:
        await self._redis.xadd(channel, {"data": json.dumps(payload)})

    async def subscribe(self, channel: str, handler: Callable) -> None:
        self._handlers.setdefault(channel, []).append(handler)
        try:
            await self._redis.xgroup_create(channel, self._consumer_group, id="0", mkstream=True)
        except Exception:
            pass  # Group already exists

    async def start(self) -> None:
        self._running = True
        asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        self._running = False

    async def _poll_loop(self) -> None:
        streams = {ch: ">" for ch in self._handlers}
        while self._running:
            try:
                results = await self._redis.xreadgroup(
                    self._consumer_group,
                    self._consumer_name,
                    streams,
                    count=10,
                    block=1000,
                )
                for stream_name, messages in results:
                    for msg_id, fields in messages:
                        payload = json.loads(fields["data"])
                        for handler in self._handlers.get(stream_name, []):
                            try:
                                await handler(payload)
                            except Exception:
                                logger.exception("Handler error on %s", stream_name)
                        await self._redis.xack(stream_name, self._consumer_group, msg_id)
            except Exception:
                if self._running:
                    await asyncio.sleep(1)


class BackgroundWorkers:
    """Registers and manages background event handlers."""

    def __init__(
        self,
        graph: GraphStorageProvider,
        embeddings: EmbeddingProvider,
        llm: LLMProvider,
        event_bus: RedisEventBus,
    ):
        self.graph = graph
        self.embeddings = embeddings
        self.llm = llm
        self.event_bus = event_bus

    async def setup(self) -> None:
        await self.event_bus.subscribe("memory.created", self._on_memory_created)
        await self.event_bus.subscribe("memory.accessed", self._on_memory_accessed)
        await self.event_bus.start()

    async def _on_memory_created(self, payload: dict) -> None:
        node_id = payload["node_id"]
        content = payload.get("content_full", "")

        node = await self.graph.get_node(node_id)
        if not node:
            return

        # Extract entities
        try:
            entities = await self.llm.extract_entities(content)
            if entities:
                await self.graph.update_node(node_id, {"entity_refs": entities})
        except Exception:
            logger.exception("Entity extraction failed for %s", node_id)

        # Classify category
        try:
            category = await self.llm.classify_category(content)
            if category:
                await self.graph.update_node(node_id, {"category": category})
        except Exception:
            logger.exception("Category classification failed for %s", node_id)

        # Generate summary
        try:
            summary = await self.llm.generate_summary(content)
            if summary:
                await self.graph.update_node(node_id, {"content_summary": summary})
        except Exception:
            logger.exception("Summary generation failed for %s", node_id)

        # Infer causal edges
        try:
            recent = []
            for status in (MemoryStatus.ACTIVE, MemoryStatus.EPISODIC, MemoryStatus.SEMANTIC):
                recent.extend(await self.graph.get_nodes_by_status(status, limit=50))
            existing = [(str(n.id), n.content_summary) for n in recent if str(n.id) != node_id]
            if existing:
                edges = await self.llm.infer_causal_edges(content, existing[:50])
                for target_id, edge_type, confidence in edges:
                    edge = MemoryEdge(
                        source_id=node.id,
                        target_id=__import__("uuid").UUID(target_id),
                        type=edge_type,
                        weight=confidence,
                    )
                    await self.graph.create_edge(edge)
        except Exception:
            logger.exception("Causal inference failed for %s", node_id)

        # Contradiction detection
        try:
            # Re-fetch node to get latest updates
            node = await self.graph.get_node(node_id)
            if node:
                await detect_contradictions(node, self.graph, self.embeddings, self.llm)
        except Exception:
            logger.exception("Contradiction detection failed for %s", node_id)

        # Core promotion check
        try:
            node = await self.graph.get_node(node_id)
            if node:
                should_promote, reason = await evaluate_core_promotion(node, self.graph)
                if should_promote and reason:
                    await promote_to_core(node_id, reason, self.graph)
        except Exception:
            logger.exception("Core promotion check failed for %s", node_id)

    async def _on_memory_accessed(self, payload: dict) -> None:
        from datetime import datetime, timezone

        node_id = payload["node_id"]
        node = await self.graph.get_node(node_id)
        if not node:
            return

        now = datetime.now(timezone.utc)

        # Update reactivation
        new_count = node.reactivation_count + 1
        timestamps = node.reactivation_timestamps + [now]
        timestamps = timestamps[-50:]  # Rolling window

        # Reclassify pattern
        if new_count == 1:
            pattern = "single"
        elif len(timestamps) >= 5:
            # Check spacing: steady if spaced > 1 day apart on average
            diffs = [(timestamps[i] - timestamps[i - 1]).total_seconds() for i in range(1, len(timestamps))]
            avg_gap = sum(diffs) / len(diffs) if diffs else 0
            pattern = "steady" if avg_gap > 86400 else "burst"
        else:
            pattern = "burst"

        updates = {
            "reactivation_count": new_count,
            "last_accessed_at": now,
            "last_reactivated_at": now,
            "reactivation_pattern": pattern,
            "reactivation_timestamps": [t.isoformat() for t in timestamps],
        }
        await self.graph.update_node(node_id, updates)

        # Cascade reactivation
        try:
            await cascade_reactivate(node_id, self.graph)
        except Exception:
            logger.exception("Cascade reactivation failed for %s", node_id)
