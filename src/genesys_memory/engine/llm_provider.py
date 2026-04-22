"""LLM provider using Anthropic Claude for memory processing."""
from __future__ import annotations

import json

from genesys_memory.models.enums import EdgeType


class AnthropicLLMProvider:
    def __init__(self, api_key: str, model: str = "claude-haiku-4-5-20251001"):
        import anthropic
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model

    async def _ask(self, prompt: str, max_tokens: int = 1024) -> str:
        resp = await self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        block = resp.content[0]
        return str(getattr(block, "text", "")).strip()

    async def extract_entities(self, text: str) -> list[str]:
        prompt = (
            "Extract all named entities and key concepts from this text. "
            "Return as a JSON array of strings.\n"
            "Include: people's names, company names, locations, project names, technologies, specific topics.\n"
            "Exclude: generic nouns, adjectives, pronouns.\n\n"
            f"Text: {text}\n\n"
            "Respond with ONLY a JSON array, no other text."
        )
        raw = await self._ask(prompt, max_tokens=512)
        try:
            result: list[str] = json.loads(raw)
            return result
        except json.JSONDecodeError:
            return []

    async def classify_category(self, text: str) -> str | None:
        prompt = (
            "Classify this memory into exactly one category. "
            "Options: professional, medical, financial, educational, family, location, preference, project, or null if none fit.\n\n"
            f"Memory: {text}\n\n"
            "Respond with ONLY the category string or \"null\", no other text."
        )
        raw = await self._ask(prompt, max_tokens=64)
        raw = raw.strip().strip('"').lower()
        if raw == "null" or raw not in (
            "professional", "medical", "financial", "educational",
            "family", "location", "preference", "project",
        ):
            return None if raw == "null" else (raw if raw in (
                "professional", "medical", "financial", "educational",
                "family", "location", "preference", "project",
            ) else None)
        return raw

    async def detect_contradiction(self, memory_a: str, memory_b: str) -> tuple[bool, float, str | None]:
        prompt = (
            "Do these two memories contradict each other? "
            "A contradiction means they cannot both be true simultaneously.\n\n"
            f"Memory A: {memory_a}\nMemory B: {memory_b}\n\n"
            'Respond with ONLY a JSON object: {"contradicts": true/false, "confidence": 0.0-1.0, "reason": "brief explanation"}'
        )
        raw = await self._ask(prompt, max_tokens=256)
        try:
            data = json.loads(raw)
            return (
                bool(data.get("contradicts", False)),
                float(data.get("confidence", 0.0)),
                data.get("reason"),
            )
        except (json.JSONDecodeError, ValueError):
            return False, 0.0, None

    async def infer_causal_edges(
        self, new_memory: str, existing_memories: list[tuple[str, str]]
    ) -> list[tuple[str, EdgeType, float, str | None]]:
        if not existing_memories:
            return []
        mem_lines = "\n".join(f"ID: {mid} | Content: {content}" for mid, content in existing_memories)
        prompt = (
            "Given a new memory and a list of existing memories, identify causal relationships.\n\n"
            f"New memory: {new_memory}\n\nExisting memories:\n{mem_lines}\n\n"
            "For each causal relationship found, specify:\n"
            "- target_id: the ID of the existing memory\n"
            '- edge_type: one of "caused_by", "supports", "derived_from"\n'
            "- confidence: 0.0 to 1.0\n"
            "- reason: brief explanation of why this relationship exists\n\n"
            "Only include relationships with confidence > 0.6.\n"
            "Respond with ONLY a JSON array of objects, no other text."
        )
        raw = await self._ask(prompt, max_tokens=1024)
        try:
            data = json.loads(raw)
            results: list[tuple[str, EdgeType, float, str | None]] = []
            for item in data:
                edge_type_str = item.get("edge_type", "")
                try:
                    edge_type = EdgeType(edge_type_str)
                except ValueError:
                    continue
                confidence = float(item.get("confidence", 0.0))
                if confidence > 0.6:
                    results.append((item["target_id"], edge_type, confidence, item.get("reason")))
            return results
        except (json.JSONDecodeError, KeyError):
            return []

    async def consolidate(self, episodic_memories: list[str]) -> str:
        memories_text = "\n".join(f"- {m}" for m in episodic_memories)
        prompt = (
            "Consolidate these related episodic memories into a single semantic summary. "
            "Preserve key facts and relationships.\n\n"
            f"Memories:\n{memories_text}\n\n"
            "Respond with ONLY the consolidated summary (1-3 sentences), no other text."
        )
        return await self._ask(prompt, max_tokens=512)

    async def generate_summary(self, text: str) -> str:
        prompt = (
            "Summarize this memory in one line, max 200 characters.\n\n"
            f"Text: {text}\n\n"
            "Respond with ONLY the summary, no other text."
        )
        raw = await self._ask(prompt, max_tokens=256)
        return raw[:200]
