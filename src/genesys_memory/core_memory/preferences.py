"""User preference management for core memory categories."""
from __future__ import annotations

from genesys_memory.storage.base import CacheProvider

_PREFS_KEY = "core_memory:preferences"


class CoreMemoryPreferences:
    """Stores user preferences for core memory auto-promotion."""

    def __init__(self, cache: CacheProvider):
        self.cache = cache
        self.auto_categories: list[str] = ["professional", "educational", "family", "location"]
        self.approval_categories: list[str] = []
        self.excluded_categories: list[str] = []

    async def load(self) -> None:
        import json
        raw = await self.cache.get(_PREFS_KEY)
        if raw:
            data = json.loads(raw)
            self.auto_categories = data.get("auto", self.auto_categories)
            self.approval_categories = data.get("approval", self.approval_categories)
            self.excluded_categories = data.get("excluded", self.excluded_categories)

    async def save(self) -> None:
        import json
        data = {
            "auto": self.auto_categories,
            "approval": self.approval_categories,
            "excluded": self.excluded_categories,
        }
        await self.cache.set(_PREFS_KEY, json.dumps(data), ttl_seconds=0)

    async def update(
        self,
        auto: list[str] | None = None,
        approval: list[str] | None = None,
        excluded: list[str] | None = None,
    ) -> dict[str, list[str]]:
        if auto is not None:
            self.auto_categories = auto
        if approval is not None:
            self.approval_categories = approval
        if excluded is not None:
            self.excluded_categories = excluded
        await self.save()
        return {
            "auto": self.auto_categories,
            "approval": self.approval_categories,
            "excluded": self.excluded_categories,
        }

    def is_eligible(self, category: str | None) -> bool:
        if category is None:
            return False
        if category in self.excluded_categories:
            return False
        return category in self.auto_categories
