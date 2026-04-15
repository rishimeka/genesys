"""File watcher for Obsidian vault changes using watchdog."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)


class _VaultHandler(FileSystemEventHandler):
    """Collects changed .md file paths with debouncing."""

    def __init__(self, vault_path: Path, debounce_seconds: float = 0.5):
        self.vault_path = vault_path
        self.debounce = debounce_seconds
        self._pending: set[str] = set()
        self._lock = asyncio.Lock() if asyncio.get_event_loop().is_running() else None
        self._callback: callable | None = None

    def _should_ignore(self, path: str) -> bool:
        rel = Path(path).relative_to(self.vault_path)
        parts = rel.parts
        return any(p.startswith(".") for p in parts) or not path.endswith(".md")

    def on_created(self, event: FileSystemEvent) -> None:
        self._handle(event.src_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        self._handle(event.src_path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        self._handle(event.src_path)

    def _handle(self, path: str) -> None:
        if self._should_ignore(path):
            return
        rel = str(Path(path).relative_to(self.vault_path))
        self._pending.add(rel)
        logger.debug("Vault change detected: %s", rel)


class VaultWatcher:
    """Watches an Obsidian vault for file changes and triggers re-indexing."""

    def __init__(self, vault_path: str | Path, provider):
        self.vault_path = Path(vault_path)
        self._provider = provider
        self._handler = _VaultHandler(self.vault_path)
        self._observer = Observer()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        """Start watching the vault directory."""
        self._observer.schedule(self._handler, str(self.vault_path), recursive=True)
        self._observer.start()
        # Start async poll loop
        try:
            loop = asyncio.get_running_loop()
            self._task = loop.create_task(self._poll_loop())
        except RuntimeError:
            pass
        logger.info("Vault watcher started for %s", self.vault_path)

    def stop(self) -> None:
        """Stop watching."""
        if self._task:
            self._task.cancel()
        self._observer.stop()
        self._observer.join(timeout=2)
        logger.info("Vault watcher stopped")

    async def _poll_loop(self) -> None:
        """Periodically flush pending changes to the provider."""
        while True:
            await asyncio.sleep(self._handler.debounce)
            if self._handler._pending:
                changed = list(self._handler._pending)
                self._handler._pending.clear()
                try:
                    await self._provider._incremental_index(changed)
                    # Update manifest
                    from genesys.storage.vault_manifest import write_manifest
                    await write_manifest(self._provider)
                    logger.info("Re-indexed %d changed files", len(changed))
                except Exception:
                    logger.exception("Error during incremental re-index")
