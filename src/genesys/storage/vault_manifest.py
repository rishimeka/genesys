"""Write .genesys/status.json after each sync cycle."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


async def write_manifest(provider, recent_events: list[str] | None = None) -> None:
    """Write status.json to the vault's .genesys/ directory."""
    stats = await provider.get_stats()

    status_summary = {}
    for key, val in stats.items():
        if key not in ("total_edges", "total_nodes"):
            status_summary[key] = val

    manifest = {
        "last_sync": datetime.now(timezone.utc).isoformat(),
        "node_count": stats.get("total_nodes", 0),
        "edge_count": stats.get("total_edges", 0),
        "status_summary": status_summary,
        "recent_events": recent_events or [],
    }

    out_path = Path(provider.db_dir) / "status.json"
    out_path.write_text(json.dumps(manifest, indent=2))
