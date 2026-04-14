#!/usr/bin/env python3
"""Migrate memories from JSON (in-memory backend) to Postgres.

Usage:
    DATABASE_URL=postgresql://... python scripts/migrate_to_postgres.py [--user-id USER_ID] [--file PATH]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from datetime import datetime, timezone

import asyncpg


def parse_datetime(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


async def migrate(database_url: str, json_path: str, user_id: str) -> None:
    with open(json_path) as f:
        data = json.load(f)

    nodes = data.get("nodes", {})
    edges = data.get("edges", {})

    if not nodes:
        print("No nodes found in JSON file.")
        return

    pool = await asyncpg.create_pool(database_url, min_size=1, max_size=5)

    async with pool.acquire() as conn:
        # Prepare node records
        node_records = []
        for nid, n in nodes.items():
            embedding = n.get("embedding")
            emb_str = None
            if embedding:
                emb_str = "[" + ",".join(str(f) for f in embedding) + "]"

            node_records.append((
                uuid.UUID(nid),
                user_id,
                n.get("status", "active"),
                n.get("content_summary", ""),
                n.get("content_full"),
                emb_str,
                n.get("category"),
                n.get("entity_refs", []),
                float(n.get("decay_score", 1.0)),
                int(n.get("causal_weight", 0)),
                int(n.get("reactivation_count", 0)),
                n.get("reactivation_pattern", "single"),
                bool(n.get("pinned", False)),
                n.get("promotion_reason"),
                n.get("source_agent", "claude"),
                n.get("source_session", ""),
                parse_datetime(n.get("created_at")) or datetime.now(timezone.utc),
                parse_datetime(n.get("last_accessed_at")),
                parse_datetime(n.get("last_reactivated_at")),
                json.dumps(n.get("metadata", {})),
            ))

        print(f"Inserting {len(node_records)} nodes...")
        for rec in node_records:
            await conn.execute(
                """INSERT INTO memory_nodes
                   (id, user_id, status, content_summary, content_full, embedding,
                    category, entity_refs, decay_score, causal_weight,
                    reactivation_count, reactivation_pattern, pinned, promotion_reason,
                    source_agent, source_session, created_at, last_accessed_at,
                    last_reactivated_at, metadata)
                   VALUES ($1,$2,$3,$4,$5,$6::vector,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20)
                   ON CONFLICT (id) DO NOTHING""",
                *rec,
            )

        # Prepare edge records (edges is a list of dicts)
        edge_records = []
        for e in edges:
            edge_records.append((
                uuid.UUID(e["id"]),
                user_id,
                uuid.UUID(e["source_id"]),
                uuid.UUID(e["target_id"]),
                e.get("type", "related_to"),
                float(e.get("weight", 1.0)),
                parse_datetime(e.get("created_at")) or datetime.now(timezone.utc),
            ))

        print(f"Inserting {len(edge_records)} edges...")
        for rec in edge_records:
            await conn.execute(
                """INSERT INTO memory_edges (id, user_id, source_id, target_id, type, weight, created_at)
                   VALUES ($1,$2,$3,$4,$5,$6,$7)
                   ON CONFLICT (source_id, target_id, type) DO NOTHING""",
                *rec,
            )

    await pool.close()
    print(f"Done. Migrated {len(node_records)} nodes and {len(edge_records)} edges for user '{user_id}'.")


def main():
    parser = argparse.ArgumentParser(description="Migrate memories from JSON to Postgres")
    parser.add_argument("--user-id", default="default_user", help="User ID to assign to all memories")
    parser.add_argument("--file", default="data/memories.json", help="Path to memories JSON file")
    args = parser.parse_args()

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL environment variable required", file=sys.stderr)
        sys.exit(1)

    asyncio.run(migrate(database_url, args.file, args.user_id))


if __name__ == "__main__":
    main()
