"""asyncpg connection pool singleton."""
from __future__ import annotations

import os
import ssl

import asyncpg

_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    """Return the shared connection pool, creating it on first call."""
    global _pool
    if _pool is not None:
        return _pool

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL environment variable is required for postgres backend")

    # Enable SSL for cloud Postgres (Neon, Supabase, etc.)
    ssl_context = None
    if "sslmode=require" in database_url or ".neon.tech" in database_url:
        ssl_context = ssl.create_default_context()
        # Strip sslmode param — asyncpg uses the ssl kwarg instead
        database_url = database_url.replace("?sslmode=require", "").replace("&sslmode=require", "")

    _pool = await asyncpg.create_pool(
        database_url,
        min_size=2,
        max_size=20,
        command_timeout=30,
        ssl=ssl_context,
    )

    # Register pgvector type codec
    async with _pool.acquire() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await conn.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')

    return _pool


async def close_pool() -> None:
    """Close the connection pool."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
