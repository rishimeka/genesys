from __future__ import annotations


class NullCacheProvider:
    """No-op cache for when Redis is not available."""

    async def get(self, key: str) -> str | None:
        return None

    async def set(self, key: str, value: str, ttl_seconds: int = 300) -> None:
        pass

    async def delete(self, key: str) -> None:
        pass

    async def exists(self, key: str) -> bool:
        return False


class RedisCacheProvider:
    def __init__(self, host: str = "localhost", port: int = 6379):
        import redis.asyncio as aioredis
        self._redis = aioredis.Redis(host=host, port=port, decode_responses=True)

    async def get(self, key: str) -> str | None:
        return await self._redis.get(key)

    async def set(self, key: str, value: str, ttl_seconds: int = 300) -> None:
        await self._redis.set(key, value, ex=ttl_seconds)

    async def delete(self, key: str) -> None:
        await self._redis.delete(key)

    async def exists(self, key: str) -> bool:
        return bool(await self._redis.exists(key))
