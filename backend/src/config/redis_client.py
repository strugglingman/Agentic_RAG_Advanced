"""
Redis client singleton for caching conversation history.
"""

import redis.asyncio as redis
from src.config.settings import Config


class RedisClient:
    """Singleton Redis client."""

    _instance = None

    def __new__(cls):
        """
        TODO: Implement singleton pattern
        - Check if _instance is None
        - If None, create new instance
        - Initialize redis client using redis.from_url(Config.REDIS_URL)
        - Set decode_responses=True to auto-decode bytes to strings
        - Return the instance
        """
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.client = redis.from_url(
                Config.REDIS_URL, decode_responses=True
            )
        return cls._instance

    async def ping(self):
        """
        TODO: Test Redis connection
        - Try to ping the redis server
        - Return True if successful, False otherwise
        - Handle exceptions gracefully
        """
        try:
            return await self.client.ping()
        except redis.RedisError:
            return False

    @property
    def redis(self):
        return self.client


# Singleton instance getter
def get_redis():
    """
    TODO: Get Redis client singleton
    - Create and return RedisClient instance
    """
    return RedisClient()
