"""
Async Redis Client Factory.

Creates Redis client with connection pooling for DI container.
Uses redis.asyncio for pure async operations - no event loop issues.
"""

import logging
import redis.asyncio as redis
from redis.asyncio import Redis
from src.config.settings import Config

logger = logging.getLogger(__name__)


async def create_redis_client() -> Redis:
    """
    Create async Redis client with connection pool.

    Returns:
        Redis: Connected async Redis client

    Raises:
        redis.ConnectionError: If Redis is not reachable

    Note:
        - Uses connection pooling (automatic with from_url)
        - decode_responses=True for automatic string decoding
        - Tests connection with ping() before returning
    """
    client = redis.from_url(
        Config.REDIS_URL,
        encoding="utf-8",
        decode_responses=True,
        socket_timeout=5.0,
        socket_connect_timeout=5.0,
    )

    # Test connection
    await client.ping()
    logger.info(f"[Redis] Connected to {Config.REDIS_URL}")

    return client


async def close_redis_client(client: Redis) -> None:
    """
    Close Redis client connection.

    Args:
        client: Redis client to close

    Note:
        Should be called on application shutdown.
    """
    if client:
        await client.close()
        logger.info("[Redis] Connection closed")