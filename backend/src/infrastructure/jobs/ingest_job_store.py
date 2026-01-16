"""
Simple Redis-based job tracking for ingestion progress.

Tracks:
- Active jobs (which files are being ingested)
- Cancel requests
"""

import json
import logging
from typing import Optional, List, TYPE_CHECKING
from src.config.settings import Config

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class IngestJobStore:
    """
    Simple Redis wrapper for tracking ingestion jobs.
    """

    KEY_PREFIX = "ingest:"
    TTL = Config.REDIS_CACHE_TTL

    def __init__(self, redis: "Redis"):
        self._redis = redis

    async def start_job(
        self,
        job_id: str,
        user_email: str,
        file_ids: List[str],
    ) -> None:
        """
        Mark a job as active.

        Args:
            job_id: Unique job identifier
            user_email: User who started the job
            file_ids: List of file IDs being ingested
        """
        pipe = self._redis.pipeline()
        pipe.setex(f"{self.KEY_PREFIX}active:{job_id}", self.TTL, user_email)
        pipe.setex(f"{self.KEY_PREFIX}files:{job_id}", self.TTL, json.dumps(file_ids))
        await pipe.execute()
        logger.info(f"[IngestJob] Started job {job_id} for {len(file_ids)} files")

    async def is_cancelled(self, job_id: str) -> bool:
        """Check if cancellation was requested."""
        result = await self._redis.exists(f"{self.KEY_PREFIX}cancel:{job_id}")
        return result > 0

    async def cancel_job(self, job_id: str) -> bool:
        """
        Request cancellation of a job.

        Returns True if job was active, False otherwise.
        """
        # Check if job is active
        is_active = await self._redis.exists(f"{self.KEY_PREFIX}active:{job_id}")
        if not is_active:
            return False

        await self._redis.setex(f"{self.KEY_PREFIX}cancel:{job_id}", self.TTL, "1")
        logger.info(f"[IngestJob] Cancel requested for job {job_id}")
        return True

    async def complete_job(self, job_id: str) -> None:
        """Mark job as complete and cleanup Redis keys."""
        keys = [
            f"{self.KEY_PREFIX}active:{job_id}",
            f"{self.KEY_PREFIX}cancel:{job_id}",
            f"{self.KEY_PREFIX}files:{job_id}",
        ]
        await self._redis.delete(*keys)
        logger.info(f"[IngestJob] Completed job {job_id}")

    async def get_active_files(self, job_id: str) -> List[str]:
        """Get list of file IDs being processed by a job."""
        data = await self._redis.get(f"{self.KEY_PREFIX}files:{job_id}")
        if data:
            return json.loads(data)
        return []

    async def get_job_owner(self, job_id: str) -> Optional[str]:
        """Get the user email who owns the job."""
        return await self._redis.get(f"{self.KEY_PREFIX}active:{job_id}")

    async def is_file_being_ingested(self, file_id: str) -> bool:
        """
        Check if a specific file is currently being ingested.

        Scans all active jobs to find if file_id is in any of them.
        Note: This is O(n) but n is small (few active jobs at a time).
        """
        # Get all active job keys
        pattern = f"{self.KEY_PREFIX}files:*"
        cursor = 0
        while True:
            cursor, keys = await self._redis.scan(cursor, match=pattern, count=100)
            for key in keys:
                data = await self._redis.get(key)
                if data:
                    file_ids = json.loads(data)
                    if file_id in file_ids:
                        return True
            if cursor == 0:
                break
        return False

    async def get_all_active_file_ids(self) -> List[str]:
        """
        Get all file IDs currently being ingested across all jobs.

        Used by frontend to disable buttons for files in progress.
        """
        all_file_ids = []
        pattern = f"{self.KEY_PREFIX}files:*"
        cursor = 0
        while True:
            cursor, keys = await self._redis.scan(cursor, match=pattern, count=100)
            for key in keys:
                data = await self._redis.get(key)
                if data:
                    file_ids = json.loads(data)
                    all_file_ids.extend(file_ids)
            if cursor == 0:
                break
        return list(set(all_file_ids))  # Dedupe
