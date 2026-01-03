"""Slack File Handler - Downloads files from Slack messages."""

import logging
import httpx
from src.config.settings import Config

logger = logging.getLogger(__name__)


class SlackFileHandler:
    """Downloads files from Slack messages."""

    def __init__(self, bot_token: str):
        self._bot_token = bot_token

    async def download_file(self, file_info: dict) -> tuple[bytes, str, str]:
        file_url = file_info.get("url_private", "")
        filename = file_info.get("name", "unknown")
        is_valid, err = self.validate_file(file_info)
        if not is_valid:
            raise ValueError(err)

        headers = {"Authorization": f"Bearer {self._bot_token}"}
        mimetype = file_info.get("mimetype", "application/octet-stream")
        async with httpx.AsyncClient() as client:
            response = await client.get(file_url, headers=headers)
            response.raise_for_status()
            content = response.content

            return content, filename, mimetype

    async def download_files_from_event(
        self, event: dict
    ) -> list[tuple[bytes, str, str]]:
        files = event.get("files", [])
        downloaded_files = []
        for file_info in files:
            try:
                content, filename, mimetype = await self.download_file(file_info)
                downloaded_files.append((content, filename, mimetype))
            except Exception as e:
                logger.error(f"Failed to download file: {str(e)}")
                continue

        return downloaded_files

    def validate_file(self, file_info: dict) -> tuple[bool, str]:
        size = file_info.get("size", 0) / 1024 / 1024
        mimetype = file_info.get("mimetype", "application/octet-stream")
        if size > Config.MAX_DOWNLOAD_SIZE_MB:
            return (
                False,
                f"File size {size:.2f}MB exceeds limit of {Config.MAX_DOWNLOAD_SIZE_MB}MB.",
            )
        if mimetype not in Config.MIME_TYPES:
            return (
                False,
                f"File type {mimetype} not supported. Allowed types: {', '.join(Config.MIME_TYPES)}.",
            )

        return True, ""
