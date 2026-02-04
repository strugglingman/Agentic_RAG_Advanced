"""Slack Bot Adapter - Orchestrates all Slack bot functionality."""

import base64
import io
import logging
import re
from typing import Optional
from slack_sdk.web.async_client import AsyncWebClient
from redis.asyncio import Redis

from src.adapters.base_bot_adapter import BaseBotAdapter, BotResponse
from src.adapters.slack.slack_identity import SlackIdentityResolver
from src.adapters.slack.slack_file_handler import SlackFileHandler
from src.adapters.slack.slack_formatter import SlackFormatter
from src.config.settings import Config

logger = logging.getLogger(__name__)

# Regex to find file links like [filename.pdf](/api/files/xxx) or [text](/api/files/xxx)
FILE_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(/api/files/([a-zA-Z0-9_-]+)\)")

SLACK_CONV_KEY_PREFIX = "slack:conv:"


class SlackBotAdapter(BaseBotAdapter):
    """Slack bot adapter - handles messages and commands from Slack."""

    def __init__(self, redis: Optional[Redis] = None):
        super().__init__()
        self._client = AsyncWebClient(token=Config.SLACK_BOT_TOKEN)
        self._identity_resolver = SlackIdentityResolver(self._client)
        self._file_handler = SlackFileHandler(Config.SLACK_BOT_TOKEN)
        self._formatter = SlackFormatter()
        self._redis = redis
        self._conversation_cache: dict[str, str] = {}

    async def handle_message(self, event: dict) -> None:
        """Handle incoming message from Slack.

        Attachments are encoded as base64 and sent in /chat request body.
        """
        if self._is_bot_message(event):
            logger.debug("Ignoring bot message to prevent loops")
            return

        channel_id = event.get("channel", "")
        message_ts = event.get("ts", "")
        text = event.get("text", "")

        try:
            bot_user = await self._identity_resolver.resolve_identity(event)
            auth_token = self.generate_auth_token(bot_user.user_email, bot_user.dept_id)

            # Convert attachments to base64 format for /chat endpoint
            # Format: [{"filename": "...", "mime_type": "...", "data": "base64..."}]
            attachments = []
            downloaded = await self._file_handler.download_files_from_event(event)
            for content, filename, mime_type in downloaded:
                attachments.append(
                    {
                        "filename": filename,
                        "mime_type": mime_type or "application/octet-stream",
                        "data": base64.b64encode(content).decode("utf-8"),
                    }
                )

            # Get existing conversation or None for new
            conv_key = self._identity_resolver.get_conversation_key(
                channel_id, bot_user.user_email
            )
            existing_conv_id = await self._get_conversation(conv_key)

            # Show typing indicator
            await self._show_typing(channel_id, message_ts)

            logger.info(
                "[SLACK] Calling /chat endpoint for user=%s, conv=%s, attachments=%d",
                bot_user.user_email,
                existing_conv_id or "new",
                len(attachments),
            )

            # Call backend /chat endpoint with base64 attachments
            conv_id, answer, contexts = await self.call_chat_endpoint(
                message=text,
                conversation_id=existing_conv_id,
                attachments=attachments if attachments else None,
                auth_token=auth_token,
            )

            logger.info(
                "[SLACK] Chat response received: conv_id=%s, answer_len=%d, contexts=%d",
                conv_id,
                len(answer) if answer else 0,
                len(contexts) if contexts else 0,
            )

            # Store new conversation mapping
            if conv_id and conv_id != existing_conv_id:
                await self._store_conversation(conv_key, conv_id)

            # Remove typing indicator
            await self._remove_typing(channel_id, message_ts)

            # Extract file links from answer and prepare for Slack upload
            file_links = self._extract_file_links(answer)
            if file_links:
                # Remove file link markdown from answer text
                answer = self._remove_file_links(answer)

            # Send response
            response = BotResponse(text=answer, contexts=contexts)
            await self.send_response(channel_id, response, thread_id=message_ts)

            # Upload referenced files to Slack
            if file_links:
                await self._upload_files_to_slack(
                    file_links, channel_id, message_ts, auth_token
                )

        except Exception as e:
            logger.exception("Error handling message: %s", e)
            await self._remove_typing(channel_id, message_ts)
            # Build detailed error message
            error_msg = str(e)
            if not error_msg or error_msg == "None":
                # Try to get more details from exception
                error_msg = f"{type(e).__name__}"
                if hasattr(e, "response") and e.response is not None:
                    error_msg += f" (HTTP {e.response.status_code})"
                if hasattr(e, "args") and e.args:
                    error_msg += f": {e.args[0]}" if e.args[0] else ""
            error_response = BotResponse(text="", error=error_msg)
            await self.send_response(channel_id, error_response, thread_id=message_ts)

    async def send_response(
        self, channel_id: str, response: BotResponse, thread_id: str | None = None
    ) -> None:
        """Send formatted response to Slack."""
        formatted = self._formatter.format_response(response)
        answer_text = response.text or ""

        if len(answer_text) > 3000:
            chunks = self._formatter.split_long_message(answer_text)
            for i, chunk in enumerate(chunks):
                chunk_response = BotResponse(
                    text=chunk,
                    contexts=response.contexts if i == len(chunks) - 1 else None,
                )
                chunk_formatted = self._formatter.format_response(chunk_response)
                await self._client.chat_postMessage(
                    channel=channel_id,
                    thread_ts=thread_id,
                    **chunk_formatted,
                )
        else:
            await self._client.chat_postMessage(
                channel=channel_id,
                thread_ts=thread_id,
                **formatted,
            )

        # Upload files if any
        if response.files_to_upload:
            for file_path in response.files_to_upload:
                try:
                    await self._client.files_upload_v2(
                        channel=channel_id,
                        file=file_path,
                        thread_ts=thread_id,
                    )
                except Exception as e:
                    logger.error("Failed to upload file %s: %s", file_path, e)

    async def _show_typing(
        self, channel_id: str, message_ts: str | None = None
    ) -> None:
        """Show typing indicator by adding reaction to user's message."""
        if message_ts:
            try:
                await self._client.reactions_add(
                    channel=channel_id,
                    timestamp=message_ts,
                    name="hourglass_flowing_sand",
                )
            except Exception:
                pass

    async def _remove_typing(
        self, channel_id: str, message_ts: str | None = None
    ) -> None:
        """Remove typing indicator reaction."""
        if message_ts:
            try:
                await self._client.reactions_remove(
                    channel=channel_id,
                    timestamp=message_ts,
                    name="hourglass_flowing_sand",
                )
            except Exception:
                pass

    def _is_bot_message(self, event: dict) -> bool:
        """Check if message is from a bot to prevent infinite loops."""
        bot_id = event.get("bot_id")
        if bot_id is not None and str(bot_id).strip() != "":
            return True
        if event.get("subtype") == "bot_message":
            return True
        return False

    async def _get_conversation(self, conv_key: str) -> str | None:
        """Get conversation_id from Redis or fallback to in-memory cache."""
        redis_key = f"{SLACK_CONV_KEY_PREFIX}{conv_key}"

        if self._redis:
            try:
                conv_id = await self._redis.get(redis_key)
                if conv_id:
                    return conv_id.decode() if isinstance(conv_id, bytes) else conv_id
            except Exception as e:
                logger.warning("Redis get error for %s: %s", redis_key, e)

        return self._conversation_cache.get(conv_key)

    async def _store_conversation(self, conv_key: str, conv_id: str) -> None:
        """Store conversation_id in Redis and fallback cache."""
        redis_key = f"{SLACK_CONV_KEY_PREFIX}{conv_key}"

        if self._redis:
            try:
                await self._redis.setex(redis_key, Config.SLACK_CONV_TTL, conv_id)
            except Exception as e:
                logger.warning("Redis set error for %s: %s", redis_key, e)

        self._conversation_cache[conv_key] = conv_id

    def _extract_file_links(self, text: str) -> list[tuple[str, str]]:
        """
        Extract file links from response text.

        Returns:
            List of (display_name, file_id) tuples
        """
        return FILE_LINK_PATTERN.findall(text)

    def _remove_file_links(self, text: str) -> str:
        """
        Remove file link markdown from text, keeping just the filename.

        Converts [filename.pdf](/api/files/xxx) to just "filename.pdf"
        """
        return FILE_LINK_PATTERN.sub(r"\1", text)

    async def _upload_files_to_slack(
        self,
        file_links: list[tuple[str, str]],
        channel_id: str,
        thread_ts: str | None,
        auth_token: str,
    ) -> None:
        """
        Download files from backend and upload to Slack.

        Args:
            file_links: List of (display_name, file_id) tuples
            channel_id: Slack channel to upload to
            thread_ts: Thread timestamp to attach files to
            auth_token: JWT token for backend API
        """
        for display_name, file_id in file_links:
            try:
                content, filename, _ = await self.download_file_from_backend(
                    file_id, auth_token
                )
                if not content:
                    logger.warning(
                        "Failed to download file %s for Slack upload", file_id
                    )
                    continue

                # Use display_name if filename is generic
                if filename == "file" or not filename:
                    filename = display_name

                # Upload to Slack
                await self._client.files_upload_v2(
                    channel=channel_id,
                    file=io.BytesIO(content),
                    filename=filename,
                    thread_ts=thread_ts,
                )
                logger.info(
                    "Uploaded file %s to Slack channel %s", filename, channel_id
                )

            except Exception as e:
                logger.error("Failed to upload file %s to Slack: %s", file_id, e)
