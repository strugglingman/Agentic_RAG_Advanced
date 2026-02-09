"""Slack Bot Adapter - Orchestrates all Slack bot functionality."""

import base64
import io
import json
import logging
import re
from slack_sdk.web.async_client import AsyncWebClient

from src.adapters.base_bot_adapter import BaseBotAdapter, BotResponse
from src.adapters.slack.slack_identity import SlackIdentityResolver
from src.adapters.slack.slack_file_handler import SlackFileHandler
from src.adapters.slack.slack_formatter import SlackFormatter
from src.config.settings import Config
from src.utils.history_utils import determine_message_count

logger = logging.getLogger(__name__)

# Regex to find file links like [filename.pdf](/api/files/xxx) or [text](/api/files/xxx)
FILE_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(/api/files/([a-zA-Z0-9_-]+)\)")


class SlackBotAdapter(BaseBotAdapter):
    """Slack bot adapter - handles messages and commands from Slack."""

    def __init__(self):
        super().__init__()
        self._client = AsyncWebClient(token=Config.SLACK_BOT_TOKEN)
        self._identity_resolver = SlackIdentityResolver(self._client)
        self._file_handler = SlackFileHandler(Config.SLACK_BOT_TOKEN)
        self._formatter = SlackFormatter()

    async def handle_message(self, event: dict) -> None:
        """Handle incoming message from Slack.

        Attachments are encoded as base64 and sent in /chat request body.
        Channel history is fetched on-demand from Slack API (not stored in DB).
        """
        if self._is_bot_message(event):
            logger.debug("Ignoring bot message to prevent loops")
            return

        channel_id = event.get("channel", "")
        message_ts = event.get("ts", "")
        text = event.get("text", "")
        source_channel_id = f"slack:{channel_id}"

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

            # Show typing indicator
            await self._show_typing(channel_id, message_ts)

            # Fetch channel history from Slack API (on-demand, not stored in DB)
            conversation_history = await self._fetch_channel_history(
                channel_id, text, message_ts
            )

            logger.info(
                "[SLACK] Calling /chat endpoint for user=%s, channel=%s, history=%d, attachments=%d",
                bot_user.user_email,
                source_channel_id,
                len(conversation_history),
                len(attachments),
            )

            # Call backend /chat endpoint with channel history and source_channel_id
            # Backend resolves conversation via source_channel_id (no Redis mapping needed)
            conv_id, answer, contexts, hitl_data = await self.call_chat_endpoint(
                message=text,
                conversation_id=None,
                attachments=attachments if attachments else None,
                auth_token=auth_token,
                source_channel_id=source_channel_id,
                conversation_history=conversation_history,
            )

            # Remove typing indicator
            await self._remove_typing(channel_id, message_ts)

            # Handle HITL interrupt — send confirmation buttons instead of final answer
            if hitl_data:
                await self._send_hitl_confirmation(
                    channel_id=channel_id,
                    thread_ts=message_ts,
                    answer=answer,
                    hitl_data=hitl_data,
                    conv_id=conv_id,
                    user_email=bot_user.user_email,
                    dept_id=bot_user.dept_id,
                )
                return

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

    async def handle_hitl_response(
        self,
        channel_id: str,
        message_ts: str,
        thread_ts: str | None,
        hitl_state: dict,
        confirmed: bool,
    ) -> None:
        """Handle user clicking Confirm/Cancel on a HITL button.

        Args:
            channel_id: Slack channel where the interaction happened
            message_ts: Timestamp of the message with buttons (to update it)
            thread_ts: Thread timestamp for reply
            hitl_state: Deserialized button value with thread_id, conversation_id, etc.
            confirmed: True if user clicked Confirm, False for Cancel
        """
        action = hitl_state.get("action", "unknown")

        try:
            # Update the button message to show resolved state
            resolved = self._formatter.format_hitl_resolved(action, confirmed)
            await self._client.chat_update(
                channel=channel_id,
                ts=message_ts,
                **resolved,
            )

            # Generate auth token from stored state
            auth_token = self.generate_auth_token(
                hitl_state["user_email"], hitl_state["dept_id"]
            )

            # Show typing indicator
            await self._show_typing(channel_id, thread_ts)

            # Call /chat/resume endpoint
            answer, contexts, new_hitl = await self.call_resume_endpoint(
                thread_id=hitl_state["thread_id"],
                confirmed=confirmed,
                conversation_id=hitl_state.get("conversation_id"),
                auth_token=auth_token,
            )

            # Remove typing indicator
            await self._remove_typing(channel_id, thread_ts)

            # Handle chained HITL (another confirmation needed)
            if new_hitl:
                await self._send_hitl_confirmation(
                    channel_id=channel_id,
                    thread_ts=thread_ts,
                    answer=answer,
                    hitl_data=new_hitl,
                    conv_id=hitl_state.get("conversation_id", ""),
                    user_email=hitl_state["user_email"],
                    dept_id=hitl_state["dept_id"],
                )
                return

            # Extract file links from answer
            file_links = self._extract_file_links(answer)
            if file_links:
                answer = self._remove_file_links(answer)

            # Send final response
            response = BotResponse(text=answer, contexts=contexts)
            await self.send_response(channel_id, response, thread_id=thread_ts)

            # Upload referenced files
            if file_links:
                await self._upload_files_to_slack(
                    file_links, channel_id, thread_ts, auth_token
                )

        except Exception as e:
            logger.exception("Error handling HITL response: %s", e)
            await self._remove_typing(channel_id, thread_ts)
            error_response = BotResponse(text="", error=str(e))
            await self.send_response(channel_id, error_response, thread_id=thread_ts)

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

    async def _send_hitl_confirmation(
        self,
        channel_id: str,
        thread_ts: str | None,
        answer: str,
        hitl_data: dict,
        conv_id: str,
        user_email: str,
        dept_id: str,
    ) -> None:
        """Send HITL confirmation message with Confirm/Cancel buttons."""
        # Serialize state into button value so interactive handler can resume
        button_value = json.dumps(
            {
                "thread_id": hitl_data.get("thread_id", ""),
                "conversation_id": conv_id,
                "channel_id": channel_id,
                "message_ts": thread_ts,
                "user_email": user_email,
                "dept_id": dept_id,
                "action": hitl_data.get("action", "unknown"),
            }
        )

        formatted = self._formatter.format_hitl_confirmation(
            partial_answer=answer,
            hitl_data=hitl_data,
            button_value=button_value,
        )

        await self._client.chat_postMessage(
            channel=channel_id,
            thread_ts=thread_ts,
            **formatted,
        )

        logger.info(
            "[SLACK] HITL confirmation sent for action=%s in channel=%s",
            hitl_data.get("action"),
            channel_id,
        )

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

    async def _fetch_channel_history(
        self, channel_id: str, current_query: str, current_message_ts: str
    ) -> list[dict[str, str]]:
        """Fetch recent channel messages from Slack API for LLM context.

        Uses LLM to determine how many messages are needed based on the query,
        then fetches that many messages from Slack's conversations.history API.
        Messages are formatted as role/content dicts in chronological order.

        Args:
            channel_id: Slack channel ID
            current_query: User's current message text (for LLM intent analysis)
            current_message_ts: Timestamp of current message (excluded from history)

        Returns:
            List of {"role": "user"|"assistant", "content": "..."} dicts
        """
        # Use LLM to determine how many history messages this query needs
        try:
            from openai import OpenAI

            openai_client = OpenAI(api_key=Config.OPENAI_KEY) if Config.OPENAI_KEY else None
        except ImportError:
            openai_client = None

        messages_needed = determine_message_count(
            current_query, openai_client, fallback_limit=Config.SLACK_HISTORY_LIMIT
        )

        if messages_needed == 0:
            logger.info("[SLACK] LLM determined no history needed for this query")
            return []

        # Fetch messages from Slack API
        try:
            result = await self._client.conversations_history(
                channel=channel_id,
                limit=messages_needed,
            )
            messages = result.get("messages", [])
        except Exception as e:
            logger.warning("[SLACK] Failed to fetch channel history: %s", e)
            return []

        # Slack returns newest-first, reverse to chronological order
        messages.reverse()

        history = []
        for msg in messages:
            # Skip the current message (it's sent separately as the query)
            if msg.get("ts") == current_message_ts:
                continue

            text = msg.get("text", "").strip()
            if not text:
                continue

            # Bot messages → assistant role
            if msg.get("bot_id") or msg.get("subtype") == "bot_message":
                history.append({"role": "assistant", "content": text})
            else:
                # User messages → prefix with display name for context
                user_id = msg.get("user", "")
                if user_id:
                    display_name = await self._identity_resolver.get_display_name(user_id)
                    history.append({"role": "user", "content": f"{display_name}: {text}"})
                else:
                    history.append({"role": "user", "content": text})

        logger.info(
            "[SLACK] Fetched %d channel messages (requested %d) for channel %s",
            len(history),
            messages_needed,
            channel_id,
        )
        return history

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
