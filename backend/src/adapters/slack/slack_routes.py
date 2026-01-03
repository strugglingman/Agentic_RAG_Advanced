"""
Slack Routes (Webhook Endpoints)
================================

FastAPI routes to receive webhooks from Slack and forward to SlackBotAdapter.

ENDPOINTS:
----------
POST /slack/events  - Receives all Slack events (messages, file_shared, etc.)
POST /slack/commands - Receives slash commands (/ingest, etc.)

SECURITY:
---------
All requests are verified using Slack's signing secret to prevent spoofing.
See: https://api.slack.com/authentication/verifying-requests-from-slack

BACKGROUND PROCESSING:
----------------------
Slack requires response within 3 seconds. We return 200 OK immediately
and process messages in background tasks.
"""

import hashlib
import hmac
import json
import logging
import time
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, Response
from redis.asyncio import Redis

from src.adapters.slack.slack_adapter import SlackBotAdapter
from src.config.settings import Config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/slack", tags=["slack"])

# Singleton adapter instance (lazy initialized)
_slack_adapter: Optional[SlackBotAdapter] = None
_redis_client: Optional[Redis] = None


def _get_adapter() -> SlackBotAdapter:
    """Get or create the SlackBotAdapter singleton."""
    global _slack_adapter, _redis_client

    if _slack_adapter is None:
        # Initialize Redis client if enabled and configured
        if Config.REDIS_ENABLED and Config.REDIS_URL:
            try:
                _redis_client = Redis.from_url(Config.REDIS_URL)
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
                _redis_client = None

        _slack_adapter = SlackBotAdapter(redis=_redis_client)
        logger.info("SlackBotAdapter initialized")

    return _slack_adapter


def _verify_slack_signature(
    body: bytes, timestamp: str, signature: str, signing_secret: str
) -> bool:
    """
    Verify that the request came from Slack using signing secret.

    See: https://api.slack.com/authentication/verifying-requests-from-slack
    """
    if not signing_secret:
        logger.warning("SLACK_SIGNING_SECRET not configured, skipping verification")
        return True  # Skip verification if not configured (dev mode)

    # Check timestamp to prevent replay attacks (5 minute window)
    try:
        request_timestamp = int(timestamp)
        current_timestamp = int(time.time())
        if abs(current_timestamp - request_timestamp) > 300:
            logger.warning("Slack request timestamp too old")
            return False
    except (ValueError, TypeError):
        logger.warning("Invalid timestamp in Slack request")
        return False

    # Compute expected signature
    sig_basestring = f"v0:{timestamp}:{body.decode('utf-8')}"
    expected_signature = (
        "v0="
        + hmac.new(
            signing_secret.encode("utf-8"),
            sig_basestring.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
    )

    # Compare signatures (timing-safe)
    return hmac.compare_digest(expected_signature, signature)


async def _process_message_event(event: dict) -> None:
    """Background task to process message events."""
    try:
        adapter = _get_adapter()
        await adapter.handle_message(event)
    except Exception as e:
        logger.exception(f"Error processing Slack message: {e}")


async def _process_ingest_event(event: dict) -> None:
    """Background task to process ingest command/file upload."""
    try:
        adapter = _get_adapter()
        await adapter.handle_ingest_command(event)
    except Exception as e:
        logger.exception(f"Error processing Slack ingest: {e}")


def _is_ingest_request(text: str) -> bool:
    """Check if message text indicates an ingest request."""
    text_lower = text.lower()
    ingest_keywords = ["ingest", "index", "upload to rag", "add to knowledge"]
    return any(keyword in text_lower for keyword in ingest_keywords)


@router.post("/events")
async def slack_events(request: Request, background_tasks: BackgroundTasks):
    """
    Handle all Slack events.

    SLACK EVENT TYPES:
    ------------------
    1. url_verification - Slack verifying your endpoint (one-time setup)
    2. event_callback - Actual events (messages, files, etc.)

    IMPORTANT: Must respond within 3 seconds. Processing happens in background.
    """
    # Check if Slack integration is enabled
    if not Config.SLACK_ENABLED:
        logger.debug("Slack integration disabled, ignoring event")
        return Response(status_code=200)

    # Read raw body for signature verification
    body = await request.body()

    # Verify request signature
    timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
    signature = request.headers.get("X-Slack-Signature", "")

    if not _verify_slack_signature(
        body, timestamp, signature, Config.SLACK_SIGNING_SECRET
    ):
        logger.warning("Invalid Slack signature, rejecting request")
        raise HTTPException(status_code=401, detail="Invalid signature")

    # Parse JSON body
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # Log raw event for debugging
    logger.debug(
        f"[SLACK] Raw event: {json.dumps(data, indent=2, default=str, ensure_ascii=False)}"
    )

    # Handle Slack URL verification (required for initial setup)
    if data.get("type") == "url_verification":
        challenge = data.get("challenge")
        logger.info("[SLACK] URL verification challenge received")
        return {"challenge": challenge}

    # Handle actual events
    if data.get("type") == "event_callback":
        event = data.get("event", {})
        event_type = event.get("type")

        logger.debug(f"[SLACK] Event type: {event_type}")

        # Handle message events
        if event_type == "message":
            # Skip bot messages, message_changed, message_deleted, etc.
            subtype = event.get("subtype")
            if subtype is None:  # Regular user message
                user = event.get("user", "unknown")
                text = event.get("text", "")
                channel = event.get("channel", "")
                files = event.get("files", [])

                logger.info(f"[SLACK] Message from {user} in {channel}: {text[:50]}...")
                if files:
                    logger.info(
                        f"[SLACK] Attached files: {[f.get('name') for f in files]}"
                    )

                # Check if this is an ingest request (has files + ingest keyword)
                if files and _is_ingest_request(text):
                    logger.info(
                        f"[SLACK] Ingest request detected with {len(files)} files"
                    )
                    background_tasks.add_task(_process_ingest_event, event)
                else:
                    # Regular chat message
                    background_tasks.add_task(_process_message_event, event)

        # Handle app_mention events (when bot is @mentioned)
        elif event_type == "app_mention":
            user = event.get("user", "unknown")
            text = event.get("text", "")
            channel = event.get("channel", "")
            files = event.get("files", [])

            logger.info(f"[SLACK] Mention from {user} in {channel}: {text[:50]}...")

            # Check if this is an ingest request
            if files and _is_ingest_request(text):
                logger.info(
                    f"[SLACK] Ingest request via mention with {len(files)} files"
                )
                background_tasks.add_task(_process_ingest_event, event)
            else:
                background_tasks.add_task(_process_message_event, event)

        # Handle file_shared events (optional - for auto-ingest)
        elif event_type == "file_shared":
            file_id = event.get("file_id")
            logger.info(f"[SLACK] File shared: {file_id}")
            # Could auto-ingest files here if desired

    # Slack expects 200 OK response within 3 seconds
    return Response(status_code=200)


@router.post("/commands")
async def slack_commands(request: Request):
    """
    Handle Slack slash commands (e.g., /ingest).

    COMMAND FORMAT:
    ---------------
    Slack sends form data (not JSON):
    - command: "/ingest"
    - text: "optional arguments"
    - user_id: "U0123456789"
    - channel_id: "C0123456789"
    - trigger_id: for modals/dialogs

    SUPPORTED COMMANDS:
    -------------------
    /ingest - Ingest attached files into RAG system

    NOTE: Slash commands cannot include file attachments directly.
    Users must upload files in a message and mention the bot with "ingest".
    """
    # Check if Slack integration is enabled
    if not Config.SLACK_ENABLED:
        return {
            "response_type": "ephemeral",
            "text": "Slack integration is disabled.",
        }

    # Read raw body for signature verification
    body = await request.body()

    # Verify request signature
    timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
    signature = request.headers.get("X-Slack-Signature", "")

    if not _verify_slack_signature(
        body, timestamp, signature, Config.SLACK_SIGNING_SECRET
    ):
        logger.warning("Invalid Slack signature for command, rejecting")
        raise HTTPException(status_code=401, detail="Invalid signature")

    # Parse form data
    form_data = await request.form()

    command = form_data.get("command", "")
    text = form_data.get("text", "")
    user_id = form_data.get("user_id", "")
    channel_id = form_data.get("channel_id", "")

    logger.info(f"[SLACK] Command {command} from {user_id} in {channel_id}: {text}")

    # Handle /ingest command
    if command == "/ingest":
        # Build event-like structure for the adapter
        # Note: Files need to be attached in a follow-up message
        # Slash commands don't include file attachments directly

        return {
            "response_type": "ephemeral",
            "text": (
                ":information_source: *To ingest files:*\n"
                "1. Upload your file(s) to this channel\n"
                "2. In the same message, mention me and say 'ingest'\n"
                "3. Or reply to a message with files and say 'ingest these'\n\n"
                "_Example: Upload a PDF, then type `@bot ingest this file`_"
            ),
        }

    # Unknown command
    return {
        "response_type": "ephemeral",
        "text": f"Unknown command: {command}",
    }


@router.post("/interactive")
async def slack_interactive(request: Request):
    """
    Handle Slack interactive components (buttons, modals, etc.).

    This endpoint receives payloads when users interact with:
    - Buttons in messages
    - Modal submissions
    - Shortcuts

    Currently not used but reserved for future enhancements.
    """
    # Check if Slack integration is enabled
    if not Config.SLACK_ENABLED:
        return Response(status_code=200)

    # Read raw body for signature verification
    body = await request.body()

    # Verify request signature
    timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
    signature = request.headers.get("X-Slack-Signature", "")

    if not _verify_slack_signature(
        body, timestamp, signature, Config.SLACK_SIGNING_SECRET
    ):
        raise HTTPException(status_code=401, detail="Invalid signature")

    # Parse form data (Slack sends payload as form field)
    form_data = await request.form()
    payload_str = form_data.get("payload", "{}")

    try:
        payload = json.loads(payload_str)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid payload")

    logger.debug(f"[SLACK] Interactive payload: {payload.get('type')}")

    # Handle different interaction types
    interaction_type = payload.get("type")

    if interaction_type == "block_actions":
        # Button clicks, select menus, etc.
        actions = payload.get("actions", [])
        for action in actions:
            action_id = action.get("action_id")
            logger.info(f"[SLACK] Block action: {action_id}")

    elif interaction_type == "view_submission":
        # Modal form submissions
        view = payload.get("view", {})
        callback_id = view.get("callback_id")
        logger.info(f"[SLACK] View submission: {callback_id}")

    return Response(status_code=200)
