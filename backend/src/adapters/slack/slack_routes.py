"""
Slack Routes (Webhook Endpoints)
================================

FastAPI routes to receive webhooks from Slack and forward to SlackBotAdapter.

ENDPOINTS:
----------
POST /slack/events      - Receives all Slack events (messages, app_mention, etc.)
POST /slack/interactive - Receives button clicks (HITL confirm/cancel, etc.)

SECURITY:
---------
All requests are verified using Slack's signing secret to prevent spoofing.
See: https://api.slack.com/authentication/verifying-requests-from-slack

BACKGROUND PROCESSING:
----------------------
Slack requires response within 3 seconds. We return 200 OK immediately
and process messages in background tasks.

NOTE: File upload and ingestion are handled via the web frontend, not Slack.
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


async def _process_hitl_action(
    channel_id: str,
    message_ts: str,
    thread_ts: str | None,
    hitl_state: dict,
    confirmed: bool,
) -> None:
    """Background task to process HITL confirm/cancel button clicks."""
    try:
        adapter = _get_adapter()
        await adapter.handle_hitl_response(
            channel_id=channel_id,
            message_ts=message_ts,
            thread_ts=thread_ts,
            hitl_state=hitl_state,
            confirmed=confirmed,
        )
    except Exception as e:
        logger.exception(f"Error processing HITL action: {e}")


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

                background_tasks.add_task(_process_message_event, event)

        # Handle app_mention events (when bot is @mentioned)
        elif event_type == "app_mention":
            user = event.get("user", "unknown")
            text = event.get("text", "")
            channel = event.get("channel", "")

            logger.info(f"[SLACK] Mention from {user} in {channel}: {text[:50]}...")
            background_tasks.add_task(_process_message_event, event)

    # Slack expects 200 OK response within 3 seconds
    return Response(status_code=200)


@router.post("/interactive")
async def slack_interactive(request: Request, background_tasks: BackgroundTasks):
    """
    Handle Slack interactive components (buttons, modals, etc.).

    This endpoint receives payloads when users interact with:
    - HITL Confirm/Cancel buttons (action_id: hitl_confirm, hitl_cancel)
    - Other buttons or modal submissions (future)
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
        actions = payload.get("actions", [])
        channel = payload.get("channel", {})
        channel_id = channel.get("id", "")
        message = payload.get("message", {})
        message_ts = message.get("ts", "")
        # thread_ts is the original message the button message is replying to
        thread_ts = message.get("thread_ts")

        for action in actions:
            action_id = action.get("action_id", "")

            if action_id in ("hitl_confirm", "hitl_cancel"):
                confirmed = action_id == "hitl_confirm"
                value_str = action.get("value", "{}")

                try:
                    hitl_state = json.loads(value_str)
                except (json.JSONDecodeError, TypeError):
                    logger.error(f"[SLACK] Failed to parse HITL button value: {value_str}")
                    continue

                logger.info(
                    "[SLACK] HITL %s by user in channel=%s, action=%s",
                    "confirmed" if confirmed else "cancelled",
                    channel_id,
                    hitl_state.get("action", "unknown"),
                )

                background_tasks.add_task(
                    _process_hitl_action,
                    channel_id=channel_id,
                    message_ts=message_ts,
                    thread_ts=thread_ts,
                    hitl_state=hitl_state,
                    confirmed=confirmed,
                )
            else:
                logger.info(f"[SLACK] Unhandled block action: {action_id}")

    elif interaction_type == "view_submission":
        # Modal form submissions (reserved for future)
        view = payload.get("view", {})
        callback_id = view.get("callback_id")
        logger.info(f"[SLACK] View submission: {callback_id}")

    # Slack expects 200 OK within 3 seconds
    return Response(status_code=200)