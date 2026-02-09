"""Slack Identity Resolver - Resolves Slack user/channel to user_email + dept_id."""

import logging
from slack_sdk.web.async_client import AsyncWebClient

from src.adapters.base_bot_adapter import BotIdentityResolver, BotUser
from src.config.settings import Config

logger = logging.getLogger(__name__)


class SlackIdentityResolver(BotIdentityResolver):
    """Resolves Slack user/channel to user_email + dept_id."""

    def __init__(self, slack_client: AsyncWebClient):
        self._slack_client = slack_client
        self._emails: dict[str, str] = {}
        self._display_names: dict[str, str] = {}

    async def resolve_identity(self, platform_event: dict) -> BotUser:
        user_id = platform_event.get("user")
        channel_id = platform_event.get("channel")
        user_email = await self.get_user_email(user_id)
        workspace_id = platform_event.get("team")
        dept_id = self.get_department(channel_id, workspace_id)
        display_name = self._display_names.get(user_id)
        return BotUser(
            user_email=user_email,
            dept_id=dept_id,
            platform_user_id=user_id,
            display_name=display_name,
        )

    async def _fetch_and_cache_user(self, platform_user_id: str) -> None:
        """Fetch user info from Slack API and cache email + display name."""
        response = await self._slack_client.users_info(user=platform_user_id)
        user_info = response.get("user", {})
        profile = user_info.get("profile", {})

        user_email = profile.get("email")
        if not user_email:
            raise ValueError(
                f"Could not retrieve email for Slack user ID {platform_user_id}. "
                "Ensure the bot has 'users:read.email' scope."
            )
        self._emails[platform_user_id] = user_email

        display_name = (
            profile.get("display_name")
            or profile.get("real_name")
            or user_email.split("@")[0]
        )
        self._display_names[platform_user_id] = display_name

    async def get_user_email(self, platform_user_id: str) -> str:
        if platform_user_id not in self._emails:
            await self._fetch_and_cache_user(platform_user_id)
        return self._emails[platform_user_id]

    async def get_display_name(self, platform_user_id: str) -> str:
        """Get cached display name, fetching from Slack API if needed."""
        if platform_user_id not in self._display_names:
            try:
                await self._fetch_and_cache_user(platform_user_id)
            except Exception:
                return "Unknown"
        return self._display_names.get(platform_user_id, "Unknown")

    def get_department(self, channel_id: str, workspace_id: str | None = None) -> str:
        return Config.SLACK_DEFAULT_DEPT

    def get_conversation_key(self, channel_id: str, user_email: str) -> str:
        return f"{channel_id}:{user_email}"
