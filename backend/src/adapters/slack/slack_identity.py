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

    async def resolve_identity(self, platform_event: dict) -> BotUser:
        user_id = platform_event.get("user")
        channel_id = platform_event.get("channel")
        user_email = await self.get_user_email(user_id)
        workspace_id = platform_event.get("team")
        dept_id = self.get_department(channel_id, workspace_id)
        return BotUser(
            user_email=user_email,
            dept_id=dept_id,
            platform_user_id=user_id,
            display_name=None,
        )

    async def get_user_email(self, platform_user_id: str) -> str:
        if platform_user_id in self._emails:
            return self._emails[platform_user_id]

        response = await self._slack_client.users_info(user=platform_user_id)
        user_info = response.get("user", {})
        user_email = user_info.get("profile", {}).get("email")
        if not user_email:
            raise ValueError(
                f"Could not retrieve email for Slack user ID {platform_user_id}. "
                "Ensure the bot has 'users:read.email' scope."
            )
        self._emails[platform_user_id] = user_email

        return user_email

    def get_department(self, channel_id: str, workspace_id: str | None = None) -> str:
        return Config.SLACK_DEFAULT_DEPT

    def get_conversation_key(self, channel_id: str, user_email: str) -> str:
        return f"slack:{channel_id}:{user_email}"
