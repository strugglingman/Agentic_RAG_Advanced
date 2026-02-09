"""Base Bot Adapter - Abstract base classes for all bot adapters (Slack, Teams, etc.)."""

from datetime import datetime, timezone, timedelta
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import jwt
import httpx
from src.config.settings import Config

logger = logging.getLogger(__name__)


@dataclass
class BotUser:
    """Resolved user identity from platform."""

    user_email: str
    dept_id: str
    platform_user_id: str
    display_name: str | None = None


@dataclass
class BotMessage:
    """Normalized message from any platform."""

    text: str
    user: BotUser
    channel_id: str
    thread_id: str | None = None
    files: list[dict] | None = None
    raw_event: dict | None = None


@dataclass
class BotResponse:
    """Response to send back to platform."""

    text: str
    contexts: list[dict] | None = None
    files_to_upload: list[str] | None = None
    error: str | None = None


class BotIdentityResolver(ABC):
    """Abstract base for resolving platform identities to user_email + dept_id."""

    @abstractmethod
    async def resolve_identity(self, platform_event: dict) -> BotUser:
        """Resolve platform event to BotUser with user_email and dept_id."""
        ...

    @abstractmethod
    async def get_user_email(self, platform_user_id: str) -> str:
        """Get user's email from platform API."""
        ...

    @abstractmethod
    def get_department(self, channel_id: str, workspace_id: str | None = None) -> str:
        """Map channel/workspace to dept_id."""
        ...


class BaseBotAdapter(ABC):
    """Abstract base for all bot adapters."""

    def __init__(self):
        """Initialize the adapter. Subclasses should set backend_url and identity_resolver."""
        self.backend_url: str = Config.BOT_BACKEND_URL
        self.identity_resolver: BotIdentityResolver | None = None

    @abstractmethod
    async def handle_message(self, event: dict) -> None:
        """Handle incoming message from platform."""
        ...

    @abstractmethod
    async def send_response(
        self, channel_id: str, response: BotResponse, thread_id: str | None = None
    ) -> None:
        """Send formatted response back to platform."""
        ...

    def generate_auth_token(self, user_email: str, dept_id: str) -> str:
        """Generate JWT token for backend API calls."""
        payload = {
            "email": user_email,
            "dept": dept_id,
            "exp": datetime.now(timezone.utc) + timedelta(hours=1),
            "iat": datetime.now(timezone.utc),
            "iss": Config.SERVICE_AUTH_ISSUER,
            "aud": Config.SERVICE_AUTH_AUDIENCE,
        }
        return jwt.encode(payload, Config.SERVICE_AUTH_SECRET, algorithm="HS256")

    async def call_chat_endpoint(
        self,
        message: str,
        conversation_id: str | None,
        attachments: list[dict] | None,
        auth_token: str,
        source_channel_id: str | None = None,
        conversation_history: list[dict] | None = None,
    ) -> tuple[str, str, list[dict], dict | None]:
        """
        Call the /chat endpoint to get AI response.

        Args:
            message: User's message text
            conversation_id: Existing conversation UUID or None for new
            attachments: List of base64-encoded attachments in format:
                [{"filename": "...", "mime_type": "...", "data": "base64..."}]
            auth_token: JWT auth token
            source_channel_id: External channel identifier (e.g. "slack:C0123ABC")
            conversation_history: Pre-fetched conversation history from external platform

        Returns:
            tuple: (conversation_id, answer, contexts, hitl_data_or_none)

        Raises:
            Exception: If HTTP request fails or returns error status
        """
        payload = {
            "messages": [{"role": "user", "content": message}],
            "conversation_id": conversation_id,
            "attachments": attachments or [],
        }
        if source_channel_id:
            payload["source_channel_id"] = source_channel_id
        if conversation_history:
            payload["conversation_history"] = conversation_history

        async with httpx.AsyncClient(timeout=Config.BACKEND_API_TIMEOUT) as client:
            response = await client.post(
                f"{self.backend_url}/chat",
                json=payload,
                headers={"Authorization": f"Bearer {auth_token}"},
            )

            if response.status_code >= 400:
                error_detail = response.text
                try:
                    error_data = response.json()
                    error_detail = error_data.get("detail", error_detail)
                except (ValueError, KeyError):
                    pass
                raise httpx.HTTPStatusError(
                    f"Chat API error ({response.status_code}): {error_detail}",
                    request=response.request,
                    response=response,
                )

            data = response.json()
            return (
                data.get("conversation_id", ""),
                data.get("message", "No message"),
                data.get("contexts", []),
                data.get("hitl"),
            )

    async def call_resume_endpoint(
        self,
        thread_id: str,
        confirmed: bool,
        conversation_id: str | None,
        auth_token: str,
    ) -> tuple[str, list[dict], dict | None]:
        """
        Call the /chat/resume endpoint to continue an interrupted workflow.

        Returns:
            tuple: (answer, contexts, hitl_data_or_none)
        """
        async with httpx.AsyncClient(timeout=Config.BACKEND_API_TIMEOUT) as client:
            response = await client.post(
                f"{self.backend_url}/chat/resume",
                json={
                    "thread_id": thread_id,
                    "confirmed": confirmed,
                    "conversation_id": conversation_id,
                },
                headers={"Authorization": f"Bearer {auth_token}"},
            )

            if response.status_code >= 400:
                error_detail = response.text
                try:
                    error_data = response.json()
                    error_detail = error_data.get("detail", error_detail)
                except (ValueError, KeyError):
                    pass
                raise httpx.HTTPStatusError(
                    f"Resume API error ({response.status_code}): {error_detail}",
                    request=response.request,
                    response=response,
                )

            # Resume returns streaming text; read full body and parse
            raw = response.text
            answer = raw
            contexts = []
            hitl = None

            # Parse __HITL__ marker if present
            if "__HITL__:" in raw:
                hitl_idx = raw.index("__HITL__:")
                answer = raw[:hitl_idx].strip()
                hitl_part = raw[hitl_idx + len("__HITL__:"):]
                # Separate from __CONTEXT__ if present
                if "__CONTEXT__:" in hitl_part:
                    ctx_idx = hitl_part.index("__CONTEXT__:")
                    hitl_json = hitl_part[:ctx_idx].strip()
                    ctx_json = hitl_part[ctx_idx + len("__CONTEXT__:"):].strip()
                else:
                    hitl_json = hitl_part.strip()
                    ctx_json = ""
                try:
                    import json
                    hitl = json.loads(hitl_json)
                except (ValueError, KeyError):
                    pass
                if ctx_json:
                    try:
                        import json
                        contexts = json.loads(ctx_json)
                    except (ValueError, KeyError):
                        pass
            elif "__CONTEXT__:" in raw:
                ctx_idx = raw.index("__CONTEXT__:")
                answer = raw[:ctx_idx].strip()
                ctx_json = raw[ctx_idx + len("__CONTEXT__:"):].strip()
                try:
                    import json
                    contexts = json.loads(ctx_json)
                except (ValueError, KeyError):
                    pass

            return answer, contexts, hitl

    async def download_file_from_backend(
        self, file_id: str, auth_token: str
    ) -> tuple[bytes, str, str]:
        """
        Download file from the /files/{file_id} endpoint.

        Returns:
            tuple: (content_bytes, filename, content_type) - empty bytes on failure
        """
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.get(
                f"{self.backend_url}/files/{file_id}",
                headers={"Authorization": f"Bearer {auth_token}"},
            )

            if response.status_code >= 400:
                logger.error(f"File download failed ({response.status_code})")
                return b"", "", ""

            # Get filename from Content-Disposition header
            content_disp = response.headers.get("content-disposition", "")
            filename = "file"
            if "filename=" in content_disp:
                # Parse filename="xxx.pdf" or filename*=UTF-8''xxx.pdf
                import re

                match = re.search(r'filename[*]?=["\']?([^"\';\n]+)', content_disp)
                if match:
                    filename = match.group(1)

            content_type = response.headers.get(
                "content-type", "application/octet-stream"
            )
            return response.content, filename, content_type
