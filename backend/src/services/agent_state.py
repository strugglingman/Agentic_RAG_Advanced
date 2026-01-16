"""
Agent Session State - Enterprise-standard state management for AgentService.

This module provides:
1. FileReference - Validated file reference (prevents ID corruption)
2. AgentSessionState - Per-conversation state with active file tracking
3. AgentSessionStateStore - Redis persistence for state across messages

Enterprise Pattern: Explicit State Injection
- Producer tools (search_documents, download_file, create_documents) → update state
- Consumer tools (send_email) → read from injected state, not LLM recall
- State is injected at prompt TOP in structured XML format
- State is persisted to Redis between user messages (same conversation)

Reference: JetBrains Koog, Anthropic MCP, LangGraph best practices 2025-2026
"""

import logging
from typing import Optional, TYPE_CHECKING
from pydantic import BaseModel, Field

from src.config.settings import Config

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class FileReference(BaseModel):
    """
    Validated file reference - prevents LLM ID corruption.

    The LLM tends to corrupt long alphanumeric IDs when recalling from
    chat history (e.g., 'cmkcwdlx' → 'cmkcwdlu'). By storing these in
    explicit state and injecting them, we avoid this issue.
    """

    file_id: str = Field(
        ..., min_length=10, description="Unique file identifier from FileRegistry"
    )
    filename: str = Field(..., description="Human-readable filename")
    source: Optional[str] = Field(None, description="Source path or URL")

    class Config:
        extra = "forbid"  # Reject unknown fields - strict validation


class AgentSessionState(BaseModel):
    """
    Per-conversation agent state.

    This state is:
    - Updated after producer tool calls (search_documents, download_file, create_documents)
    - Injected into system prompt as structured XML
    - Used by consumer tools (send_email) instead of LLM recall from history

    Key principle: OVERWRITE, not accumulate
    - New search_documents call → replaces active_file_ids
    - This keeps context focused and prevents stale references
    """

    conversation_id: str = Field(..., description="Conversation this state belongs to")
    active_file_ids: list[FileReference] = Field(
        default_factory=list, description="Files from most recent producer tool call"
    )
    last_producer_tool: Optional[str] = Field(
        None, description="Name of the last tool that produced file references"
    )

    class Config:
        extra = "forbid"

    def update_from_search(self, contexts: list[dict]) -> None:
        """
        Update state from search_documents results.

        APPENDS to active_file_ids (clearing is handled by _update_state_from_tool).
        """
        self.last_producer_tool = "search_documents"

        # Get existing file_ids to avoid duplicates
        seen_ids = {f.file_id for f in self.active_file_ids}
        for ctx in contexts:
            file_id = ctx.get("file_id")
            if file_id and file_id not in seen_ids:
                seen_ids.add(file_id)
                self.active_file_ids.append(
                    FileReference(
                        file_id=file_id,
                        filename=ctx.get("source", "unknown"),
                        source=ctx.get("source"),
                    )
                )

    def update_from_download(
        self, file_id: str, filename: str, source_url: Optional[str] = None
    ) -> None:
        """
        Update state from download_file result.

        APPENDS to active_file_ids (downloads are additive within same turn).
        """
        self.last_producer_tool = "download_file"

        # Check if already exists
        if not any(f.file_id == file_id for f in self.active_file_ids):
            self.active_file_ids.append(
                FileReference(file_id=file_id, filename=filename, source=source_url)
            )

    def update_from_create(self, file_id: str, filename: str) -> None:
        """
        Update state from create_documents result.

        APPENDS to active_file_ids (creates are additive within same turn).
        """
        self.last_producer_tool = "create_documents"

        # Check if already exists
        if not any(f.file_id == file_id for f in self.active_file_ids):
            self.active_file_ids.append(
                FileReference(file_id=file_id, filename=filename, source=None)
            )

    def clear_for_new_search(self) -> None:
        """
        Clear state before a new search operation.

        Called at the START of search_documents to ensure clean slate.
        """
        self.active_file_ids = []
        self.last_producer_tool = None

    def to_xml_injection(self) -> str:
        """
        Format state as XML for prompt injection.

        This is injected at the TOP of the system prompt, not buried in history.
        XML structure is more "potent" than plain text instructions (JetBrains Koog).

        Returns empty string if no active files.
        """
        if not self.active_file_ids:
            return ""

        files_json = [
            {"file_id": f.file_id, "filename": f.filename} for f in self.active_file_ids
        ]

        import json

        return f"""<active_files>
{json.dumps(files_json, indent=2)}
</active_files>

Note: These are the files available for email attachments. The system will automatically
use these file_ids when you call send_email with attachments.
"""


class AgentSessionStateStore:
    """
    Redis-backed persistence for AgentSessionState.

    Stores state per conversation_id with TTL. This allows the agent to
    remember active file IDs across multiple user messages in the same
    conversation, preventing LLM file ID corruption.

    Key pattern: agent:state:{conversation_id}
    TTL: AGENT_STATE_TTL (default 1 hour)
    """

    KEY_PREFIX = "agent:state:"

    def __init__(self, redis: "Redis"):
        self._redis = redis
        self._ttl = Config.AGENT_STATE_TTL

    async def load(self, conversation_id: str) -> AgentSessionState:
        """
        Load state from Redis or create new if not exists.

        Args:
            conversation_id: Conversation identifier

        Returns:
            AgentSessionState - either loaded from Redis or fresh instance
        """
        key = f"{self.KEY_PREFIX}{conversation_id}"
        data = await self._redis.get(key)

        if data:
            try:
                state = AgentSessionState.model_validate_json(data)
                logger.debug(
                    f"[AgentState] Loaded state for {conversation_id}: "
                    f"{len(state.active_file_ids)} active files"
                )
                return state
            except Exception as e:
                logger.warning(
                    f"[AgentState] Failed to parse state for {conversation_id}: {e}"
                )

        # Return fresh state if not found or parse error
        logger.debug(f"[AgentState] Creating fresh state for {conversation_id}")
        return AgentSessionState(conversation_id=conversation_id)

    async def save(self, state: AgentSessionState) -> None:
        """
        Save state to Redis with TTL.

        Args:
            state: AgentSessionState to persist
        """
        key = f"{self.KEY_PREFIX}{state.conversation_id}"
        data = state.model_dump_json()

        await self._redis.setex(key, self._ttl, data)
        logger.debug(
            f"[AgentState] Saved state for {state.conversation_id}: "
            f"{len(state.active_file_ids)} active files, TTL={self._ttl}s"
        )

    async def delete(self, conversation_id: str) -> None:
        """
        Delete state from Redis.

        Called when conversation is deleted or state should be cleared.

        Args:
            conversation_id: Conversation identifier
        """
        key = f"{self.KEY_PREFIX}{conversation_id}"
        await self._redis.delete(key)
        logger.debug(f"[AgentState] Deleted state for {conversation_id}")
