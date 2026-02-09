"""
SendMessage Command - Process user message and get AI response.

Command data:
- conversation_id: ConversationId
- user_email: UserEmail
- content: str
- attachments: Optional[list]

Handler:
1. Load conversation from repo
2. Verify user owns conversation
3. Save user message
4. Call agent service to get response
5. Save assistant message
6. Return response

Example:
```python
@dataclass
class SendMessageCommand:
    conversation_id: ConversationId
    user_email: UserEmail
    content: str
    attachments: Optional[list] = None

class SendMessageHandler:
    def __init__(
        self,
        conv_repo: ConversationRepository,
        msg_repo: MessageRepository,
        agent_service: AgentService,
    ):
        self.conv_repo = conv_repo
        self.msg_repo = msg_repo
        self.agent_service = agent_service

    async def execute(self, cmd: SendMessageCommand) -> Message:
        # 1. Verify conversation
        conv = await self.conv_repo.get_by_id(cmd.conversation_id)
        if not conv:
            raise EntityNotFoundError("Conversation not found")
        if conv.user_email != cmd.user_email:
            raise AccessDeniedError()

        # 2. Save user message
        user_msg = Message(...)
        await self.msg_repo.save(user_msg)

        # 3. Get AI response
        response = await self.agent_service.process(cmd.content, conv.id)

        # 4. Save assistant message
        assistant_msg = Message(...)
        await self.msg_repo.save(assistant_msg)

        return assistant_msg
```

Maps from: src/routes/chat.py chat_agent() function
"""

import base64
import logging
from dataclasses import dataclass
from typing import Optional, Any
from openai import OpenAI
from src.utils.sanitizer import sanitize_text
from src.utils.url_formatter import format_urls_as_markdown
from src.domain.value_objects.conversation_id import ConversationId
from src.domain.value_objects.user_email import UserEmail
from src.domain.value_objects.dept_id import DeptId
from src.domain.entities.conversation import Conversation
from src.domain.entities.message import Message
from src.domain.ports.repositories.conversation_repository import ConversationRepository
from src.domain.ports.repositories.message_repository import MessageRepository
from src.application.common.interfaces import Command, CommandHandler
from src.application.services import FileService
from src.services.query_supervisor import QuerySupervisor
from src.services.vector_db import VectorDB
from src.services.llm_client import chat_completion
from src.services.agent_state import AgentSessionStateStore
from src.utils.safety import looks_like_injection
from src.config.settings import Config

logger = logging.getLogger(__name__)


@dataclass
class HITLInterruptInfo:
    """Information about a pending HITL action requiring confirmation."""

    action: str  # "send_email", etc.
    thread_id: str  # Thread ID for resumption
    details: dict[str, Any]  # Action-specific details for UI
    previous_steps: list[dict[str, Any]]  # Completed step results


@dataclass
class SendMessageResult:
    answer: str
    contexts: list[dict[str, Any]]
    conversation_id: ConversationId
    hitl_interrupt: Optional[HITLInterruptInfo] = None

    @property
    def requires_confirmation(self) -> bool:
        """Check if this result requires human confirmation."""
        return self.hitl_interrupt is not None


@dataclass(frozen=True)
class SendMessageCommand(Command[SendMessageResult]):
    conversation_id: ConversationId
    user_email: UserEmail
    dept_id: DeptId
    content: str
    attachments: Optional[list] = None
    filters: Optional[dict] = None  # Additional filters for query processing
    source_channel_id: Optional[str] = None  # For Slack/Teams: "slack:C0123ABC"
    conversation_history: Optional[list[dict[str, str]]] = None  # Pre-fetched history from bot adapters


class SendMessageHandler(CommandHandler[SendMessageResult]):
    def __init__(
        self,
        conv_repo: ConversationRepository,
        msg_repo: MessageRepository,
        query_supervisor: QuerySupervisor,
        file_service: FileService,
        vector_db: Optional[VectorDB] = None,
        openai_client: Optional[OpenAI] = None,
        agent_state_store: Optional[AgentSessionStateStore] = None,
    ):
        self.conv_repo = conv_repo
        self.msg_repo = msg_repo
        self.query_supervisor = query_supervisor
        self.file_service = file_service
        self.vector_db = vector_db
        self.openai_client = openai_client
        self.agent_state_store = agent_state_store

    async def execute(self, command: SendMessageCommand) -> SendMessageResult:
        query = command.content.strip()
        if not query:
            raise ValueError("Message content cannot be empty.")
        dept_id = str(command.dept_id).strip()
        if not dept_id:
            raise ValueError("Department ID cannot be empty.")
        user_email = str(command.user_email).strip()
        if not user_email:
            raise ValueError("User email cannot be empty.")
        injection_result, injection_error = looks_like_injection(command.content)
        if injection_result:
            raise ValueError(f"Potential prompt injection detected: {injection_error}")

        # Unified conversation lookup: source_channel_id (Slack/Teams) > conversation_id (web)
        conversation = await self.conv_repo.find_conversation(
            user_email=command.user_email,
            conversation_id=command.conversation_id if command.conversation_id.value else None,
            source_channel_id=command.source_channel_id,
        )

        # Edge case: Web UI provided conversation_id but not found (deleted/invalid)
        if not conversation and command.conversation_id.value and not command.source_channel_id:
            raise ValueError("Conversation not found.")

        # Verify ownership for any found conversation
        # (get_by_source filters by user, but get_by_id fallback doesn't - always check)
        if conversation and conversation.user_email.value != user_email:
            raise PermissionError("Access denied to this conversation.")

        # Create new conversation if not found (new chat from web, or Slack channel)
        if not conversation:
            conversation = Conversation.create(
                user_email=command.user_email,
                title=command.content[:20] + "...",
                source_channel_id=command.source_channel_id,
            )
            await self.conv_repo.save(conversation)

        conversation_id = str(conversation.id).strip()

        user_message = Message.create(
            conversation_id=ConversationId(conversation_id),
            role="user",
            content=query,
        )
        await self.msg_repo.save(user_message)

        # Conversation history: use pre-fetched (Slack/Teams) or fetch from DB (Web UI)
        if command.conversation_history is not None:
            # Slack/Teams: channel history already fetched by adapter
            conversation_history = self._get_sanitized(command.conversation_history)
        else:
            # Web UI: fetch from DB via smart history (LLM determines count)
            history_messages = await self.msg_repo.get_smart_history(
                query=query,
                conversation_id=ConversationId(conversation_id),
                context={"openai_client": self.openai_client}
            )
            conversation_history = [
                {"role": msg.role, "content": msg.content} for msg in history_messages
            ]
            conversation_history = self._get_sanitized(conversation_history)

        # Process attachments and discover available files
        attachment_file_ids = await self._process_attachments(
            command.attachments or [], user_email, conversation_id, dept_id
        )
        available_files = await self._discover_files(
            query,
            user_email,
            conversation_id,
            dept_id=dept_id,
            conversation_history=conversation_history,
        )
        logger.debug(
            f"=============[SendMessage] available_files: {[f['id'] for f in available_files]}, attachments: {[a['file_id'] for a in attachment_file_ids]}"
        )

        agent_context = {
            "vector_db": self.vector_db,
            "dept_id": dept_id,
            "user_id": user_email,
            "conversation_id": conversation_id,
            "conversation_history": conversation_history,
            "request_data": command.filters,
            "use_hybrid": Config.USE_HYBRID,
            "use_reranker": Config.USE_RERANKER,
            "openai_client": self.openai_client,
            "model": Config.OPENAI_MODEL,
            "temperature": Config.OPENAI_TEMPERATURE,
            "available_files": available_files,
            "attachment_file_ids": attachment_file_ids,
            "file_service": self.file_service,
            "_state_store": self.agent_state_store,  # Redis persistence for agent state
        }
        query_result = await self.query_supervisor.process_query(query, agent_context)

        # Check if HITL interrupt occurred
        hitl_interrupt_info = None
        if query_result.hitl_interrupt:
            hitl = query_result.hitl_interrupt
            hitl_interrupt_info = HITLInterruptInfo(
                action=hitl.action,
                thread_id=hitl.thread_id,
                details=hitl.details,
                previous_steps=hitl.previous_steps,
            )
            logger.info(
                f"[HITL] Workflow interrupted for action: {hitl.action}, "
                f"thread_id: {hitl.thread_id}"
            )

        # Ensure all URLs in the answer are formatted as clickable markdown links
        final_answer = format_urls_as_markdown(query_result.answer)

        assistant_message = Message.create(
            conversation_id=ConversationId(conversation_id),
            role="assistant",
            content=final_answer,
        )
        await self.msg_repo.save(assistant_message)

        return SendMessageResult(
            answer=final_answer,
            contexts=query_result.contexts,
            conversation_id=ConversationId(conversation_id),
            hitl_interrupt=hitl_interrupt_info,
        )

    async def _process_attachments(
        self,
        attachments: list,
        user_email: str,
        conversation_id: str,
        dept_id: str,
    ) -> list[dict]:
        """Save attachments to disk + FileRegistry and return file IDs."""
        attachment_file_ids = []
        if not attachments:
            return attachment_file_ids

        for attachment in attachments:
            try:
                file_id = await self.file_service.save_chat_attachment(
                    user_email=user_email,
                    filename=attachment.get("filename", "unknown"),
                    content=base64.b64decode(attachment.get("data", "")),
                    mime_type=attachment.get("mime_type", "application/octet-stream"),
                    conversation_id=conversation_id,
                    dept_id=dept_id,
                )
                attachment_file_ids.append(
                    {
                        "file_id": file_id,
                        "filename": attachment.get("filename", "unknown"),
                        "mime_type": attachment.get(
                            "mime_type", "application/octet-stream"
                        ),
                    }
                )
                logger.info(
                    f"[SendMessage] Saved attachment: {file_id} ({attachment.get('filename')})"
                )
            except Exception as e:
                logger.error(f"[SendMessage] Failed to save attachment: {e}")

        return attachment_file_ids

    def _detect_file_intent_with_llm(
        self, query: str, conversation_history: list[dict] = None
    ) -> bool:
        """Use LLM to detect if user wants file links/references (multilingual support).

        This replaces keyword-based detection to support all languages.
        Uses a fast, cheap model for classification.
        """
        if not self.openai_client:
            logger.warning(
                "OpenAI client not available for intent detection, falling back to keyword matching"
            )
            return self._detect_file_intent_with_keywords(query)

        try:
            # Include recent conversation context for short queries like "confirm", "yes"
            context_section = ""
            if conversation_history and len(query.strip()) < 30:
                recent = conversation_history[-4:]  # Last 2 exchanges
                context_lines = [f"{m['role']}: {m['content'][:100]}" for m in recent]
                context_section = (
                    f"\nRecent conversation:\n" + "\n".join(context_lines) + "\n"
                )

            prompt = """Analyze this user query and determine if they are requesting or expecting:
- Links to source files/documents
- Download links
- File references or attachments
- Original document sources
- Confirming a pending action that involves files/emails (e.g., "confirm", "yes", "proceed", "send it")
{context}
User query: {query}

CRITICAL! Answer ONLY with: yes or no"""

            response = chat_completion(
                client=self.openai_client,
                model=Config.OPENAI_SIMPLE_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt.format(query=query, context=context_section),
                    }
                ],
                max_tokens=5,
                temperature=0,
            )
            answer = response.choices[0].message.content.strip().lower()
            needs_files = answer == "yes"
            logger.info(f"LLM intent detection for file discovery: {answer}")
            return needs_files
        except Exception as e:
            logger.warning(
                f"LLM intent detection failed: {e}, falling back to keyword matching"
            )
            return self._detect_file_intent_with_keywords(query)

    def _detect_file_intent_with_keywords(self, query: str) -> bool:
        """Fallback keyword-based detection for file-related queries."""
        file_keywords = {
            "file",
            "document",
            "attach",
            "email",
            "send",
            "download",
            "upload",
            "pdf",
            "doc",
            "docx",
            "report",
            "policy",
            "invoice",
            "spreadsheet",
            "xls",
            "xlsx",
            "csv",
            "presentation",
            "ppt",
            "image",
            "photo",
            "picture",
            "link",
            "source",
            "original",
        }
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in file_keywords)

    async def _discover_files(
        self,
        query: str,
        user_email: str,
        conversation_id: str,
        dept_id: str = None,
        conversation_history: list[dict] = None,
    ) -> list[dict]:
        """Fetch user's files if query implies need for file links/references.

        Uses LLM-based intent detection to support multilingual queries.
        Includes shared files in same department when dept_id is provided.
        """
        needs_file_discovery = self._detect_file_intent_with_llm(
            query, conversation_history
        )

        if not needs_file_discovery:
            logger.debug(
                f"Skipping file discovery for user {user_email} "
                f"(LLM determined no file-related intent in query)"
            )
            return []

        available_files = []
        try:
            # Get ALL indexed RAG files
            # Include dept_id to also get shared files in the department
            indexed_files = await self.file_service.list_files(
                user_email=user_email,
                category="uploaded",
                dept_id=dept_id,
                limit=Config.FILE_DISCOVERY_INDEXED_LIMIT,
            )

            # Get conversation-specific files (chat attachments, downloads, created docs)
            # These are scoped to the current conversation only (no dept_id)
            # Fetch each category separately to avoid getting uploaded files again
            conv_files = []
            for category in ["downloaded", "created", "chat"]:
                category_files = await self.file_service.list_files(
                    user_email=user_email,
                    conversation_id=conversation_id,
                    category=category,
                    limit=Config.FILE_DISCOVERY_CONVERSATION_LIMIT,
                )
                conv_files.extend(category_files)

            # Combine: all indexed files + conversation files (dedupe by id)
            all_files = {f["id"]: f for f in indexed_files}
            for f in conv_files:
                all_files[f["id"]] = f
            available_files = list(all_files.values())
        except Exception as e:
            logger.error(f"Failed to fetch user files: {e}")

        return available_files

    def _get_sanitized(self, conversation_history: list[dict]) -> list[dict]:
        """Sanitize conversation history for LLM context."""
        sanitized_history = []
        for h in conversation_history:
            sanitized_msg = {
                "role": h["role"],
                "content": sanitize_text(h["content"], max_length=5000),
            }
            sanitized_history.append(sanitized_msg)
        return sanitized_history
