"""
SendMessage Command - Process user message and get AI response.

TODO: Implement

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
from chromadb.api.models.Collection import Collection
from openai import OpenAI
from src.utils.sanitizer import sanitize_text
from src.domain.value_objects.conversation_id import ConversationId
from src.domain.value_objects.user_email import UserEmail
from src.domain.value_objects.dept_id import DeptId
from src.domain.entities.conversation import Conversation
from src.domain.entities.message import Message
from src.domain.ports.repositories.conversation_repository import ConversationRepository
from src.domain.ports.repositories.message_repository import MessageRepository
from src.application.common.interfaces import Command, CommandHandler
from src.services.query_supervisor import QuerySupervisor
from src.services.file_manager import FileManager
from src.utils.safety import looks_like_injection
from src.config.settings import Config

logger = logging.getLogger(__name__)


@dataclass
class SendMessageResult:
    answer: str
    contexts: list[dict[str, Any]]
    conversation_id: ConversationId


@dataclass(frozen=True)
class SendMessageCommand(Command[SendMessageResult]):
    conversation_id: ConversationId
    user_email: UserEmail
    dept_id: DeptId
    content: str
    attachments: Optional[list] = None
    filters: Optional[dict] = None  # Additional filters for query processing


class SendMessageHandler(CommandHandler[SendMessageResult]):
    def __init__(
        self,
        conv_repo: ConversationRepository,
        msg_repo: MessageRepository,
        query_supervisor: QuerySupervisor,
        collection: Optional[Collection] = None,
        openai_client: Optional[OpenAI] = None,
    ):
        self.conv_repo = conv_repo
        self.msg_repo = msg_repo
        self.query_supervisor = query_supervisor
        self.collection = collection
        self.openai_client = openai_client

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

        conversation_id = str(command.conversation_id).strip()
        if conversation_id:
            conversation = await self.conv_repo.get_by_id(command.conversation_id)
            if not conversation:
                raise ValueError("Conversation not found.")
            if conversation.user_email.value != user_email:
                raise PermissionError("Access denied to this conversation.")
        if not conversation_id:
            conversation = Conversation.create(
                user_email=command.user_email,
                title=command.content[:20] + "...",
            )
            await self.conv_repo.save(conversation)
            conversation_id = str(conversation.id).strip()

        user_message = Message.create(
            conversation_id=ConversationId(conversation_id),
            role="user",
            content=query,
        )
        await self.msg_repo.save(user_message)

        conversation_history = await self.msg_repo.get_by_conversation(
            ConversationId(conversation_id), limit=Config.REDIS_CACHE_LIMIT
        )
        conversation_history = [
            {"role": msg.role, "content": msg.content} for msg in conversation_history
        ]
        conversation_history = self._get_sanitized(conversation_history)

        # Process attachments and discover available files
        attachment_file_ids = await self._process_attachments(
            command.attachments or [], user_email, conversation_id, dept_id
        )
        available_files = await self._discover_files(
            query, user_email, conversation_id, dept_id=dept_id
        )
        logger.debug(
            f"********************************available_files: {available_files}"
        )

        agent_context = {
            "collection": self.collection,
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
        }
        final_answer, contexts = await self.query_supervisor.process_query(
            query, agent_context
        )

        assistant_message = Message.create(
            conversation_id=ConversationId(conversation_id),
            role="assistant",
            content=final_answer,
        )
        await self.msg_repo.save(assistant_message)

        return SendMessageResult(
            answer=final_answer,
            contexts=contexts,
            conversation_id=ConversationId(conversation_id),
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

        try:
            async with FileManager() as fm:
                for attachment in attachments:
                    try:
                        file_id = await fm.save_chat_attachment(
                            user_email=user_email,
                            filename=attachment.get("filename", "unknown"),
                            content=base64.b64decode(attachment.get("data", "")),
                            mime_type=attachment.get(
                                "mime_type", "application/octet-stream"
                            ),
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
                            f"Saved attachment: {file_id} ({attachment.get('filename')})"
                        )
                    except Exception as e:
                        logger.error(f"Failed to save attachment: {e}")
        except Exception as e:
            logger.error(f"Error processing attachments: {e}")

        return attachment_file_ids

    def _detect_file_intent_with_llm(self, query: str) -> bool:
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
            prompt = """Analyze this user query and determine if they are requesting or expecting:
- Links to source files/documents
- Download links
- File references or attachments
- Original document sources

User query: {query}

CRITICAL! Answer ONLY with: yes or no"""

            response = self.openai_client.chat.completions.create(
                model=Config.OPENAI_SIMPLE_MODEL,
                messages=[{"role": "user", "content": prompt.format(query=query)}],
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
    ) -> list[dict]:
        """Fetch user's files if query implies need for file links/references.

        Uses LLM-based intent detection to support multilingual queries.
        Includes shared files in same department when dept_id is provided.
        """
        needs_file_discovery = self._detect_file_intent_with_llm(query)

        if not needs_file_discovery:
            logger.debug(
                f"Skipping file discovery for user {user_email} "
                f"(LLM determined no file-related intent in query)"
            )
            return []

        available_files = []
        try:
            async with FileManager() as fm:
                # Get ALL indexed RAG files
                # Include dept_id to also get shared files in the department
                indexed_files = await fm.list_files(
                    user_email=user_email,
                    category="uploaded",
                    dept_id=dept_id,
                    limit=Config.FILE_DISCOVERY_INDEXED_LIMIT,
                )

                # Get conversation files (chat attachments, recent downloads, created docs)
                conv_files = await fm.list_files(
                    user_email=user_email,
                    conversation_id=conversation_id,
                    dept_id=dept_id,
                    limit=Config.FILE_DISCOVERY_CONVERSATION_LIMIT,
                )
                # Combine: all indexed files + conversation files
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
