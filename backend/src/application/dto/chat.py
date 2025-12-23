"""Chat DTOs for API request/response."""

from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    attachments: Optional[list[dict]] = None

class ChatResponse(BaseModel):
    message: str
    conversation_id: str
    tokens_used: Optional[int] = None
    latency_ms: Optional[int] = None