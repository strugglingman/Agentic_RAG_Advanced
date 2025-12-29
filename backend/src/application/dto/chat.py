"""Chat DTOs for API request/response."""

from datetime import datetime
from pydantic import BaseModel
from typing import Optional


class MessageDTO(BaseModel):
    """DTO for message data returned to frontend."""

    id: str
    conversation_id: str
    role: str
    content: str
    created_at: datetime
    tokens_used: Optional[int] = None
    latency_ms: Optional[int] = None
