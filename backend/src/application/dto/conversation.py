"""
Conversation DTOs.

TODO: Implement

Example:
```python
from pydantic import BaseModel
from datetime import datetime

class ConversationDTO(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    last_message: Optional[str] = None

class ConversationListDTO(BaseModel):
    conversations: list[ConversationDTO]
    total: int
```
"""