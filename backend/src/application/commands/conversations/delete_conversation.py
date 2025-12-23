"""
Delete Conversation Command.

Guidelines:
- Command: @dataclass(frozen=True) with fields: conversation_id, user_email
- Must verify ownership before deletion (security)
- Raises EntityNotFoundError if conversation doesn't exist
- Raises AccessDeniedError if user doesn't own the conversation
- Returns: bool

Steps to implement:
1. Create DeleteConversationCommand(Command[bool]) with fields: conversation_id, user_email
2. Create DeleteConversationHandler(CommandHandler[bool])
3. Handler.__init__ receives ConversationRepository
4. Handler.execute():
   - Create value objects (ConversationId, UserEmail)
   - Fetch conversation via repo.get_by_id()
   - If not found → raise EntityNotFoundError
   - If user doesn't own → raise AccessDeniedError
   - Delete via repo.delete()
   - Return True
"""