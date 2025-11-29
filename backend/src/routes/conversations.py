"""
Conversation management routes for creating, listing, and managing user conversations.

This module provides endpoints for:
- Listing all conversations for the authenticated user
- Creating new conversations
- Retrieving conversation messages
- Deleting conversations

All endpoints require authentication and enforce user ownership verification.
"""

from flask import Blueprint, request, jsonify, g
from src.middleware.auth import require_identity
from src.services.conversation_service import ConversationService
from src.config.settings import Config

conversations_bp = Blueprint("conversations", __name__)

# Initialize conversation service
conversation_service = ConversationService()


@conversations_bp.get("/conversations")
@require_identity
async def list_conversations():
    """
    List all conversations for the authenticated user.

    TODO: Implement this endpoint

    STEPS:
    1. Get user_email from g.identity.get("user_id")
    2. Connect to database: await conversation_service.connect()
    3. Query conversations from PostgreSQL:
       - Use: await conversation_service.prisma_client.conversation.find_many()
       - Filter: where={"user_email": user_email}
       - Order: order={"updated_at": "desc"}  (newest first)
       - Optional: Limit to last 50 conversations: take=50

    4. For each conversation, create response dict with:
       - id: conversation.id
       - title: conversation.title
       - updated_at: conversation.updated_at (convert to ISO string)
       - preview: First 50 chars of last message (optional)

    5. Return: jsonify({"conversations": [...]})

    SECURITY:
    - Already filtered by user_email, so user only sees their own conversations

    RESPONSE FORMAT:
    {
        "conversations": [
            {
                "id": "uuid",
                "title": "Conversation title",
                "updated_at": "2025-01-27T12:00:00Z",
                "preview": "Last message preview..."
            },
            ...
        ]
    }
    """
    user_id = g.identity.get("user_id", "")
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        conversations_list = await conversation_service.get_user_conversations_list(
            user_id, limit=Config.CONVERSATION_USER_LIMIT
        )
        if not conversations_list:
            return jsonify({"conversations": []})

        conversation_dicts = []
        for conv in conversations_list:
            conversation_dicts.append(
                {
                    "id": conv.get("id", ""),
                    "title": conv.get("title", ""),
                    "updated_at": (
                        conv.get("updated_at", "").isoformat()
                        if conv.get("updated_at")
                        else ""
                    ),
                    "preview": (
                        conv.get("messages", [])[0].get("content", "")[:50]
                        if conv.get("messages")
                        else ""
                    ),
                }
            )

        return jsonify({"conversations": conversation_dicts})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@conversations_bp.post("/conversations")
@require_identity
async def create_conversation():
    """
    Create a new conversation for the authenticated user.

    TODO: Implement this endpoint

    REQUEST BODY:
    {
        "title": "Optional conversation title"
    }

    STEPS:
    1. Get user_email from g.identity.get("user_id")
    2. Get title from request: request.json.get("title", "New Conversation")
    3. Create conversation:
       conversation_id = await conversation_service.create_conversation(
           user_email=user_email,
           title=title
       )
    4. Return: jsonify({"id": conversation_id, "title": title})

    RESPONSE FORMAT:
    {
        "id": "uuid",
        "title": "Conversation title"
    }
    """
    pass


@conversations_bp.get("/conversations/<conversation_id>")
@require_identity
async def get_conversation(conversation_id: str):
    """
    Get conversation details and messages.

    TODO: Implement this endpoint

    STEPS:
    1. Get user_email from g.identity.get("user_id")
    2. Connect to database: await conversation_service.connect()

    3. CRITICAL - Verify ownership:
       conversation = await conversation_service.prisma_client.conversation.find_unique(
           where={"id": conversation_id}
       )
       if not conversation:
           return jsonify({"error": "Conversation not found"}), 404
       if conversation.user_email != user_email:
           return jsonify({"error": "Unauthorized"}), 403

    4. Get messages:
       messages = await conversation_service.load_conversation_history_db(
           conversation_id,
           limit=Config.CONVERSATION_MESSAGE_LIMIT
       )

    5. Return: jsonify({
           "id": conversation.id,
           "title": conversation.title,
           "messages": messages,
           "created_at": conversation.created_at,
           "updated_at": conversation.updated_at
       })

    SECURITY:
    - MUST verify conversation.user_email == user_email before returning data
    - Prevents users from accessing other users' conversations by guessing IDs

    RESPONSE FORMAT:
    {
        "id": "uuid",
        "title": "Conversation title",
        "created_at": "2025-01-27T12:00:00Z",
        "updated_at": "2025-01-27T12:30:00Z",
        "messages": [
            {
                "id": "uuid",
                "role": "user",
                "content": "Message content",
                "created_at": "2025-01-27T12:00:00Z"
            },
            ...
        ]
    }
    """
    user_id = g.identity.get("user_id", "")
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401
    try:
        await conversation_service.connect()
        conversation = (
            await conversation_service.prisma_client.conversation.find_unique(
                where={"id": conversation_id}
            )
        )
        if not conversation:
            return jsonify({"error": "Conversation not found"}), 404
        if conversation.user_email != user_id:
            return jsonify({"error": "Unauthorized"}), 403

        messages = await conversation_service.load_message_history_db(
            conversation_id, limit=Config.CONVERSATION_MESSAGE_LIMIT
        )
        message_dicts = []
        for msg in messages:
            message_dicts.append(
                {
                    "id": msg.get("id", ""),
                    "role": msg.get("role", ""),
                    "content": msg.get("content", ""),
                    "created_at": (
                        msg.get("created_at", "").isoformat()
                        if msg.get("created_at")
                        else ""
                    ),
                }
            )
        return jsonify(
            {
                "id": conversation.id,
                "title": conversation.title,
                "created_at": conversation.created_at.isoformat(),
                "updated_at": conversation.updated_at.isoformat(),
                "messages": message_dicts,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@conversations_bp.delete("/conversations/<conversation_id>")
@require_identity
async def delete_conversation(conversation_id: str):
    """
    Delete a conversation and all its messages.

    TODO: Implement this endpoint

    STEPS:
    1. Get user_email from g.identity.get("user_id")
    2. Connect to database: await conversation_service.connect()

    3. CRITICAL - Verify ownership:
       conversation = await conversation_service.prisma_client.conversation.find_unique(
           where={"id": conversation_id}
       )
       if not conversation:
           return jsonify({"error": "Conversation not found"}), 404
       if conversation.user_email != user_email:
           return jsonify({"error": "Unauthorized"}), 403

    4. Delete conversation:
       success = await conversation_service.delete_conversation(conversation_id)

    5. Return: jsonify({"success": success})

    SECURITY:
    - MUST verify ownership before deletion
    - Cascade delete will automatically remove all messages (defined in schema)

    RESPONSE FORMAT:
    {
        "success": true
    }
    """
    user_id = g.identity.get("user_id", "")
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        success = await conversation_service.delete_conversation(
            conversation_id, user_id
        )
        if not success:
            return jsonify({"error": "Unauthorized or conversation not found"}), 403

        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@conversations_bp.patch("/conversations/<conversation_id>")
@require_identity
async def update_conversation(conversation_id: str):
    """
    Update conversation metadata (e.g., title).

    TODO: Implement this endpoint (OPTIONAL)

    REQUEST BODY:
    {
        "title": "New conversation title"
    }

    STEPS:
    1. Get user_email from g.identity.get("user_id")
    2. Get new title from request: request.json.get("title")
    3. Verify ownership (same as delete endpoint)
    4. Update:
       await conversation_service.prisma_client.conversation.update(
           where={"id": conversation_id},
           data={"title": new_title}
       )
    5. Return updated conversation

    SECURITY:
    - MUST verify ownership before update

    RESPONSE FORMAT:
    {
        "id": "uuid",
        "title": "Updated title",
        "updated_at": "2025-01-27T12:45:00Z"
    }
    """
    pass


# ==================== INTEGRATION NOTES ====================

"""
AFTER IMPLEMENTING THIS FILE:
============================

1. Register the blueprint in src/app.py:

   from src.routes.conversations import conversations_bp
   app.register_blueprint(conversations_bp)

2. Apply rate limiting (optional but recommended):

   from src.app import limiter

   limiter.limit("60 per minute")(app.view_functions[f"{conversations_bp.name}.list_conversations"])
   limiter.limit("30 per minute")(app.view_functions[f"{conversations_bp.name}.create_conversation"])
   limiter.limit("10 per minute")(app.view_functions[f"{conversations_bp.name}.delete_conversation"])

3. Test endpoints with curl or Python:

   # List conversations
   curl -H "Authorization: Bearer <token>" http://localhost:5001/conversations

   # Create conversation
   curl -X POST -H "Authorization: Bearer <token>" -H "Content-Type: application/json" \
        -d '{"title": "My Conversation"}' http://localhost:5001/conversations

   # Get conversation
   curl -H "Authorization: Bearer <token>" http://localhost:5001/conversations/<id>

   # Delete conversation
   curl -X DELETE -H "Authorization: Bearer <token>" http://localhost:5001/conversations/<id>

4. Common errors to watch for:
   - Missing await on async calls (will return coroutine instead of result)
   - Not verifying ownership (security vulnerability!)
   - Not handling conversation not found case (404)
   - Forgetting to connect to database before queries

5. Database indexes (already in schema.prisma):
   - @@index([user_email]) on Conversation - for fast filtering by user
   - @@index([updated_at]) on Conversation - for sorting by recent

   If queries are slow, check EXPLAIN ANALYZE in PostgreSQL.
"""
