"""Authentication middleware"""

import jwt
import asyncio
import logging
from functools import wraps
from flask import request, jsonify, g
from src.config.logging_config import correlation_id_var

logger = logging.getLogger(__name__)


def load_identity(secret: str, issuer: str, audience: str):
    """Load identity from JWT token"""
    g.identity = None
    if not secret:
        return

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return

    token = auth_header.split(" ", 1)[1].strip()
    try:
        claims = jwt.decode(
            token,
            secret,
            algorithms=["HS256"],
            audience=audience,
            issuer=issuer,
            options={"require": ["exp", "iat", "aud", "iss"]},
        )
    except Exception as e:
        print(f"JWT decode failed: {e}")
        return

    email = claims.get("email", "")
    dept = claims.get("dept", "")
    if not email or not dept:
        return

    correlation_id = request.headers.get("X-Correlation-ID", "NO Correlation ID")

    # Set in both contextvars (for async/threads) and g (for sync Flask)
    correlation_id_var.set(correlation_id)
    g.identity = {"user_id": email, "dept_id": dept}

    print(f"Set correlation_id: {correlation_id}")


def require_identity(fn):
    """Decorator to require valid authentication (supports both sync and async functions)"""

    @wraps(fn)
    async def async_wrapper(*args, **kwargs):
        identity = getattr(g, "identity", None)
        if not identity:
            return jsonify({"error": "Unauthorized"}), 401

        user_id = identity.get("user_id", "")
        dept_id = identity.get("dept_id", "")
        if not user_id or not dept_id:
            return jsonify({"error": "Unauthorized"}), 401

        return await fn(*args, **kwargs)

    @wraps(fn)
    def sync_wrapper(*args, **kwargs):
        identity = getattr(g, "identity", None)
        if not identity:
            return jsonify({"error": "Unauthorized"}), 401

        user_id = identity.get("user_id", "")
        dept_id = identity.get("dept_id", "")
        if not user_id or not dept_id:
            return jsonify({"error": "Unauthorized"}), 401

        return fn(*args, **kwargs)

    # Return async wrapper if function is async, otherwise sync wrapper
    if asyncio.iscoroutinefunction(fn):
        return async_wrapper
    else:
        return sync_wrapper
