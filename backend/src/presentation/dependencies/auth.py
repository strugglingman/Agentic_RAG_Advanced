"""
Authentication Dependency for FastAPI.

Guidelines:
- Extracts and validates JWT token from Authorization header
- Returns User entity for use in route handlers
- Raises HTTPException 401 if unauthorized

Steps to implement:
1. Get token from Authorization header (Bearer scheme)
2. Decode JWT using same secret/issuer/audience as Flask version
3. Extract user_email and dept_id from claims
4. Return User entity

Config needed (from src.config.settings):
- JWT_SECRET
- JWT_ISSUER
- JWT_AUDIENCE
"""

import jwt
from dataclasses import dataclass
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from src.domain.value_objects.user_email import UserEmail
from src.domain.value_objects.dept_id import DeptId
from src.config.settings import Config


@dataclass
class AuthUser:
    email: UserEmail
    dept: DeptId

    def __post_init__(self):
        if not self.email or not self.dept:
            raise ValueError("AuthUser must have both email and dept defined.")


security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> AuthUser:
    """
    Extract and validate user from JWT token.

    Raises:
        HTTPException 401 if token is invalid, expired, or missing required claims
    """
    claims = None
    try:
        token = credentials.credentials
        claims = jwt.decode(
            token,
            Config.SERVICE_AUTH_SECRET,
            algorithms=["HS256"],
            audience=Config.SERVICE_AUTH_AUDIENCE,
            issuer=Config.SERVICE_AUTH_ISSUER,
            options={"require": ["exp", "iat", "aud", "iss"]},
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}",
        )

    if not claims:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token claims",
        )

    email = claims.get("email")
    dept = claims.get("dept")
    if not email or not dept:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing required claims in token",
        )

    return AuthUser(
        email=UserEmail(email),
        dept=DeptId(dept),
    )
