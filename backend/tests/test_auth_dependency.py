import time
import asyncio

import jwt
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from src.config.settings import Config
from src.presentation.dependencies.auth import get_current_user

pytestmark = pytest.mark.unit


def _mint_token(
    *,
    secret: str,
    issuer: str,
    audience: str,
    email: str | None = "user@example.com",
    dept: str | None = "ENG",
    exp_delta_seconds: int = 300,
) -> str:
    now = int(time.time())
    payload = {
        "iat": now,
        "exp": now + exp_delta_seconds,
        "iss": issuer,
        "aud": audience,
    }
    if email is not None:
        payload["email"] = email
    if dept is not None:
        payload["dept"] = dept
    return jwt.encode(payload, secret, algorithm="HS256")


def _creds(token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


def test_get_current_user_valid_token(monkeypatch):
    monkeypatch.setattr(Config, "SERVICE_AUTH_SECRET", "test-secret", raising=False)
    monkeypatch.setattr(Config, "SERVICE_AUTH_ISSUER", "test-issuer", raising=False)
    monkeypatch.setattr(Config, "SERVICE_AUTH_AUDIENCE", "test-audience", raising=False)

    token = _mint_token(
        secret=Config.SERVICE_AUTH_SECRET,
        issuer=Config.SERVICE_AUTH_ISSUER,
        audience=Config.SERVICE_AUTH_AUDIENCE,
        email="alice@example.com",
        dept="SALES",
    )

    user = asyncio.run(get_current_user(_creds(token)))
    assert user.email.value == "alice@example.com"
    assert user.dept.value == "SALES"


def test_get_current_user_expired_token(monkeypatch):
    monkeypatch.setattr(Config, "SERVICE_AUTH_SECRET", "test-secret", raising=False)
    monkeypatch.setattr(Config, "SERVICE_AUTH_ISSUER", "test-issuer", raising=False)
    monkeypatch.setattr(Config, "SERVICE_AUTH_AUDIENCE", "test-audience", raising=False)

    token = _mint_token(
        secret=Config.SERVICE_AUTH_SECRET,
        issuer=Config.SERVICE_AUTH_ISSUER,
        audience=Config.SERVICE_AUTH_AUDIENCE,
        exp_delta_seconds=-1,
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(get_current_user(_creds(token)))

    assert exc_info.value.status_code == 401
    assert "expired" in str(exc_info.value.detail).lower()


def test_get_current_user_invalid_audience(monkeypatch):
    monkeypatch.setattr(Config, "SERVICE_AUTH_SECRET", "test-secret", raising=False)
    monkeypatch.setattr(Config, "SERVICE_AUTH_ISSUER", "test-issuer", raising=False)
    monkeypatch.setattr(Config, "SERVICE_AUTH_AUDIENCE", "expected-audience", raising=False)

    token = _mint_token(
        secret=Config.SERVICE_AUTH_SECRET,
        issuer=Config.SERVICE_AUTH_ISSUER,
        audience="wrong-audience",
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(get_current_user(_creds(token)))

    assert exc_info.value.status_code == 401
    assert "invalid token" in str(exc_info.value.detail).lower()


def test_get_current_user_missing_required_claims(monkeypatch):
    monkeypatch.setattr(Config, "SERVICE_AUTH_SECRET", "test-secret", raising=False)
    monkeypatch.setattr(Config, "SERVICE_AUTH_ISSUER", "test-issuer", raising=False)
    monkeypatch.setattr(Config, "SERVICE_AUTH_AUDIENCE", "test-audience", raising=False)

    token_missing_dept = _mint_token(
        secret=Config.SERVICE_AUTH_SECRET,
        issuer=Config.SERVICE_AUTH_ISSUER,
        audience=Config.SERVICE_AUTH_AUDIENCE,
        email="user@example.com",
        dept=None,
    )
    with pytest.raises(HTTPException) as exc_info_dept:
        asyncio.run(get_current_user(_creds(token_missing_dept)))
    assert exc_info_dept.value.status_code == 401
    assert "missing required claims" in str(exc_info_dept.value.detail).lower()

    token_missing_email = _mint_token(
        secret=Config.SERVICE_AUTH_SECRET,
        issuer=Config.SERVICE_AUTH_ISSUER,
        audience=Config.SERVICE_AUTH_AUDIENCE,
        email=None,
        dept="ENG",
    )
    with pytest.raises(HTTPException) as exc_info_email:
        asyncio.run(get_current_user(_creds(token_missing_email)))
    assert exc_info_email.value.status_code == 401
    assert "missing required claims" in str(exc_info_email.value.detail).lower()
