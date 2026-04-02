import time

import jwt
import pytest

from src.config.settings import Config


PROTECTED_ROUTES = [
    ("/upload", "post"),
    ("/ingest", "post"),
    ("/ingest/cancel", "post"),
    ("/ingest/active", "get"),
    ("/chat", "post"),
    ("/chat/agent", "post"),
    ("/chat/resume", "post"),
    ("/files", "get"),
    ("/files/delete", "post"),
    ("/conversations", "get"),
]


def _request(client, method: str, route: str, headers=None):
    method_fn = getattr(client, method)

    if route in ("/chat", "/chat/agent"):
        return method_fn(
            route,
            headers=headers,
            json={"messages": [{"role": "user", "content": "hello"}]},
        )
    if route == "/chat/resume":
        return method_fn(
            route,
            headers=headers,
            json={"thread_id": "t1", "confirmed": True},
        )
    if route == "/ingest":
        return method_fn(route, headers=headers, json={"file_ids": ["ALL"]})
    if route == "/ingest/cancel":
        return method_fn(route, headers=headers, json={"job_id": "job-1"})
    if route == "/files/delete":
        return method_fn(route, headers=headers, json={"file_ids": ["file-1"]})
    return method_fn(route, headers=headers)


@pytest.mark.parametrize("route,method", PROTECTED_ROUTES)
def test_protected_routes_missing_auth(client, route, method):
    """
    Protected endpoints must reject requests without Authorization header.
    """
    res = _request(client, method, route)
    assert res.status_code in (401, 403)


@pytest.mark.parametrize("route,method", PROTECTED_ROUTES)
def test_protected_routes_invalid_token(client, route, method):
    """
    Protected endpoints must reject malformed/invalid JWT tokens.
    """
    res = _request(
        client,
        method,
        route,
        headers={"Authorization": "Bearer not-a-valid-jwt"},
    )
    assert res.status_code == 401


@pytest.mark.parametrize("route,method", PROTECTED_ROUTES)
def test_protected_routes_with_auth_not_unauthorized(client, auth_headers, route, method):
    """
    With a valid token, endpoints may still fail validation/business rules,
    but should not fail as unauthorized.
    """
    res = _request(client, method, route, headers=auth_headers)
    assert res.status_code != 401


def test_missing_required_claims_rejected(client):
    now = int(time.time())
    token = jwt.encode(
        {
            "iat": now,
            "exp": now + 300,
            "iss": Config.SERVICE_AUTH_ISSUER,
            "aud": Config.SERVICE_AUTH_AUDIENCE,
            # Missing "email" and "dept" on purpose
        },
        Config.SERVICE_AUTH_SECRET,
        algorithm="HS256",
    )

    res = _request(
        client,
        "post",
        "/chat/agent",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 401
