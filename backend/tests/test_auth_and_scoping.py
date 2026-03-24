import pytest

pytestmark = pytest.mark.component


def _is_prisma_binary_error(exc: Exception) -> bool:
    text = f"{exc!r}\n{exc}"
    return "BinaryNotFoundError" in text or "prisma-query-engine" in text


def _request_or_skip_for_prisma(client, method: str, route: str, **kwargs):
    try:
        return getattr(client, method)(route, **kwargs)
    except Exception as exc:
        if _is_prisma_binary_error(exc):
            pytest.skip(
                "Prisma query engine is not available for integration tests. "
                "Run `prisma py fetch` to enable this lane locally."
            )
        raise


routes = [
    ("/upload", "post", 401),
    ("/ingest", "post", 401),
    ("/chat/agent", "post", 401),
    ("/files", "get", 401),
]


@pytest.mark.parametrize("route,method,expected_code", routes)
def test_protected_routes_missing_auth(client, route, method, expected_code):
    """Requests without auth should return 401 Unauthorized."""
    res = _request_or_skip_for_prisma(client, method, route)
    assert res.status_code == expected_code
    payload = {}
    try:
        payload = res.json()
    except Exception:
        payload = {}
    assert "error" in payload or "detail" in payload


@pytest.mark.parametrize("route,method,_", routes)
@pytest.mark.integration
def test_protected_routes_with_auth(client, auth_headers, route, method, _):
    """Requests with valid auth should not return 401 Unauthorized."""
    res = _request_or_skip_for_prisma(client, method, route, headers=auth_headers)
    assert res.status_code in [
        200,
        201,
        204,
        400,
        403,
        404,
        405,
        422,
    ]  # Acceptable codes for protected endpoints
