import pytest


pytestmark = pytest.mark.integration


def _is_prisma_binary_error(exc: Exception) -> bool:
    text = f"{exc!r}\n{exc}"
    return "BinaryNotFoundError" in text or "prisma-query-engine" in text


def test_chat_rate_limit(client, auth_headers):
    payload = {
        "messages": [
            {"role": "user", "content": "Hello, this is a test message."}
        ]
    }
    status_codes = []
    try:
        for _ in range(40):
            res = client.post("/chat/agent", headers=auth_headers, json=payload)
            status_codes.append(res.status_code)
    except Exception as exc:
        if _is_prisma_binary_error(exc):
            pytest.skip(
                "Prisma query engine is not available for integration tests. "
                "Run `prisma py fetch` to enable rate-limit integration checks."
            )
        raise

    assert any(code == 429 for code in status_codes), "Expected at least one 429 Too Many Requests response"
