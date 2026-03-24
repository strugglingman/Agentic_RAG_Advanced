import os
import io
import pytest

pytestmark = pytest.mark.integration

MAX_UPLOAD_SIZE_IN_BYTE = int(float(os.getenv("MAX_UPLOAD_MB", "25")) * 1024 * 1024)


def _is_prisma_binary_error(exc: Exception) -> bool:
    text = f"{exc!r}\n{exc}"
    return "BinaryNotFoundError" in text or "prisma-query-engine" in text


def test_upload_file_size_limit(client, auth_headers):
    files = {"file": ("big.pdf", io.BytesIO(b"0" * (MAX_UPLOAD_SIZE_IN_BYTE + 10)), "application/pdf")}
    try:
        res = client.post("/upload", headers=auth_headers, files=files)
    except Exception as exc:
        if _is_prisma_binary_error(exc):
            pytest.skip(
                "Prisma query engine is not available for integration tests. "
                "Run `prisma py fetch` to enable upload integration checks."
            )
        raise
    assert res.status_code == 413, f"Expected 413 Payload Too Large, got {res.status_code}"


def test_upload_file_within_limit(client, auth_headers):
    files = {"file": ("small.pdf", io.BytesIO(b"0" * (MAX_UPLOAD_SIZE_IN_BYTE - 1000)), "application/pdf")}
    try:
        res = client.post("/upload", headers=auth_headers, files=files)
    except Exception as exc:
        if _is_prisma_binary_error(exc):
            pytest.skip(
                "Prisma query engine is not available for integration tests. "
                "Run `prisma py fetch` to enable upload integration checks."
            )
        raise
    assert res.status_code == 200, f"Expected 200 OK, got {res.status_code}"
