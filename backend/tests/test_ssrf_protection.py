import socket

import pytest

from src.utils.ssrf_protection import validate_url

pytestmark = pytest.mark.unit


def test_validate_url_blocks_non_http_scheme():
    with pytest.raises(ValueError) as exc_info:
        validate_url("file:///etc/passwd")
    assert "scheme" in str(exc_info.value).lower()


def test_validate_url_blocks_known_internal_hostname():
    with pytest.raises(ValueError) as exc_info:
        validate_url("http://localhost:8000/health")
    assert "blocked internal hostname" in str(exc_info.value).lower()


def test_validate_url_blocks_private_ip_resolution(monkeypatch):
    def fake_getaddrinfo(hostname, port):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.10", port)),
        ]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    with pytest.raises(ValueError) as exc_info:
        validate_url("https://example.com/file.pdf")

    assert "blocked internal address" in str(exc_info.value).lower()


def test_validate_url_allows_public_resolution(monkeypatch):
    def fake_getaddrinfo(hostname, port):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", port)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.1.1.1", port)),
        ]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    # Should not raise
    validate_url("https://example.com/data.csv")

