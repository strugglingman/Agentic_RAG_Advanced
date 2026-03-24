"""
Manual smoke test for conversation CRUD endpoints on a running backend.
"""

import os
import uuid

import pytest
import requests

from jwt_generation import generate_jwt_token


pytestmark = [pytest.mark.manual, pytest.mark.integration]

BASE_URL = os.getenv("MANUAL_BACKEND_BASE_URL", "http://localhost:5001")
TIMEOUT_SECONDS = int(os.getenv("MANUAL_HTTP_TIMEOUT_SECONDS", "20"))


@pytest.fixture
def manual_api_headers():
    if os.getenv("RUN_MANUAL_TESTS") != "1":
        pytest.skip("Set RUN_MANUAL_TESTS=1 to run manual API smoke tests.")
    return {
        "Authorization": f"Bearer {generate_jwt_token()}",
        "Content-Type": "application/json",
    }


def _request(method: str, path: str, headers: dict, **kwargs):
    try:
        return requests.request(
            method,
            f"{BASE_URL}{path}",
            headers=headers,
            timeout=TIMEOUT_SECONDS,
            **kwargs,
        )
    except requests.exceptions.ConnectionError:
        pytest.skip(
            f"Backend is not reachable at {BASE_URL}. "
            "Start the API server or set MANUAL_BACKEND_BASE_URL."
        )


def test_conversation_crud_smoke(manual_api_headers):
    """
    Full CRUD smoke:
    create -> list -> get -> patch -> delete
    """
    create_payload = {"title": f"Manual smoke {uuid.uuid4().hex[:8]}"}
    create_resp = _request("POST", "/conversations", headers=manual_api_headers, json=create_payload)
    assert create_resp.status_code == 201, create_resp.text
    created = create_resp.json()
    conv_id = created["id"]
    assert created["title"] == create_payload["title"]

    list_resp = _request("GET", "/conversations", headers=manual_api_headers)
    assert list_resp.status_code == 200, list_resp.text
    conversations = list_resp.json().get("conversations", [])
    assert any(c["id"] == conv_id for c in conversations)

    get_resp = _request("GET", f"/conversations/{conv_id}", headers=manual_api_headers)
    assert get_resp.status_code == 200, get_resp.text
    conversation = get_resp.json()
    assert conversation["id"] == conv_id

    new_title = f"Updated {uuid.uuid4().hex[:8]}"
    patch_resp = _request(
        "PATCH",
        f"/conversations/{conv_id}",
        headers=manual_api_headers,
        json={"title": new_title},
    )
    assert patch_resp.status_code == 200, patch_resp.text
    patched = patch_resp.json()
    assert patched["id"] == conv_id
    assert patched["title"] == new_title

    delete_resp = _request("DELETE", f"/conversations/{conv_id}", headers=manual_api_headers)
    assert delete_resp.status_code == 200, delete_resp.text
    assert delete_resp.json().get("success") is True
