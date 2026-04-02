import os
import jwt
import time
import pytest
import sys

# Disable LangSmith tracing by default for automated tests.
# Keep API keys intact so manual/external lanes can opt in when needed.
if os.getenv("RUN_MANUAL_TESTS") != "1":
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
    os.environ.setdefault("LANGSMITH_TRACING", "false")

# Add backend directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))

# Import FastAPI app
from fastapi.testclient import TestClient
from src.fastapi_app import create_fastapi_app
from src.config.settings import Config

# Keep test token generation aligned with runtime auth validation config.
SERVICE_AUTH_SECRET = Config.SERVICE_AUTH_SECRET
AUD = Config.SERVICE_AUTH_AUDIENCE
ISS = Config.SERVICE_AUTH_ISSUER
LANE_MARKERS = {"unit", "component", "integration", "external", "manual", "e2e"}


def _service_token(email="user@example.com", dept="eng", sid="test-sid"):
    now = int(time.time())
    return jwt.encode(
        {
            "sub": email,
            "email": email,
            "dept": dept,
            "sid": sid,
            "iat": now,
            "exp": now + 300,
            "iss": ISS,
            "aud": AUD,
        },
        SERVICE_AUTH_SECRET,
        algorithm="HS256",
    )


@pytest.fixture()
def app():
    """Create and configure a new FastAPI app instance for each test."""
    return create_fastapi_app()


@pytest.fixture()
def client(app):
    """A test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture()
def auth_headers():
    """Authentication headers with valid JWT token."""
    return {"Authorization": f"Bearer {_service_token()}"}


def pytest_collection_modifyitems(config, items):
    """
    Enforce test lane ownership for every collected test.

    This prevents unclassified tests from silently bypassing the lane model.
    """
    unclassified = []
    for item in items:
        if not any(item.get_closest_marker(marker) for marker in LANE_MARKERS):
            unclassified.append(item.nodeid)

    if unclassified:
        formatted = "\n".join(f"- {nodeid}" for nodeid in unclassified)
        raise pytest.UsageError(
            "Every test must define at least one lane marker: "
            f"{', '.join(sorted(LANE_MARKERS))}\n"
            "Unclassified tests:\n"
            f"{formatted}"
        )

