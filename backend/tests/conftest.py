import os
import jwt
import time
import pytest
import sys

# Add backend directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))

# Import FastAPI app
from fastapi.testclient import TestClient
from src.fastapi_app import create_fastapi_app

SERVICE_AUTH_SECRET = os.getenv("SERVICE_AUTH_SECRET", "test-secret")
AUD = os.getenv("SERVICE_AUTH_AUDIENCE", "your_service_audience")
ISS = os.getenv("SERVICE_AUTH_ISSUER", "your_service_name")


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

