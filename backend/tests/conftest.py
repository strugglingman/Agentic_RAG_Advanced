import os
import jwt
import time
import pytest
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))
from app import app as flask_app

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
    flask_app.config.update(TESTING=True)
    return flask_app


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def auth_headers():
    return {"Authorization": f"Bearer {_service_token()}"}
