import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add parent directory to path so we can import src
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

import jwt
from src.config.settings import Config


def generate_jwt_token(
    email: str = "strugglingman@gmail.com", dept: str = "MYHB|software|ml"
) -> str:
    """Generate a valid JWT token for testing API endpoints"""
    payload = {
        "email": email,  # Changed from user_id
        "dept": dept,  # Changed from dept_id
        "sid": "test-session-id",
        "exp": datetime.now(timezone.utc) + timedelta(hours=1),
        "iat": datetime.now(timezone.utc),
        "iss": Config.SERVICE_AUTH_ISSUER,
        "aud": Config.SERVICE_AUTH_AUDIENCE,
    }

    token = jwt.encode(payload, Config.SERVICE_AUTH_SECRET, algorithm="HS256")
    return token


if __name__ == "__main__":
    token = generate_jwt_token()
    print(f"Bearer {token}")
