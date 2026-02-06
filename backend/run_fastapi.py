"""
Main entry point for the FastAPI application.
Run this file to start the FastAPI server.

Usage:
    python run_fastapi.py

Or with uvicorn directly:
    uvicorn src.fastapi_app:app --host 0.0.0.0 --port 5001 --reload
"""

import os
import sys

# NOTE: Windows UTF-8 console fix is now handled in src/fastapi_app.py at
# module level, so it also covers uvicorn --reload worker processes.

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import uvicorn

if __name__ == "__main__":
    env = os.getenv("FLASK_ENV", "development")  # Reuse same env var
    debug = env == "development"
    port = int(os.getenv("PORT", 5001))
    host = os.getenv("HOST", "0.0.0.0")

    print(f"Starting FastAPI application in {env} mode...")
    print(f"Server running on http://{host}:{port}")
    print(f"API docs available at http://{host}:{port}/docs")

    uvicorn.run(
        "src.fastapi_app:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info" if debug else "warning",
    )
