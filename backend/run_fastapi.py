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

# CRITICAL: Set UTF-8 mode for ALL Python processes (including uvicorn workers)
# Must be set BEFORE any other imports
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"  # Force UTF-8 mode system-wide (PEP 540)

# Set UTF-8 encoding for stdout/stderr to handle Unicode characters on Windows
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "buffer"):
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer,
                encoding="utf-8",
                errors="replace",
                line_buffering=True,
            )
        if hasattr(sys.stderr, "buffer"):
            sys.stderr = io.TextIOWrapper(
                sys.stderr.buffer,
                encoding="utf-8",
                errors="replace",
                line_buffering=True,
            )
    except (AttributeError, ValueError):
        pass  # Already wrapped

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
