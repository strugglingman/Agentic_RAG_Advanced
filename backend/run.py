"""
Main entry point for the Flask application.
Run this file to start the server.
"""

import os
import sys
import io

# Set UTF-8 encoding for stdout/stderr to handle Unicode characters on Windows
# This must be done BEFORE any other imports that might use print/logging
if sys.platform == "win32":
    # Reconfigure stdout and stderr with UTF-8 encoding
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )

# Set environment variable for Python's default encoding
os.environ["PYTHONIOENCODING"] = "utf-8"

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the application factory
from src.app import create_app

# Determine environment from FLASK_ENV variable
env = os.getenv("FLASK_ENV", "development")

# Create the application
app, limiter, collection = create_app(env)

if __name__ == "__main__":
    # Get configuration for debug and port settings
    debug = env == "development"
    port = int(os.getenv("PORT", 5001))
    host = os.getenv("HOST", "0.0.0.0")

    print(f"Starting Flask application in {env} mode...")
    print(f"Server running on http://{host}:{port}")

    app.run(host=host, port=port, debug=debug)
