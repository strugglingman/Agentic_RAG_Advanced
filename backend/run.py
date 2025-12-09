"""
Main entry point for the Flask application.
Run this file to start the server.
"""

import os
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
