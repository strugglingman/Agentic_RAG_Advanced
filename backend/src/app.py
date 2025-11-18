"""
Flask application factory.
Creates and configures the Flask application with all routes, middleware, and error handlers.
"""

import os
from flask import Flask, jsonify, g
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.exceptions import RequestEntityTooLarge, TooManyRequests
import chromadb
from chromadb.utils.embedding_functions import sentence_transformer_embedding_function

from src.config.settings import get_config
from src.middleware.auth import load_identity
from src.routes.chat import chat_bp
from src.routes.upload import upload_bp
from src.routes.ingest import ingest_bp
from src.routes.files import files_bp
from src.routes.org import org_bp


def get_limiter_key():
    """Get rate limiting key from identity or IP"""
    if not hasattr(g, "identity") or not g.identity:
        return get_remote_address()
    return f"{g.identity.get('dept_id','')}-{g.identity.get('user_id','')}"


def create_app(config_name="development"):
    """
    Application factory for creating Flask app.

    Args:
        config_name: Environment name ('development', 'production', 'testing')

    Returns:
        Tuple of (app, limiter, collection)
    """
    app = Flask(__name__)

    # Load configuration
    config = get_config(config_name)
    app.config.from_object(config)

    # Set maximum upload size
    app.config["MAX_CONTENT_LENGTH"] = int(config.MAX_UPLOAD_MB * 1024 * 1024)

    # Initialize ChromaDB
    embed_model_name = config.EMBED_MODEL_NAME
    chroma_path = config.CHROMA_PATH
    embedding_fun = (
        sentence_transformer_embedding_function.SentenceTransformerEmbeddingFunction(
            model_name=embed_model_name
        )
    )
    chroma_client = chromadb.PersistentClient(path=chroma_path)
    collection = chroma_client.get_or_create_collection(
        name="docs", metadata={"hnsw:space": "cosine"}, embedding_function=embedding_fun
    )

    # Store collection in app context for dependency injection
    app.collection = collection

    # Initialize rate limiter
    limiter = Limiter(
        key_func=get_limiter_key,
        storage_uri=config.RATELIMIT_STORAGE_URI,
        app=app,
        default_limits=config.DEFAULT_RATE_LIMITS,
    )

    # Error handlers
    @app.errorhandler(RequestEntityTooLarge)
    def file_too_large(e):
        return (
            jsonify(
                {
                    "error": f"Error: {str(e)}, Maximum upload size is {config.MAX_UPLOAD_MB} MB."
                }
            ),
            e.code,
        )

    @app.errorhandler(TooManyRequests)
    def ratelimit_error(e):
        return jsonify({"error": "Too many requests. Please try again later."}), 429

    # Authentication middleware
    @app.before_request
    def setup_auth():
        load_identity(
            config.SERVICE_AUTH_SECRET,
            config.SERVICE_AUTH_ISSUER,
            config.SERVICE_AUTH_AUDIENCE,
        )

    # Dependency injection wrapper for routes that need collection
    def inject_collection(f):
        """Decorator to inject collection into route handlers."""
        from functools import wraps

        @wraps(f)
        def wrapper(*args, **kwargs):
            return f(collection, *args, **kwargs)

        return wrapper

    # Register blueprints
    app.register_blueprint(upload_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(ingest_bp)
    app.register_blueprint(files_bp)
    app.register_blueprint(org_bp)

    # Register blueprints that need collection dependency
    # We need to wrap the route functions to inject collection
    from src.routes import chat, ingest, org

    # Monkey-patch the route functions to inject collection
    chat_endpoint = f"{chat_bp.name}.chat"
    chat_route = app.view_functions[chat_endpoint]
    inject_collection(chat_route)
    app.view_functions[chat_endpoint] = inject_collection(chat_route)

    # Inject collection into chat_agent endpoint
    chat_agent_endpoint = f"{chat_bp.name}.chat_agent"
    chat_agent_route = app.view_functions[chat_agent_endpoint]
    app.view_functions[chat_agent_endpoint] = inject_collection(chat_agent_route)

    ingest_endpoint = f"{ingest_bp.name}.ingest"
    ingest_route = app.view_functions[ingest_endpoint]
    app.view_functions[ingest_endpoint] = inject_collection(ingest_route)

    # Apply rate limiting to org-structure endpoint
    limiter.limit("1 per minute; 10 per day", key_func=get_remote_address)(
        app.view_functions[f"{org_bp.name}.org_structure"]
    )

    # Apply rate limiting to chat endpoints
    limiter.limit("30 per minute; 1000 per day")(app.view_functions[chat_endpoint])
    limiter.limit("30 per minute; 1000 per day")(app.view_functions[chat_agent_endpoint])

    # Health check routes
    @app.get("/")
    @limiter.exempt
    def root():
        return jsonify({"message": "Server is running."}), 200

    @app.get("/health")
    @limiter.exempt
    def health():
        return jsonify({"status": "healthy"}), 200

    return app, limiter, collection
