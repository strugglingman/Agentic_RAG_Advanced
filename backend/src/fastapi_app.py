"""
FastAPI Application Factory.
Creates and configures the FastAPI application with all routers, middleware, and DI.

Migrated endpoints:
- chat, conversations, files, upload, ingest, org-structure
"""

# ---------------------------------------------------------------------------
# Windows UTF-8 console fix  (must run before ANY import that prints/logs)
# ---------------------------------------------------------------------------
# On Windows the default console encoding (cp1252 / cp936 …) cannot represent
# characters used in logs and prompts (→ ✓ ✗ ✅ 📊, CJK text, etc.).
# Wrapping stdout/stderr at module level ensures every code path — including
# uvicorn --reload worker processes — gets UTF-8 output with safe fallback.
# ---------------------------------------------------------------------------
import sys, os, io  # noqa: E401, E402

if sys.platform == "win32":
    for _stream_name in ("stdout", "stderr"):
        _stream = getattr(sys, _stream_name)
        if hasattr(_stream, "buffer") and not (
            hasattr(_stream, "encoding") and _stream.encoding == "utf-8"
        ):
            setattr(
                sys,
                _stream_name,
                io.TextIOWrapper(
                    _stream.buffer,
                    encoding="utf-8",
                    errors="replace",
                    line_buffering=True,
                ),
            )
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from dishka import make_async_container
from dishka.integrations.fastapi import setup_dishka
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from src.config.logging_config import setup_logging, correlation_id_var
from src.config.settings import Config
from src.setup.ioc.container import AppProvider
from src.presentation.api import (
    chat_router,
    conversations_router,
    upload_router,
    files_router,
    org_router,
    ingest_router,
    metrics_router,
)
from src.adapters.slack.slack_routes import router as slack_router

# Setup logging
setup_logging(Config.LOG_LEVEL, Config.LOG_PATH)


# ==============================================================================
# Rate Limiter Setup
# ==============================================================================
def get_user_identifier(request: Request) -> str:
    """
    Get rate limit key from auth header or IP address.
    Matches Flask behavior: dept_id-user_id if authenticated, else IP.
    """
    # Try to get user info from auth header (JWT)
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        try:
            import jwt

            token = auth_header[7:]
            # Decode without verification just to get claims for rate limiting
            payload = jwt.decode(token, options={"verify_signature": False})
            dept_id = payload.get("dept", "")
            user_id = payload.get("email", "")
            if dept_id and user_id:
                return f"{dept_id}-{user_id}"
        except Exception:
            pass
    # Fallback to IP address
    return get_remote_address(request)


# Create limiter instance (matches Flask-Limiter configuration)
limiter = Limiter(
    key_func=get_user_identifier,
    storage_uri=Config.RATELIMIT_STORAGE_URI,
    default_limits=Config.DEFAULT_RATE_LIMITS,
)


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Middleware to extract and set correlation ID from request headers."""

    async def dispatch(self, request: Request, call_next):
        # Get correlation ID from header or use default
        correlation_id = request.headers.get("X-Correlation-ID", "NO Correlation ID")

        # Set in contextvars (propagates to async tasks and logging)
        correlation_id_var.set(correlation_id)

        # Process request
        response = await call_next(request)

        # Optionally add correlation ID to response headers
        response.headers["X-Correlation-ID"] = correlation_id

        return response


class MaxBodySizeMiddleware(BaseHTTPMiddleware):
    """
    Middleware to limit request body size (matches Flask MAX_CONTENT_LENGTH).

    This rejects requests with Content-Length exceeding the limit BEFORE
    reading the body, which is more efficient than reading then checking.
    """

    def __init__(self, app, max_body_size: int):
        super().__init__(app)
        self.max_body_size = max_body_size

    async def dispatch(self, request: Request, call_next):
        # Check Content-Length header if present
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > self.max_body_size:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "error": f"Request body too large. Maximum size is {Config.MAX_UPLOAD_MB} MB."
                        },
                    )
            except ValueError:
                pass  # Invalid content-length, let the request proceed

        return await call_next(request)


# Create container at module level (before app starts)
# This is required because Dishka adds middleware, which must happen before app starts
container = make_async_container(AppProvider())


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.

    Handles startup and shutdown events:
    - Startup: Log that app started (container already created)
    - Shutdown: Close DI container (disconnects Prisma, etc.)
    """
    # Startup - container and Dishka already setup before app starts
    print("FastAPI application started. DI container initialized.")
    yield
    # Shutdown
    await container.close()
    print("FastAPI application shutdown. DI container closed.")


def create_fastapi_app() -> FastAPI:
    """
    Application factory for creating FastAPI app.

    Returns:
        FastAPI application instance
    """
    app = FastAPI(
        title="Agentic RAG API",
        description="FastAPI backend for Agentic RAG application",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Setup Dishka BEFORE app starts (must add middleware before startup)
    setup_dishka(container, app)

    # Setup rate limiter
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Correlation ID middleware (must be added before CORS)
    app.add_middleware(CorrelationIdMiddleware)

    # Max body size middleware (matches Flask MAX_CONTENT_LENGTH)
    app.add_middleware(MaxBodySizeMiddleware, max_body_size=Config.MAX_CONTENT_LENGTH)

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Validation error handler - shows detailed Pydantic errors
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        errors = exc.errors()
        print(f"[VALIDATION ERROR] {errors}")  # Log to console
        return JSONResponse(
            status_code=400,
            content={"error": "Validation error", "details": errors},
        )

    # HTTP exception handler - catch all HTTPException including 400s
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        import traceback

        print(f"[HTTP ERROR {exc.status_code}] {exc.detail}")
        print(f"[HTTP ERROR TRACEBACK]\n{traceback.format_exc()}")
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.detail},
        )

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        import traceback

        print(f"[GLOBAL ERROR] {type(exc).__name__}: {exc}")  # Log to console
        print(f"[GLOBAL ERROR TRACEBACK]\n{traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal server error: {str(exc)}"},
        )

    # Health check routes
    @app.get("/", tags=["health"])
    async def root():
        return {"message": "FastAPI server is running."}

    @app.get("/health", tags=["health"])
    async def health():
        return {"status": "healthy"}

    # Register routers
    app.include_router(chat_router)
    app.include_router(conversations_router)
    app.include_router(upload_router)  # POST /upload
    app.include_router(files_router)  # GET /files, GET /files/{file_id}
    app.include_router(org_router)  # GET /org-structure
    app.include_router(ingest_router)  # POST /ingest
    app.include_router(slack_router)  # Slack integration routes
    app.include_router(metrics_router)  # Prometheus /metrics endpoint

    return app


# Create the app instance
app = create_fastapi_app()
