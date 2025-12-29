"""
FastAPI Application Factory.
Creates and configures the FastAPI application with all routers, middleware, and DI.

Migrated endpoints:
- chat, conversations, files, upload, ingest, org-structure
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from dishka import make_async_container
from dishka.integrations.fastapi import setup_dishka

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
)

# Setup logging
setup_logging(Config.LOG_LEVEL, Config.LOG_PATH)


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

    # Correlation ID middleware (must be added before CORS)
    app.add_middleware(CorrelationIdMiddleware)

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
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
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

    return app


# Create the app instance
app = create_fastapi_app()
