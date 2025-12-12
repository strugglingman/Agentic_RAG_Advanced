import logging
from logging.handlers import RotatingFileHandler
import io
import sys
from pathlib import Path
from contextvars import ContextVar
from flask import has_request_context, g
from src.config.settings import Config

# Context variable to store correlation ID across async/thread boundaries
correlation_id_var: ContextVar[str] = ContextVar(
    "correlation_id", default="NO Correlation ID"
)


class CorrelationIdFilter(logging.Filter):
    """Logging filter to add correlation ID to log records."""

    def filter(self, record):
        # Try contextvars first (works across threads/async)
        correlation_id = correlation_id_var.get()

        # Fallback to Flask's g if available
        if (
            correlation_id == "NO Correlation ID"
            and has_request_context()
            and hasattr(g, "identity")
        ):
            print("*****************Come into filter again")
            g_identity = g.identity
            if isinstance(g_identity, dict):
                correlation_id = g_identity.get("correlation_id", "NO Correlation ID")

        record.correlation_id = correlation_id
        return True


class SafeFormatter(logging.Formatter):
    """Formatter that ensures correlation_id always exists."""

    def format(self, record):
        if not hasattr(record, "correlation_id"):
            record.correlation_id = "NO Correlation ID"
        return super().format(record)


def setup_logging(level: str = "INFO", log_file: str | None = None):
    # Set up root logger
    print(f"Setting up logging: level={level}, log_file={log_file}")
    root = logging.getLogger()
    root.setLevel(logging.WARNING)  # Set root to WARNING to avoid too much noise
    # Add a filter for correlation ID if exists
    root.addFilter(CorrelationIdFilter())
    logger_handler = logging.StreamHandler(
        io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    )
    formatter = SafeFormatter(Config.LOG_FORMAT)
    logger_handler.setFormatter(formatter)
    logger_handler.addFilter(CorrelationIdFilter())
    root.addHandler(logger_handler)

    # Set up file logging if log_file provided with rotation
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(CorrelationIdFilter())
        root.addHandler(file_handler)

    logging.info("Logging is set up.")

    # # Silent specific noisy loggers
    # noisy_loggers = [
    #     # HTTP/Network
    #     "urllib3.connectionpool",
    #     "httpx",
    #     "httpcore",
    #     "hpack",  # ← ADD (HTTP/2 header compression - decode_integer, _decode_literal)
    #     # LangChain/LangGraph
    #     "langchain",
    #     "langchain_core",
    #     "langchain_openai",
    #     "langgraph",
    #     # Vector DB
    #     "chromadb",
    #     # OpenAI
    #     "openai",
    #     "openai._base_client",
    #     # Prisma / Database    ← ADD these
    #     "prisma",
    #     "prisma.engine",
    #     # Other
    #     "chardet.charsetprober",
    #     "sentence_transformers",
    #     "asyncio",  # ← ADD (IocpProactor logs)
    # ]
    # for nl in noisy_loggers:
    #     logging.getLogger(nl).setLevel(logging.WARNING)

    # Remove all other logs except from src

    logging.getLogger("src").setLevel(getattr(logging, level.upper(), logging.INFO))

    return root
