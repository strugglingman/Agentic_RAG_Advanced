import os
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


class DescendingFileHandler(RotatingFileHandler):
    """Custom handler that writes log entries in descending order (newest first)."""

    def emit(self, record):
        """
        Emit a record by prepending it to the file instead of appending.
        This ensures newest logs appear at the top.
        """
        try:
            msg = self.format(record)
            # Read existing content if file exists
            existing_content = ""
            if os.path.exists(self.baseFilename):
                try:
                    with open(self.baseFilename, "r", encoding="utf-8") as f:
                        existing_content = f.read()
                except Exception:
                    existing_content = ""

            # Write new message first, then existing content
            with open(self.baseFilename, "w", encoding="utf-8") as f:
                f.write(msg + "\n")
                if existing_content:
                    f.write(existing_content)

            # Check if rotation is needed based on file size
            if self.shouldRollover(record):
                self.doRollover()
        except Exception:
            self.handleError(record)


def setup_logging(level: str = "INFO", log_file: str | None = None):
    # Set up root logger
    root = logging.getLogger()
    root.setLevel(logging.WARNING)  # Set root to WARNING to avoid too much noise
    # Add a filter for correlation ID if exists
    root.addFilter(CorrelationIdFilter())

    # Handle both raw stdout and already-wrapped stdout (from run.py UTF-8 setup)
    if hasattr(sys.stdout, "buffer"):
        # Raw stdout - wrap with UTF-8
        stream = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    else:
        # Already wrapped stdout - use as-is
        stream = sys.stdout

    logger_handler = logging.StreamHandler(stream)
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
