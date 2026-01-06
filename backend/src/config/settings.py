"""Application configuration settings"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ==============================================================================
    # Python logging configuration
    # ==============================================================================
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = os.getenv(
        "LOG_FORMAT",
        "%(asctime)s - %(correlation_id)s - %(funcName)s - %(lineno)d - %(message)s",
    )
    LOG_PATH: str = os.getenv("LOG_PATH", "logs/app.log")

    # ==============================================================================
    # Flask Application Settings
    # ==============================================================================
    SECRET_KEY = os.getenv("FLASK_SECRET", "default-secret-key")
    TESTING = os.getenv("TESTING", "false").lower() in {"1", "true", "yes", "on"}
    DEBUG = os.getenv("DEBUG", "false").lower() in {"1", "true", "yes", "on"}
    MAX_CONTENT_LENGTH = int(float(os.getenv("MAX_UPLOAD_MB", "25")) * 1024 * 1024)

    # ==============================================================================
    # OpenAI API Settings
    # ==============================================================================
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
    OPENAI_SIMPLE_MODEL = os.getenv(
        "OPENAI_SIMPLE_MODEL", "gpt-4o-mini"
    )  # Fast and cheap for simple, classification questions.
    OPENAI_VISION_MODEL = os.getenv(
        "OPENAI_VISION_MODEL", "gpt-4o-mini"
    )  # Vision-capable model for image analysis.
    OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
    OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
    AGENT_MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", "20"))

    # ==============================================================================
    # LangChain & LangSmith Tracing
    # ==============================================================================
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "agentic-rag-chatbot")
    LANGCHAIN_ENDPOINT = os.getenv(
        "LANGCHAIN_ENDPOINT", "https://eu.api.smith.langchain.com"
    )
    LANGCHAIN_EMBEDDING_MODEL = os.getenv(
        "LANGCHAIN_EMBEDDING_MODEL", "text-embedding-3-small"
    )

    # ==============================================================================
    # LangGraph Configuration
    # ==============================================================================
    USE_LANGGRAPH = os.getenv("USE_LANGGRAPH", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    LANGGRAPH_MAX_ITERATIONS = 20
    LANGGRAPH_TIMEOUT = 120  # seconds
    CHECKPOINT_ENABLED = os.getenv("CHECKPOINT_ENABLED", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    # ==============================================================================
    # Embedding & Reranking Models
    # ==============================================================================
    # Embedding provider: "local" (SentenceTransformer) or "openai"
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")

    # For local provider: SentenceTransformer model name
    EMBEDDING_MODEL_NAME = os.getenv(
        "EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-base"
    )
    # For openai provider: OpenAI embedding model
    # Options: text-embedding-3-small (1536-dim), text-embedding-3-large (3072-dim)
    OPENAI_EMBEDDING_MODEL = os.getenv(
        "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
    )
    RERANKER_MODEL_NAME = os.getenv(
        # cross-encoder/ms-marco-MiniLM-L-6-v2, BAAI/bge-reranker-base, BAAI/bge-reranker-v2-m3
        "RERANKER_MODEL_NAME",
        "BAAI/bge-reranker-v2-m3",
    )

    # ==============================================================================
    # Vector Database (ChromaDB)
    # ==============================================================================
    CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")

    # ==============================================================================
    # PostgreSQL Database (Conversations & Checkpoints)
    # ==============================================================================
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    CHECKPOINT_POSTGRES_DATABASE_URL: str = os.getenv(
        "CHECKPOINT_POSTGRES_DATABASE_URL", ""
    )
    CONVERSATION_MESSAGE_LIMIT: int = int(
        os.getenv("CONVERSATION_MESSAGE_LIMIT", "200")
    )
    CONVERSATION_USER_LIMIT: int = int(os.getenv("CONVERSATION_USER_LIMIT", "50"))

    # ==============================================================================
    # Redis Cache
    # ==============================================================================
    REDIS_ENABLED: bool = os.getenv("REDIS_ENABLED", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    REDIS_CACHE_TTL: int = int(os.getenv("REDIS_CACHE_TTL", "3600"))
    REDIS_CACHE_LIMIT: int = int(os.getenv("REDIS_CACHE_LIMIT", "15"))

    # Slack conversation mapping TTL (how long to remember channel → conversation_id mapping)
    SLACK_CONV_TTL: int = int(os.getenv("SLACK_CONV_TTL", "86400"))  # 24 hours

    # ==============================================================================
    # Retrieval & Search Settings
    # ==============================================================================
    USE_HYBRID = os.getenv("USE_HYBRID", "false").lower() in {"1", "true", "yes", "on"}
    USE_RERANKER = os.getenv("USE_RERANKER", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    TOP_K = int(os.getenv("TOP_K", "8"))
    CANDIDATES = int(os.getenv("CANDIDATES", "50"))
    FUSE_ALPHA = float(os.getenv("FUSE_ALPHA", "0.5"))

    # Quality thresholds for hybrid search and reranking
    # Note: These thresholds assume normalized scores in [0,1] range
    MIN_HYBRID = float(os.getenv("MIN_HYBRID", "0.15"))
    AVG_HYBRID = float(os.getenv("AVG_HYBRID", "0.1"))
    MIN_SEM_SIM = float(os.getenv("MIN_SEM_SIM", "0.35"))
    AVG_SEM_SIM = float(os.getenv("AVG_SEM_SIM", "0.25"))
    MIN_RERANK = float(os.getenv("MIN_RERANK", "0.45"))
    AVG_RERANK = float(os.getenv("AVG_RERANK", "0.35"))
    ENFORCE_CITATIONS = os.getenv("ENFORCE_CITATIONS", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    # ==============================================================================
    # Document Processing & Chunking
    # ==============================================================================
    SENT_TARGET = int(os.getenv("SENT_TARGET", "1600"))
    SENT_OVERLAP = int(os.getenv("SENT_OVERLAP", "250"))
    TEXT_MAX = int(os.getenv("TEXT_MAX", "-1"))

    # ==============================================================================
    # Chat Settings
    # ==============================================================================
    # Note: For reasoning models (o1, gpt-5.2), this needs to be higher because
    # reasoning tokens count against this limit. Recommended: 8000-16000 for complex queries.
    CHAT_MAX_TOKENS = int(os.getenv("CHAT_MAX_TOKENS", "16000"))
    ONE_HISTORY_MAX_TOKENS = int(os.getenv("ONE_HISTORY_MAX_TOKENS", "1500"))

    # ==============================================================================
    # File Upload Configuration
    # ==============================================================================
    UPLOAD_BASE = os.getenv("UPLOAD_BASE", "uploads")
    MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "25"))
    ALLOWED_EXTENSIONS = os.getenv("ALLOWED_EXTENSIONS", "txt,pdf,docx,md").split(",")
    MIME_TYPES = os.getenv(
        "MIME_TYPES",
        "text/plain,text/markdown,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/vnd.openxmlformats-officedocument",
    ).split(",")
    FOLDER_SHARED = os.getenv("FOLDER_SHARED", "shared")
    DEPT_SPLIT = os.getenv("DEPT_SPLIT", "|")

    # ==============================================================================
    # Authentication & Authorization
    # ==============================================================================
    SERVICE_AUTH_SECRET = os.getenv("SERVICE_AUTH_SECRET", "")
    SERVICE_AUTH_ISSUER = os.getenv("SERVICE_AUTH_ISSUER", "your_service_name")
    SERVICE_AUTH_AUDIENCE = os.getenv("SERVICE_AUTH_AUDIENCE", "your_service_audience")

    # ==============================================================================
    # Organization Structure
    # ==============================================================================
    ORG_STRUCTURE_FILE = os.getenv("ORG_STRUCTURE_FILE", "org_structure.json")

    # ==============================================================================
    # Rate Limiting
    # ==============================================================================
    RATELIMIT_STORAGE_URI = os.getenv("RATELIMIT_STORAGE_URI", "memory://")
    DEFAULT_RATE_LIMITS = os.getenv(
        "DEFAULT_RATE_LIMITS", "500 per day,20 per minute"
    ).split(",")

    # ==============================================================================
    # MCP (Model Context Protocol) Settings
    # ==============================================================================
    USE_MCP = os.getenv("USE_MCP", "false").lower() in {"1", "true", "yes", "on"}
    MCP_TRIGGER_THRESHOLD = float(os.getenv("MCP_TRIGGER_THRESHOLD", "0.6"))
    MCP_SERVER_COMMAND = os.getenv(
        "MCP_SERVER_COMMAND", "npx -y @modelcontextprotocol/server-brave-search"
    )
    BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")

    # ==============================================================================
    # Web Search Configuration
    # ==============================================================================
    WEB_SEARCH_ENABLED: bool = os.getenv("WEB_SEARCH_ENABLED", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    WEB_SEARCH_PROVIDER: str = os.getenv("WEB_SEARCH_PROVIDER", "duckduckgo")
    WEB_SEARCH_MAX_RESULTS: int = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

    # ==============================================================================
    # Debug & Logging
    # ==============================================================================
    # Show detailed score calculations in logs (for debugging retrieval/evaluation)
    SHOW_SCORES = os.getenv("SHOW_SCORES", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    # ==============================================================================
    # Self-Reflection / Retrieval Evaluation
    # ==============================================================================
    # Master switch - enables/disables the entire self-reflection feature
    USE_SELF_REFLECTION = os.getenv("USE_SELF_REFLECTION", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    # Performance mode: "fast", "balanced", "thorough"
    # - "fast": Heuristics only, no LLM (50ms, cheap)
    # - "balanced": Light LLM check (500ms, moderate) - RECOMMENDED
    # - "thorough": Full LLM evaluation (2s, expensive)
    REFLECTION_MODE = os.getenv("REFLECTION_MODE", "balanced")

    # Quality thresholds (0.0-1.0)
    REFLECTION_THRESHOLD_EXCELLENT = float(
        os.getenv("REFLECTION_THRESHOLD_EXCELLENT", "0.85")
    )
    REFLECTION_THRESHOLD_GOOD = float(os.getenv("REFLECTION_THRESHOLD_GOOD", "0.70"))
    REFLECTION_THRESHOLD_PARTIAL = float(
        os.getenv("REFLECTION_THRESHOLD_PARTIAL", "0.50")
    )

    # Minimum number of contexts required to proceed with answer
    REFLECTION_MIN_CONTEXTS = int(os.getenv("REFLECTION_MIN_CONTEXTS", "1"))
    REFLECTION_AVG_SCORE = float(os.getenv("REFLECTION_AVG_SCORE", "0.6"))
    REFLECTION_KEYWORD_OVERLAP = float(os.getenv("REFLECTION_KEYWORD_OVERLAP", "0.3"))

    # Automatic action flags
    REFLECTION_AUTO_REFINE = os.getenv("REFLECTION_AUTO_REFINE", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    REFLECTION_AUTO_EXTERNAL = os.getenv(
        "REFLECTION_AUTO_EXTERNAL", "false"
    ).lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    # Maximum refinement attempts (prevents infinite loops)
    REFLECTION_MAX_REFINEMENT_ATTEMPTS = int(
        os.getenv("REFLECTION_MAX_REFINEMENT_ATTEMPTS", "3")
    )

    # ==============================================================================
    # Query Routing
    # ==============================================================================
    COMPLEXITY_THRESHOLD = float(os.getenv("COMPLEXITY_THRESHOLD", "0.6"))
    ENABLE_QUERY_ROUTING = os.getenv("ENABLE_QUERY_ROUTING", "true").lower() in {
        "1",
        "true",
        "yes",
    }

    # ==============================================================================
    # File Discovery Settings
    # ==============================================================================
    FILE_DISCOVERY_INDEXED_LIMIT = int(os.getenv("FILE_DISCOVERY_INDEXED_LIMIT", "200"))
    FILE_DISCOVERY_CONVERSATION_LIMIT = int(
        os.getenv("FILE_DISCOVERY_CONVERSATION_LIMIT", "20")
    )

    # ==============================================================================
    # Email Settings
    # ==============================================================================
    SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.outlook.com")
    SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USER = os.getenv("SMTP_USER", "myname@outlook.com")
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "12345!A")

    # ==============================================================================
    # Download Settings
    # ==============================================================================
    DOWNLOAD_BASE = os.getenv("DOWNLOAD_BASE", "downloads")
    MAX_DOWNLOAD_SIZE_MB = float(os.getenv("MAX_DOWNLOAD_SIZE_MB", "100"))
    DOWNLOAD_TIMEOUT = int(os.getenv("DOWNLOAD_TIMEOUT", "30"))  # seconds

    # ==============================================================================
    # Browser Automation (Browser-Use) Settings
    # ==============================================================================
    # Enable browser automation for complex downloads (login-required, JS-rendered)
    BROWSER_USE_ENABLED = os.getenv("BROWSER_USE_ENABLED", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    # Run browser in headless mode (no visible window)
    BROWSER_HEADLESS = os.getenv("BROWSER_HEADLESS", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    # Browser automation timeout (longer than HTTP due to page navigation)
    BROWSER_TIMEOUT = int(os.getenv("BROWSER_TIMEOUT", "60"))  # seconds
    # Max steps for browser agent (each step = one action: click, type, navigate)
    BROWSER_MAX_STEPS = int(os.getenv("BROWSER_MAX_STEPS", "15"))
    # Test credentials for development (replace with secure credential management later)
    BROWSER_TEST_USERNAME = os.getenv("BROWSER_TEST_USERNAME", "")
    BROWSER_TEST_PASSWORD = os.getenv("BROWSER_TEST_PASSWORD", "")
    # Save browser agent conversation logs for debugging (empty = disabled)
    BROWSER_LOG_PATH = os.getenv("BROWSER_LOG_PATH", "logs/browser_agent.log")

    # ==============================================================================
    # Backend API Settings
    # ==============================================================================
    BACKEND_API_TIMEOUT = int(os.getenv("BACKEND_API_TIMEOUT", "120"))

    # ==============================================================================
    # Bot General Configuration
    # ==============================================================================
    BOT_BACKEND_URL = os.getenv("BOT_BACKEND_URL", "http://localhost:5001")

    # ==============================================================================
    # Slack Bot Configuration
    # ==============================================================================
    # Enable/disable Slack integration
    SLACK_ENABLED = os.getenv("SLACK_ENABLED", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    # Slack Bot User OAuth Token (starts with xoxb-)
    # Get from: Slack App → OAuth & Permissions → Bot User OAuth Token
    SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")

    # Slack Signing Secret (for verifying webhook requests)
    # Get from: Slack App → Basic Information → Signing Secret
    SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET", "")

    # Default department for channels not in SLACK_CHANNEL_TO_DEPT mapping
    SLACK_DEFAULT_DEPT = os.getenv("SLACK_DEFAULT_DEPT", "MYHB|software|ml")

    # Channel → Department mapping (JSON string)
    # Example: {"C0123456789": "engineering", "C9876543210": "sales"}
    # Leave empty to use SLACK_DEFAULT_DEPT for all channels
    SLACK_CHANNEL_TO_DEPT: dict = {}  # TODO: Parse from env if needed

    # Workspace → Department mapping (for Enterprise Grid with multiple workspaces)
    # Example: {"T0123456789": "company_a", "T9876543210": "company_b"}
    SLACK_WORKSPACE_TO_DEPT: dict = {}  # TODO: Parse from env if needed


class DevelopmentConfig(Config):
    """Development configuration"""

    DEBUG = True


class TestingConfig(Config):
    """Testing configuration"""

    TESTING = True
    MAX_CONTENT_LENGTH = int(float(os.getenv("MAX_UPLOAD_MB", "25")) * 1024 * 1024)


class ProductionConfig(Config):
    """Production configuration"""

    pass


# Configuration dictionary
config = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}


def get_config(env=None):
    """Get configuration based on environment"""
    if env is None:
        env = os.getenv("FLASK_ENV", "development")
    return config.get(env, config["default"])
