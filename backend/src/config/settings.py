"""Application configuration settings"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Flask settings
    SECRET_KEY = os.getenv("FLASK_SECRET", "default-secret-key")
    TESTING = os.getenv("TESTING", "false").lower() in {"1", "true", "yes", "on"}
    DEBUG = os.getenv("DEBUG", "false").lower() in {"1", "true", "yes", "on"}
    MAX_CONTENT_LENGTH = int(float(os.getenv("MAX_UPLOAD_MB", "25")) * 1024 * 1024)

    # Model settings
    EMBED_MODEL_NAME = os.getenv(
        "EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
    )
    RERANKER_MODEL_NAME = os.getenv(
        "RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    # Search settings
    USE_HYBRID = os.getenv("USE_HYBRID", "false").lower() in {"1", "true", "yes", "on"}
    USE_RERANKER = os.getenv("USE_RERANKER", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    TOP_K = int(os.getenv("TOP_K", "5"))
    CANDIDATES = int(os.getenv("CANDIDATES", "20"))
    FUSE_ALPHA = float(os.getenv("FUSE_ALPHA", "0.5"))
    MIN_HYBRID = float(os.getenv("MIN_HYBRID", "0.1"))
    AVG_HYBRID = float(os.getenv("AVG_HYBRID", "0.1"))
    MIN_SEM_SIM = float(os.getenv("MIN_SEM_SIM", "0.35"))
    AVG_SEM_SIM = float(os.getenv("AVG_SEM_SIM", "0.2"))
    MIN_RERANK = float(os.getenv("MIN_RERANK", "0.5"))
    AVG_RERANK = float(os.getenv("AVG_RERANK", "0.3"))

    # Document processing
    SENT_TARGET = int(os.getenv("SENT_TARGET", "400"))
    SENT_OVERLAP = int(os.getenv("SENT_OVERLAP", "90"))
    TEXT_MAX = int(os.getenv("TEXT_MAX", "400000"))

    # Chat settings
    CHAT_MAX_TOKENS = int(os.getenv("CHAT_MAX_TOKENS", "200"))

    # OpenAI
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
    OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
    AGENT_MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", "5"))

    # Database
    CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")

    # File upload
    UPLOAD_BASE = os.getenv("UPLOAD_BASE", "uploads")
    MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "25"))
    ALLOWED_EXTENSIONS = os.getenv("ALLOWED_EXTENSIONS", "txt,pdf,docx,md").split(",")
    MIME_TYPES = os.getenv(
        "MIME_TYPES",
        "text/plain,text/markdown,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/vnd.openxmlformats-officedocument",
    ).split(",")
    FOLDER_SHARED = os.getenv("FOLDER_SHARED", "shared")
    DEPT_SPLIT = os.getenv("DEPT_SPLIT", "|")

    # Auth
    SERVICE_AUTH_SECRET = os.getenv("SERVICE_AUTH_SECRET", "")
    SERVICE_AUTH_ISSUER = os.getenv("SERVICE_AUTH_ISSUER", "your_service_name")
    SERVICE_AUTH_AUDIENCE = os.getenv("SERVICE_AUTH_AUDIENCE", "your_service_audience")

    # Organization
    ORG_STRUCTURE_FILE = os.getenv("ORG_STRUCTURE_FILE", "org_structure.json")

    # Rate limiting
    RATELIMIT_STORAGE_URI = os.getenv("RATELIMIT_STORAGE_URI", "memory://")
    DEFAULT_RATE_LIMITS = os.getenv(
        "DEFAULT_RATE_LIMITS", "500 per day,20 per minute"
    ).split(",")

    # MCP Server settings
    USE_MCP = os.getenv("USE_MCP", "false").lower() in {"1", "true", "yes", "on"}
    MCP_TRIGGER_THRESHOLD = float(os.getenv("MCP_TRIGGER_THRESHOLD", "0.6"))
    MCP_SERVER_COMMAND = os.getenv(
        "MCP_SERVER_COMMAND", "npx -y @modelcontextprotocol/server-brave-search"
    )  # e.g., "npx -y @modelcontextprotocol/server-brave-search"
    BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")

    # Self-Reflection / Retrieval Evaluation settings (Week 1)
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
    # These determine when to recommend different actions
    REFLECTION_THRESHOLD_EXCELLENT = float(
        os.getenv("REFLECTION_THRESHOLD_EXCELLENT", "0.85")
    )  # Confidence > 0.85 ? excellent quality
    REFLECTION_THRESHOLD_GOOD = float(
        os.getenv("REFLECTION_THRESHOLD_GOOD", "0.70")
    )  # Confidence > 0.70 ? good quality
    REFLECTION_THRESHOLD_PARTIAL = float(
        os.getenv("REFLECTION_THRESHOLD_PARTIAL", "0.50")
    )  # Confidence > 0.50 ? partial quality, else poor

    # Minimum number of contexts required to proceed with answer
    # If retrieved contexts < this number, recommend refine/external
    REFLECTION_MIN_CONTEXTS = int(os.getenv("REFLECTION_MIN_CONTEXTS", "1"))
    # Average score threshold for contexts (0.0-1.0)
    REFLECTION_AVG_SCORE = float(os.getenv("REFLECTION_AVG_SCORE", "0.6"))
    # Keyword overlap threshold (0.0-1.0)
    REFLECTION_KEYWORD_OVERLAP = float(os.getenv("REFLECTION_KEYWORD_OVERLAP", "0.3"))

    # Automatic action flags (for future integration with other tools)
    # Week 1: These log recommendations only
    # Week 2+: These trigger actual actions when tools are ready
    REFLECTION_AUTO_REFINE = os.getenv("REFLECTION_AUTO_REFINE", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }  # Auto-refine query if quality is poor
    REFLECTION_AUTO_EXTERNAL = os.getenv(
        "REFLECTION_AUTO_EXTERNAL", "false"
    ).lower() in {
        "1",
        "true",
        "yes",
        "on",
    }  # Auto-search external if poor (requires MCP, Week 3)

    # Maximum refinement attempts before giving up
    # Prevents infinite refinement loops
    REFLECTION_MAX_REFINEMENT_ATTEMPTS = int(
        os.getenv("REFLECTION_MAX_REFINEMENT_ATTEMPTS", "3")
    )

    # =============================================================================
    # WEB SEARCH CONFIGURATION (Week 3)
    # =============================================================================
    # Master switch - enables/disables the web search fallback feature
    WEB_SEARCH_ENABLED: bool = os.getenv("WEB_SEARCH_ENABLED", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    # Search provider: "duckduckgo" (free, no API key) or "tavily" (requires API key)
    WEB_SEARCH_PROVIDER: str = os.getenv("WEB_SEARCH_PROVIDER", "duckduckgo")

    # Maximum number of web search results to return
    WEB_SEARCH_MAX_RESULTS: int = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))

    # Tavily API key (optional, only needed if using Tavily provider)
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

    # Langchain, LangGraph Settings for Advanced Agents
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
    USE_LANGGRAPH = os.getenv("USE_LANGGRAPH", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    LANGGRAPH_MAX_ITERATIONS = 20
    LANGGRAPH_TIMEOUT = 120  # seconds

    # Query Routing
    COMPLEXITY_THRESHOLD = float(os.getenv("COMPLEXITY_THRESHOLD", "0.6"))
    ENABLE_QUERY_ROUTING = os.getenv("ENABLE_QUERY_ROUTING", "true").lower() in {
        "1",
        "true",
        "yes",
    }

    # Redis settings
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    REDIS_CACHE_TTL: int = int(os.getenv("REDIS_CACHE_TTL", "3600"))
    REDIS_CACHE_LIMIT: int = int(os.getenv("REDIS_CACHE_LIMIT", "15"))

    # Postgresql Database settings
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    CONVERSATION_MESSAGE_LIMIT: int = int(
        os.getenv("CONVERSATION_MESSAGE_LIMIT", "200")
    )
    CONVERSATION_USER_LIMIT: int = int(os.getenv("CONVERSATION_USER_LIMIT", "50"))


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
