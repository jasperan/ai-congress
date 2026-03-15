"""
Configuration Loader - Loads YAML/JSON configuration files
"""
import os
from typing import Dict, Any, Optional
import yaml
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    timeout: int = 120
    max_retries: int = 3


class OpenAIConfig(BaseModel):
    """Config for OpenAI-compatible API backends (e.g. OCA via LiteLLM)."""
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    timeout: int = 120
    max_retries: int = 3


class SwarmConfig(BaseModel):
    default_mode: str = "multi_model"
    multi_request: Dict[str, Any] = Field(default_factory=lambda: {"temperatures": [0.3, 0.7, 1.0, 1.2]})
    hybrid: Dict[str, Any] = Field(default_factory=lambda: {"temperatures": [0.5, 0.9], "top_models": 8, "streaming": False})
    max_concurrent_requests: int = 10
    request_timeout: int = 60


class DebateVotingConfig(BaseModel):
    enabled: bool = True
    max_rounds: int = 3
    temp_schedule: list = Field(default_factory=lambda: [0.9, 0.5, 0.2])
    conviction_bonus: float = 1.2


class VotingConfig(BaseModel):
    default_algorithm: str = "weighted_majority"
    consensus_threshold: float = 0.6
    enable_clustering: bool = True
    summarizer_model: str = "phi3:3.8b"
    semantic_confidence_threshold: float = 0.6
    mode: str = "classic"
    debate: DebateVotingConfig = Field(default_factory=DebateVotingConfig)


class ModelWeights(BaseModel):
    preferred: list = Field(default_factory=lambda: ["phi3:3.8b", "mistral:7b", "llama3.2:3b"])
    weights: Dict[str, float] = Field(default_factory=dict)


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    workers: int = 4
    cors_origins: list = Field(default_factory=lambda: ["http://localhost:3000", "http://localhost:5173"])


class CLIConfig(BaseModel):
    default_verbosity: int = 2
    use_colors: bool = True
    panel_style: str = "bold cyan"
    save_history: bool = True
    history_file: str = "~/.ai_congress_history"


class DatabaseConfig(BaseModel):
    url: str = "sqlite+aiosqlite:///./ai_congress.db"
    echo: bool = False


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "logs/ai_congress.log"
    rich: bool = True
    verbosity: int = 2  # 0=minimal, 1=normal, 2=verbose, 3=debug


class AgentsConfig(BaseModel):
    base_model: str = "mistral:7b"


class OracleDBConfig(BaseModel):
    user: str = "admin"
    password: str = "your_password_here"
    dsn: str = "localhost:1521/FREEPDB1"
    use_tls: bool = True
    vector_table: str = "document_vectors"
    embedding_dimension: int = 384
    enable_cache: bool = True
    cache_size: int = 1000
    cache_ttl: int = 3600  # seconds
    batch_size: int = 100


class RAGConfig(BaseModel):
    enabled: bool = True
    auto_on_upload: bool = True
    top_k: int = 10
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_size: int = 100
    adaptive_chunking: bool = True  # Use semantic-aware chunking


class VoiceConfig(BaseModel):
    model: str = "base"
    language: str = "en"
    device: str = "cpu"
    compute_type: str = "int8"


class WebSearchConfig(BaseModel):
    default_engine: str = "duckduckgo"  # duckduckgo, searxng, yacy
    max_results: int = 5
    timeout: int = 10
    # Optional SearXNG instance URL (self-hosted)
    searxng_url: str = ""  # e.g., "http://localhost:8080"
    # Optional Yacy instance URL (self-hosted)
    yacy_url: str = ""  # e.g., "http://localhost:8090"


class DocumentExtractionConfig(BaseModel):
    use_advanced_extractors: bool = False
    # Apache Tika server URL (optional, docker: apache/tika)
    tika_url: str = ""  # e.g., "http://localhost:9998"
    # Docling server URL (optional, docker: quay.io/docling-project/docling-serve)
    docling_url: str = ""  # e.g., "http://localhost:5001"
    prefer: str = "docling"  # preferred extractor: docling or tika


class DataLakeConfig(BaseModel):
    user: str = "AI_CONGRESS"
    password: str = "AiCongress2026"
    host: str = "localhost"
    port: int = 1521
    service: str = "FREEPDB1"
    pool_min: int = 2
    pool_max: int = 5


class ImageGenConfig(BaseModel):
    model: str = "stable-diffusion"
    default_steps: int = 30
    width: int = 512
    height: int = 512
    output_dir: str = "static/generated_images"


class Config(BaseModel):
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    swarm: SwarmConfig = Field(default_factory=SwarmConfig)
    voting: VotingConfig = Field(default_factory=VotingConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    models: ModelWeights = Field(default_factory=ModelWeights)
    api: APIConfig = Field(default_factory=APIConfig)
    cli: CLIConfig = Field(default_factory=CLIConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    oracle_db: OracleDBConfig = Field(default_factory=OracleDBConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    web_search: WebSearchConfig = Field(default_factory=WebSearchConfig)
    image_gen: ImageGenConfig = Field(default_factory=ImageGenConfig)
    document_extraction: DocumentExtractionConfig = Field(default_factory=DocumentExtractionConfig)
    datalake: DataLakeConfig = Field(default_factory=DataLakeConfig)


def _apply_codex_config(config: Config) -> None:
    """Read ~/.codex/config.toml and populate config.openai from it."""
    codex_path = os.path.expanduser("~/.codex/config.toml")
    if not os.path.exists(codex_path):
        return

    try:
        import tomllib
    except ModuleNotFoundError:
        try:
            import tomli as tomllib  # Python <3.11 fallback
        except ModuleNotFoundError:
            logger.debug("No TOML parser available; skipping codex config")
            return

    try:
        with open(codex_path, "rb") as f:
            codex = tomllib.load(f)

        # Find the active provider
        provider_name = codex.get("model_provider", "")
        providers = codex.get("model_providers", {})
        provider = providers.get(provider_name, {})

        if provider.get("base_url"):
            # The OpenAI SDK appends /chat/completions to base_url,
            # so for LiteLLM proxies we need the URL to end with /v1.
            base = provider["base_url"].rstrip("/")
            if not base.endswith("/v1"):
                base += "/v1"
            config.openai.base_url = base

        # Model: use top-level model (the active profile's model)
        if codex.get("model"):
            config.openai.model = codex["model"]
        elif provider.get("model"):
            config.openai.model = provider["model"]

        # API key: check env vars that codex CLI typically uses
        for env_key in ("OPENAI_API_KEY", "OCA_API_KEY", "CODEX_API_KEY"):
            val = os.getenv(env_key, "")
            if val:
                config.openai.api_key = val
                break

        if config.openai.base_url:
            logger.info(f"Loaded OpenAI config from codex: {config.openai.base_url} / {config.openai.model}")

    except Exception as e:
        logger.debug(f"Could not read codex config: {e}")


def load_config(config_file: str = "config/config.yaml") -> Config:
    """Load configuration from YAML file"""
    try:
        if not os.path.exists(config_file):
            logger.warning(f"Config file {config_file} not found, using defaults")
            return Config()

        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)

        config = Config(**config_data)

        # Environment variable overrides
        if os.getenv("OLLAMA_BASE_URL"):
            config.ollama.base_url = os.getenv("OLLAMA_BASE_URL")
            logger.info(f"OLLAMA_BASE_URL override applied: {config.ollama.base_url}")

        # Auto-populate OpenAI config from ~/.codex/config.toml if not set in YAML
        if not config.openai.base_url:
            _apply_codex_config(config)

        # Env-var overrides for OpenAI
        if os.getenv("OPENAI_BASE_URL"):
            config.openai.base_url = os.getenv("OPENAI_BASE_URL")
        if os.getenv("OPENAI_API_KEY"):
            config.openai.api_key = os.getenv("OPENAI_API_KEY")
        if os.getenv("OPENAI_MODEL"):
            config.openai.model = os.getenv("OPENAI_MODEL")

        logger.info(f"Loaded configuration from {config_file}")
        return config

    except Exception as e:
        logger.error(f"Error loading config from {config_file}: {e}")
        logger.warning("Using default configuration")
        return Config()


def save_config(config: Config, config_file: str = "config/config.yaml") -> bool:
    """Save configuration to YAML file"""
    try:
        os.makedirs(os.path.dirname(config_file), exist_ok=True)

        config_dict = config.model_dump()
        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

        logger.info(f"Saved configuration to {config_file}")
        return True

    except Exception as e:
        logger.error(f"Error saving config to {config_file}: {e}")
        return False
