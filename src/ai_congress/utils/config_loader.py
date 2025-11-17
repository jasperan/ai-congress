"""
Configuration Loader - Loads YAML/JSON configuration files
"""
import os
from typing import Dict, Any
import yaml
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    timeout: int = 120
    max_retries: int = 3


class SwarmConfig(BaseModel):
    default_mode: str = "multi_model"
    multi_request: Dict[str, Any] = Field(default_factory=lambda: {"temperatures": [0.3, 0.7, 1.0, 1.2]})
    max_concurrent_requests: int = 10
    request_timeout: int = 60


class VotingConfig(BaseModel):
    default_algorithm: str = "weighted_majority"
    consensus_threshold: float = 0.6
    enable_clustering: bool = True


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


class Config(BaseModel):
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    swarm: SwarmConfig = Field(default_factory=SwarmConfig)
    voting: VotingConfig = Field(default_factory=VotingConfig)
    models: ModelWeights = Field(default_factory=ModelWeights)
    api: APIConfig = Field(default_factory=APIConfig)
    cli: CLIConfig = Field(default_factory=CLIConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


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
