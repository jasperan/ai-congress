"""
Model Registry - Manages Ollama models and their performance metrics
"""
import asyncio
from typing import List, Dict, Optional
from .ollama_client import OllamaClient
from ..utils.config_loader import OllamaConfig
import logging

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for managing Ollama models and their performance data"""

    def __init__(self, ollama_config: OllamaConfig):
        self.ollama_client = OllamaClient(
            base_url=ollama_config.base_url,
            timeout=ollama_config.timeout,
            max_retries=ollama_config.max_retries
        )
        self.models_cache: Dict[str, Dict] = {}
        self.weights: Dict[str, float] = {}

    async def list_available_models(self) -> List[Dict]:
        """List all available Ollama models"""
        try:
            models_response = await self.ollama_client.list_models()
            models = []

            for model in models_response:
                model_info = {
                    'name': model['name'],
                    'size': model.get('size', 0),
                    'modified_at': model.get('modified_at'),
                    'digest': model.get('digest')
                }
                models.append(model_info)
                self.models_cache[model['name']] = model_info

            logger.info(f"Found {len(models)} available models")
            return models

        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

    async def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get detailed information about a specific model"""
        try:
            if model_name in self.models_cache:
                return self.models_cache[model_name]

            # Fetch from Ollama
            await self.list_available_models()
            return self.models_cache.get(model_name)

        except Exception as e:
            logger.error(f"Error getting model info for {model_name}: {e}")
            return None

    def set_model_weight(self, model_name: str, weight: float):
        """Set performance weight for a model (0.0 - 1.0)"""
        self.weights[model_name] = max(0.0, min(1.0, weight))
        logger.info(f"Set weight for {model_name}: {weight:.2f}")

    def get_model_weight(self, model_name: str) -> float:
        """Get performance weight for a model"""
        return self.weights.get(model_name, 0.5)  # Default to 0.5

    async def load_benchmark_weights(self, benchmark_file: str):
        """Load model weights from benchmark file"""
        import json
        try:
            with open(benchmark_file, 'r') as f:
                benchmarks = json.load(f)

            for model_name, data in benchmarks.items():
                # Normalize accuracy to 0-1 range
                accuracy = data.get('accuracy', 0.5)
                self.set_model_weight(model_name, accuracy)

            logger.info(f"Loaded benchmark weights for {len(self.weights)} models")

        except Exception as e:
            logger.warning(f"Could not load benchmark file: {e}")

    async def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama library"""
        try:
            logger.info(f"Pulling model: {model_name}")

            success = await self.ollama_client.pull_model(model_name)
            if success:
                logger.info(f"Model {model_name} pulled successfully")
                await self.list_available_models()  # Refresh cache
                return True

            return False

        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False

    def filter_lightweight_models(self, models: List[Dict], max_size_gb: float = 10.0) -> List[Dict]:
        """Filter models by size to get lightweight options"""
        max_size_bytes = max_size_gb * 1024 * 1024 * 1024
        return [m for m in models if m.get('size', 0) <= max_size_bytes]

    def get_top_models(self, n: int = 5) -> List[str]:
        """Get top N models by weight"""
        sorted_models = sorted(self.weights.items(), key=lambda x: x[1], reverse=True)
        return [model for model, _ in sorted_models[:n]]
