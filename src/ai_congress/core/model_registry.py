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
        """List all available Ollama models.

        Also bootstraps ``weights`` from the live catalog (3.5.1): every
        model actually present gets at least the neutral 0.5 base weight,
        and stale benchmark keys for models no longer installed are purged
        so learning starts on real models instead of ghosts.
        """
        try:
            models_response = await self.ollama_client.list_models()
            models = []
            live_names: set[str] = set()

            for model in models_response:
                # ollama>=0.6 dumps the id under 'model' (no computed 'name')
                model_name = model.get('name') or model.get('model')
                if not model_name:
                    continue
                live_names.add(model_name)
                model_info = {
                    'name': model_name,
                    'size': model.get('size', 0),
                    'modified_at': model.get('modified_at'),
                    'digest': model.get('digest')
                }
                models.append(model_info)
                self.models_cache[model_name] = model_info
                # Neutral bootstrap weight for any live model without one
                self.weights.setdefault(model_name, 0.5)

            # Drop weights for models no longer in the catalog — only when the
            # catalog was actually discovered (don't wipe on an Ollama outage).
            if live_names:
                self.weights = {
                    name: w for name, w in self.weights.items() if name in live_names
                }

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
        """Load model weights from benchmark file.

        Only weights for models present in the live catalog (``models_cache``)
        are applied — stale benchmark keys for uninstalled models are ignored
        so they cannot dominate ranking (3.5.1).
        """
        import json
        try:
            with open(benchmark_file, 'r') as f:
                benchmarks = json.load(f)

            applied = 0
            skipped = 0
            for model_name, data in benchmarks.items():
                if model_name not in self.models_cache:
                    skipped += 1
                    continue
                # Normalize accuracy to 0-1 range
                accuracy = data.get('accuracy', 0.5)
                self.set_model_weight(model_name, accuracy)
                applied += 1

            logger.info(
                f"Loaded benchmark weights: {applied} applied, {skipped} skipped (not installed)"
            )

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
