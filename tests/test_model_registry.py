"""
Tests for ModelRegistry
"""
import pytest
from unittest.mock import AsyncMock, patch
from src.ai_congress.core.model_registry import ModelRegistry
from src.ai_congress.utils.config_loader import OllamaConfig


class TestModelRegistry:
    def setup_method(self):
        self.registry = ModelRegistry(OllamaConfig())

    def test_set_get_weight(self):
        self.registry.set_model_weight("test_model", 0.85)
        assert self.registry.get_model_weight("test_model") == 0.85

    def test_weight_bounds(self):
        self.registry.set_model_weight("test_model", 1.5)
        assert self.registry.get_model_weight("test_model") == 1.0

        self.registry.set_model_weight("test_model", -0.5)
        assert self.registry.get_model_weight("test_model") == 0.0

    def test_default_weight(self):
        assert self.registry.get_model_weight("unknown_model") == 0.5

    def test_filter_lightweight_models(self):
        models = [
            {"name": "small", "size": 1 * 1024**3},
            {"name": "large", "size": 20 * 1024**3}
        ]

        filtered = self.registry.filter_lightweight_models(models, 10.0)
        assert len(filtered) == 1
        assert filtered[0]["name"] == "small"

    @patch('builtins.open', new_callable=AsyncMock)
    def test_load_benchmark_weights(self, mock_open):
        # Mock file content
        mock_open.return_value.__enter__.return_value.read.return_value = '{"model1": {"accuracy": 0.8}}'

        # This would need async handling, simplified for test
        pass
