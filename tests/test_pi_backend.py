"""Tests for the pi inference backend (deepseek-v4-flash via opencode-go)."""
import pytest

from src.ai_congress.utils.config_loader import load_config
from src.ai_congress.core.swarm_orchestrator import SwarmOrchestrator


class TestPiBackendConfig:
    def test_pi_defaults(self, monkeypatch):
        """Defaults point at opencode-go / deepseek-v4-flash.

        Hermetic against PI_* overrides: config_loader honours PI_MODEL /
        PI_BASE_URL from the environment, and PI_MODEL is also used by unrelated
        tooling (e.g. the pi coding agent sets PI_MODEL), which would otherwise
        leak into this assertion.
        """
        monkeypatch.delenv("PI_MODEL", raising=False)
        monkeypatch.delenv("PI_BASE_URL", raising=False)
        c = load_config()
        assert c.pi.base_url == "https://opencode.ai/zen/go/v1"
        assert c.pi.model == "deepseek-v4-flash"
        assert "deepseek-v4-flash" in c.pi.preferred_models

    def test_pi_enabled_only_with_key(self, monkeypatch):
        """enabled is False without an API key, True with one."""
        c = load_config()
        monkeypatch.setattr(c.pi, "api_key", "")
        assert not c.pi.enabled
        monkeypatch.setattr(c.pi, "api_key", "sk-test")
        assert c.pi.enabled

    def test_pi_config_applied_from_env(self, monkeypatch):
        """Env overrides populate the pi backend (even without config.yaml)."""
        monkeypatch.setenv("OPENCODE_GO_API_KEY", "sk-env-key")
        monkeypatch.setenv("PI_MODEL", "deepseek-v4-pro")
        c = load_config()
        assert c.pi.api_key == "sk-env-key"
        assert c.pi.model == "deepseek-v4-pro"


class TestPiClientRouting:
    def test_pi_client_created_when_enabled(self, monkeypatch):
        """Swarm builds a pi client only when the backend is enabled."""
        from src.ai_congress.utils.config_loader import OllamaConfig, PiBackendConfig
        from src.ai_congress.core.model_registry import ModelRegistry
        from src.ai_congress.core.voting_engine import VotingEngine

        mr = ModelRegistry(OllamaConfig())
        ve = VotingEngine()

        s = SwarmOrchestrator(
            mr, ve, OllamaConfig(),
            pi_config=PiBackendConfig(api_key="sk-test"),
        )
        assert s.pi_client is not None
        assert s.pi_client.model == "deepseek-v4-flash"

        s2 = SwarmOrchestrator(
            mr, ve, OllamaConfig(),
            pi_config=PiBackendConfig(api_key=""),
        )
        assert s2.pi_client is None

    def test_query_model_routes_pi_backend(self, monkeypatch):
        """query_model resolves the pi client + remote model for backend=pi."""
        from src.ai_congress.utils.config_loader import OllamaConfig, PiBackendConfig
        from src.ai_congress.core.model_registry import ModelRegistry
        from src.ai_congress.core.voting_engine import VotingEngine

        mr = ModelRegistry(OllamaConfig())
        ve = VotingEngine()
        s = SwarmOrchestrator(
            mr, ve, OllamaConfig(),
            pi_config=PiBackendConfig(api_key="sk-test"),
            inference_backend="pi",
        )
        assert s.inference_backend == "pi"

        captured = {}

        async def fake_chat(model, messages, options=None, stream=False):
            captured["model"] = model
            captured["options"] = options
            return {"message": {"content": "hello from pi"}}

        monkeypatch.setattr(s.pi_client, "chat", fake_chat)
        import asyncio
        result = asyncio.run(s.query_model("any-label", "hi", temperature=0.3))
        assert captured["model"] == "deepseek-v4-flash"
        assert result["success"] is True
        assert result["response"] == "hello from pi"
        assert result["backend"] == "pi"
