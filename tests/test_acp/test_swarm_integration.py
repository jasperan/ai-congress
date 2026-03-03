import pytest
from unittest.mock import MagicMock
from src.ai_congress.core.swarm_orchestrator import SwarmOrchestrator
from src.ai_congress.core.model_registry import ModelRegistry
from src.ai_congress.core.voting_engine import VotingEngine
from src.ai_congress.utils.config_loader import OllamaConfig


@pytest.fixture
def mock_deps():
    mock_registry = MagicMock(spec=ModelRegistry)
    mock_voting = MagicMock(spec=VotingEngine)
    mock_config = MagicMock(spec=OllamaConfig)
    mock_config.base_url = "http://localhost:11434"
    mock_config.timeout = 30
    mock_config.max_retries = 3
    return mock_registry, mock_voting, mock_config


def test_orchestrator_has_acp_components(mock_deps):
    model_reg, voting, ollama_cfg = mock_deps
    orch = SwarmOrchestrator(model_reg, voting, ollama_cfg)
    assert hasattr(orch, "registry")
    assert hasattr(orch, "coordination")
    assert hasattr(orch, "message_bus")
    assert hasattr(orch, "personality_loader")


def test_orchestrator_registers_models(mock_deps):
    model_reg, voting, ollama_cfg = mock_deps
    orch = SwarmOrchestrator(model_reg, voting, ollama_cfg)
    orch.register_model_agents(["phi3:3.8b", "mistral:7b"])
    agents = orch.registry.get_active()
    assert len(agents) == 2


def test_coordination_level_from_config(mock_deps):
    model_reg, voting, ollama_cfg = mock_deps
    orch = SwarmOrchestrator(model_reg, voting, ollama_cfg, coordination_level="moderate")
    assert orch.coordination.level == "moderate"


def test_coordination_level_default(mock_deps):
    model_reg, voting, ollama_cfg = mock_deps
    orch = SwarmOrchestrator(model_reg, voting, ollama_cfg)
    assert orch.coordination.level == "moderate"


def test_coordination_level_chatty(mock_deps):
    model_reg, voting, ollama_cfg = mock_deps
    orch = SwarmOrchestrator(model_reg, voting, ollama_cfg, coordination_level="chatty")
    assert orch.coordination.level == "chatty"


def test_register_model_agents_creates_bus_entries(mock_deps):
    model_reg, voting, ollama_cfg = mock_deps
    orch = SwarmOrchestrator(model_reg, voting, ollama_cfg)
    orch.register_model_agents(["phi3:3.8b"])
    # Agent should be registered in message bus
    assert "phi3:3.8b" in orch.message_bus._agents


def test_register_model_agents_with_personality(mock_deps):
    model_reg, voting, ollama_cfg = mock_deps
    orch = SwarmOrchestrator(model_reg, voting, ollama_cfg)
    orch.register_model_agents(["phi3:3.8b"])
    agents = orch.registry.get_active()
    assert len(agents) == 1
    agent = agents[0]
    assert agent.name == "phi3:3.8b"
    assert agent.role == "voter"
    assert "reasoning" in agent.capabilities
    assert "voting" in agent.capabilities
