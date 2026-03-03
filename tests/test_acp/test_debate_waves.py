import pytest
from unittest.mock import AsyncMock, MagicMock
from src.ai_congress.core.swarm_orchestrator import SwarmOrchestrator
from src.ai_congress.core.model_registry import ModelRegistry
from src.ai_congress.core.voting_engine import VotingEngine
from src.ai_congress.utils.config_loader import OllamaConfig


@pytest.fixture
def mock_deps():
    mock_registry = MagicMock(spec=ModelRegistry)
    mock_registry.get_model_weight.return_value = 0.5
    mock_voting = MagicMock(spec=VotingEngine)
    mock_config = MagicMock(spec=OllamaConfig)
    mock_config.base_url = "http://localhost:11434"
    mock_config.timeout = 30
    mock_config.max_retries = 3
    return mock_registry, mock_voting, mock_config


@pytest.mark.asyncio
async def test_debate_waves_at_chatty_level(mock_deps):
    model_reg, voting, ollama_cfg = mock_deps
    orch = SwarmOrchestrator(model_reg, voting, ollama_cfg, coordination_level="chatty")

    responses = iter([
        {"response": "AI is great", "model": "phi3:3.8b", "success": True},
        {"response": "AI is complex", "model": "mistral:7b", "success": True},
        {"response": "AI is great and complex", "model": "phi3:3.8b", "success": True},
        {"response": "AI is great and complex", "model": "mistral:7b", "success": True},
        {"response": "AI is transformative", "model": "phi3:3.8b", "success": True},
        {"response": "AI is transformative", "model": "mistral:7b", "success": True},
    ])
    orch.query_model = AsyncMock(side_effect=lambda *a, **kw: next(responses))

    result = await orch.multi_model_swarm_with_debate(
        prompt="What is AI?",
        models=["phi3:3.8b", "mistral:7b"],
    )
    assert result["final_answer"] is not None
    assert result["debate_waves"] == 3  # chatty = 3 waves
    assert orch.query_model.call_count == 6


@pytest.mark.asyncio
async def test_no_debate_at_none_level(mock_deps):
    model_reg, voting, ollama_cfg = mock_deps
    orch = SwarmOrchestrator(model_reg, voting, ollama_cfg, coordination_level="none")

    orch.query_model = AsyncMock(return_value={
        "response": "AI is great", "model": "phi3:3.8b", "success": True
    })

    result = await orch.multi_model_swarm_with_debate(
        prompt="What is AI?",
        models=["phi3:3.8b", "mistral:7b"],
    )
    assert result["debate_waves"] == 1


@pytest.mark.asyncio
async def test_moderate_level_has_two_waves(mock_deps):
    model_reg, voting, ollama_cfg = mock_deps
    orch = SwarmOrchestrator(model_reg, voting, ollama_cfg, coordination_level="moderate")

    responses = iter([
        {"response": "AI is great", "model": "phi3:3.8b", "success": True},
        {"response": "AI is complex", "model": "mistral:7b", "success": True},
        {"response": "AI is great and complex", "model": "phi3:3.8b", "success": True},
        {"response": "AI is great and complex", "model": "mistral:7b", "success": True},
    ])
    orch.query_model = AsyncMock(side_effect=lambda *a, **kw: next(responses))

    result = await orch.multi_model_swarm_with_debate(
        prompt="What is AI?",
        models=["phi3:3.8b", "mistral:7b"],
    )
    assert result["debate_waves"] == 2
    assert result["coordination_level"] == "moderate"
    assert orch.query_model.call_count == 4


@pytest.mark.asyncio
async def test_debate_result_structure(mock_deps):
    model_reg, voting, ollama_cfg = mock_deps
    orch = SwarmOrchestrator(model_reg, voting, ollama_cfg, coordination_level="none")

    orch.query_model = AsyncMock(return_value={
        "response": "AI is great", "model": "phi3:3.8b", "success": True
    })

    result = await orch.multi_model_swarm_with_debate(
        prompt="What is AI?",
        models=["phi3:3.8b"],
    )
    assert "responses" in result
    assert "final_answer" in result
    assert "confidence" in result
    assert "vote_breakdown" in result
    assert "models_used" in result
    assert "debate_waves" in result
    assert "coordination_level" in result


@pytest.mark.asyncio
async def test_debate_handles_failed_initial_responses(mock_deps):
    model_reg, voting, ollama_cfg = mock_deps
    orch = SwarmOrchestrator(model_reg, voting, ollama_cfg, coordination_level="moderate")

    responses = iter([
        {"response": "AI is great", "model": "phi3:3.8b", "success": True},
        {"response": "", "model": "mistral:7b", "success": False, "error": "timeout"},
        # Only one critique since only one succeeded
        {"response": "AI is amazing", "model": "phi3:3.8b", "success": True},
    ])
    orch.query_model = AsyncMock(side_effect=lambda *a, **kw: next(responses))

    result = await orch.multi_model_swarm_with_debate(
        prompt="What is AI?",
        models=["phi3:3.8b", "mistral:7b"],
    )
    # Only 1 successful response, so no critique wave (needs > 1)
    assert result["debate_waves"] == 1
    assert len(result["models_used"]) == 1
