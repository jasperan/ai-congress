"""
Tests for DebateManager - patience mechanic, indecisiveness detection,
conviction bonus, and full debate flow.
"""
import pytest
import json
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from src.ai_congress.core.semantic_voting import (
    ModelResponse,
    Cluster,
    SemanticVoteResult,
    SemanticVotingEngine,
)
from src.ai_congress.core.debate_manager import DebateManager, DebateConfig


class TestDebateConfig:
    def test_defaults(self):
        cfg = DebateConfig()
        assert cfg.max_rounds == 3
        assert cfg.consensus_threshold == 0.6
        assert cfg.temp_schedule == [0.9, 0.5, 0.2]
        assert cfg.conviction_bonus == 1.2

    def test_pressure_prompt_round_0(self):
        cfg = DebateConfig()
        prompt = cfg.pressure_prompt(0, is_indecisive=False)
        assert "consider" in prompt.lower()

    def test_pressure_prompt_escalates(self):
        cfg = DebateConfig()
        p0 = cfg.pressure_prompt(0, is_indecisive=False)
        p1 = cfg.pressure_prompt(1, is_indecisive=False)
        p2 = cfg.pressure_prompt(2, is_indecisive=False)

        assert "consider" in p0.lower()
        assert "clear position" in p1.lower()
        assert "commit" in p2.lower() or "final" in p2.lower()

    def test_pressure_prompt_indecisive_appends_note(self):
        cfg = DebateConfig()
        prompt = cfg.pressure_prompt(0, is_indecisive=True)
        assert "inconsistent" in prompt.lower() or "uncommitted" in prompt.lower()
        assert "must" in prompt.lower()


class TestDebateManagerIndecisive:
    def setup_method(self):
        self.mock_client = AsyncMock()
        self.mock_voting = MagicMock(spec=SemanticVotingEngine)
        self.dm = DebateManager(self.mock_client, self.mock_voting)

    def test_model_that_switched_clusters_is_indecisive(self):
        position_history = {"a": [1, 2], "b": [1, 1]}
        indecisive = self.dm.detect_indecisive(position_history, round_num=1)
        assert "a" in indecisive
        assert "b" not in indecisive

    def test_singleton_cluster_is_indecisive(self):
        clusters = [
            Cluster(
                id=1,
                label="majority",
                models=["a", "b"],
                key_claims=[],
                responses={"a": "yes", "b": "yes"},
            ),
            Cluster(
                id=2,
                label="lone wolf",
                models=["c"],
                key_claims=[],
                responses={"c": "no"},
            ),
        ]
        indecisive = self.dm.detect_indecisive_from_clusters(clusters)
        assert "c" in indecisive
        assert "a" not in indecisive
        assert "b" not in indecisive


class TestConvictionBonus:
    def setup_method(self):
        self.mock_client = AsyncMock()
        self.mock_voting = MagicMock(spec=SemanticVotingEngine)
        self.dm = DebateManager(self.mock_client, self.mock_voting)

    def test_consistent_model_gets_bonus(self):
        position_history = {"a": [1, 1, 1], "b": [1, 2, 1]}
        weights = {"a": 1.0, "b": 1.0}
        adjusted = self.dm.apply_conviction_bonus(weights, position_history, bonus=1.2)
        assert adjusted["a"] == pytest.approx(1.2)
        assert adjusted["b"] == pytest.approx(1.0)


class TestDebateFlow:
    def setup_method(self):
        self.mock_client = AsyncMock()
        self.mock_voting = AsyncMock(spec=SemanticVotingEngine)
        self.config = DebateConfig(max_rounds=3, consensus_threshold=0.6)

    def _make_responses(self):
        return [
            ModelResponse("model_a", "Answer A", 0.5, 0.7),
            ModelResponse("model_b", "Answer B", 0.5, 0.7),
        ]

    @pytest.mark.asyncio
    async def test_debate_reaches_consensus_after_one_round(self):
        """Models converge in round 1 — debate ends early."""
        responses = self._make_responses()

        initial_clusters = [
            Cluster(id=1, label="A", models=["model_a"], key_claims=[], responses={"model_a": "Answer A"}),
            Cluster(id=2, label="B", models=["model_b"], key_claims=[], responses={"model_b": "Answer B"}),
        ]
        initial_analysis = "Split decision."

        # After round 1, both models converge to same cluster
        converged_clusters = [
            Cluster(id=1, label="A", models=["model_a", "model_b"], key_claims=[], responses={"model_a": "Answer A revised", "model_b": "Answer A revised too"}),
        ]

        # Mock model debate responses (plain text)
        self.mock_client.chat.return_value = {
            "message": {"content": "Answer A revised"}
        }

        # Mock judge re-grouping: returns converged clusters
        self.mock_voting.judge_group.return_value = (converged_clusters, "All agree now.")

        dm = DebateManager(self.mock_client, self.mock_voting, self.config)
        result = dm.run_debate(
            "What is the answer?", responses, initial_clusters, initial_analysis
        )
        result = await result

        assert result is not None
        assert result.consensus >= 0.6
        assert result.debate_rounds <= 1
        assert result.debate_triggered is True

    @pytest.mark.asyncio
    async def test_debate_exhausts_patience(self):
        """Consensus never reached — force-selects after max_rounds."""
        responses = self._make_responses()

        split_clusters = [
            Cluster(id=1, label="A", models=["model_a"], key_claims=[], responses={"model_a": "Answer A"}),
            Cluster(id=2, label="B", models=["model_b"], key_claims=[], responses={"model_b": "Answer B"}),
        ]
        initial_analysis = "Split."

        # Model chat always returns same answer (no convergence)
        self.mock_client.chat.return_value = {
            "message": {"content": "Still my original answer"}
        }

        # Judge always returns split clusters
        self.mock_voting.judge_group.return_value = (split_clusters, "Still split.")

        config = DebateConfig(max_rounds=2, consensus_threshold=0.6)
        dm = DebateManager(self.mock_client, self.mock_voting, config)
        result = await dm.run_debate(
            "What is the answer?", responses, split_clusters, initial_analysis
        )

        assert result is not None
        assert result.debate_rounds == 2
        assert result.debate_triggered is True
        # Force-selected: should have a winner even without consensus
        assert result.winner != ""
        assert result.winning_model != ""
