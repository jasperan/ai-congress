"""
Tests for SemanticVotingEngine
"""
import pytest
import json
from unittest.mock import AsyncMock, MagicMock

from src.ai_congress.core.semantic_voting import (
    ModelResponse,
    Cluster,
    SemanticVoteResult,
    SemanticVotingEngine,
)


class TestModelResponse:
    def test_model_response_creation(self):
        r = ModelResponse(
            model_name="mistral:7b",
            response_text="Paris is the capital of France.",
            weight=0.85,
            temperature=0.7,
        )
        assert r.model_name == "mistral:7b"
        assert r.response_text == "Paris is the capital of France."
        assert r.weight == 0.85
        assert r.temperature == 0.7


class TestCluster:
    def test_cluster_score(self):
        cluster = Cluster(
            id=1,
            label="Paris answer",
            models=["mistral:7b", "phi3:3.8b"],
            key_claims=["Paris is the capital"],
            responses={"mistral:7b": "Paris", "phi3:3.8b": "Paris"},
        )
        weights = {"mistral:7b": 0.8, "phi3:3.8b": 0.6, "llama3.2:3b": 0.5}
        assert cluster.score(weights) == pytest.approx(1.4)

    def test_cluster_score_missing_model(self):
        cluster = Cluster(
            id=1,
            label="test",
            models=["mistral:7b", "unknown_model"],
            key_claims=[],
            responses={"mistral:7b": "yes"},
        )
        weights = {"mistral:7b": 0.8}
        # unknown_model not in weights dict, should default to 0.0
        assert cluster.score(weights) == pytest.approx(0.8)


class TestSemanticVoteResult:
    def test_semantic_vote_result_to_dict(self):
        cluster = Cluster(
            id=1,
            label="Agreement",
            models=["m1"],
            key_claims=["claim1"],
            responses={"m1": "response1"},
        )
        result = SemanticVoteResult(
            winner="response1",
            winning_model="m1",
            clusters=[cluster],
            consensus=0.85,
            debate_triggered=False,
            debate_rounds=0,
            debate_transcript=[],
            dissenting_summary="",
            conviction_scores={},
        )
        d = result.to_dict()
        assert d["winner"] == "response1"
        assert d["winning_model"] == "m1"
        assert d["consensus"] == 0.85
        assert d["debate_triggered"] is False
        assert len(d["clusters"]) == 1
        assert d["clusters"][0]["id"] == 1
        assert d["clusters"][0]["models"] == ["m1"]


class TestSemanticVotingEngine:
    def setup_method(self):
        self.mock_client = AsyncMock()
        self.engine = SemanticVotingEngine(
            ollama_client=self.mock_client, consensus_threshold=0.6
        )

    def _make_responses(self, items):
        """Helper: items is list of (model_name, text, weight)."""
        return [
            ModelResponse(
                model_name=name,
                response_text=text,
                weight=weight,
                temperature=0.7,
            )
            for name, text, weight in items
        ]

    @pytest.mark.asyncio
    async def test_judge_groups_responses(self):
        """Mock ollama returns valid JSON, verify clusters parsed."""
        responses = self._make_responses([
            ("mistral:7b", "Paris is the capital of France.", 0.8),
            ("phi3:3.8b", "The capital of France is Paris.", 0.7),
            ("llama3.2:3b", "London is the capital of England.", 0.6),
        ])

        judge_output = json.dumps({
            "clusters": [
                {
                    "id": 1,
                    "label": "Paris answer",
                    "models": ["mistral:7b", "phi3:3.8b"],
                    "key_claims": ["Paris is the capital of France"],
                },
                {
                    "id": 2,
                    "label": "London answer",
                    "models": ["llama3.2:3b"],
                    "key_claims": ["London is the capital of England"],
                },
            ],
            "analysis": "Two models agree on Paris, one says London.",
        })

        self.mock_client.chat.return_value = {
            "message": {"content": judge_output}
        }

        clusters, analysis = await self.engine.judge_group(responses)

        assert len(clusters) == 2
        assert clusters[0].models == ["mistral:7b", "phi3:3.8b"]
        assert clusters[1].models == ["llama3.2:3b"]
        assert "Two models" in analysis

    @pytest.mark.asyncio
    async def test_judge_fallback_on_invalid_json(self):
        """Mock returns garbage, verify fallback to individual clusters."""
        responses = self._make_responses([
            ("mistral:7b", "Paris", 0.8),
            ("phi3:3.8b", "London", 0.7),
        ])

        self.mock_client.chat.return_value = {
            "message": {"content": "This is not valid JSON at all!!!"}
        }

        clusters, analysis = await self.engine.judge_group(responses)

        # Fallback: each response in its own cluster
        assert len(clusters) == 2
        assert clusters[0].models == ["mistral:7b"]
        assert clusters[1].models == ["phi3:3.8b"]

    @pytest.mark.asyncio
    async def test_vote_reaches_consensus(self):
        """2/3 models agree, consensus >= 0.6."""
        responses = self._make_responses([
            ("mistral:7b", "Paris is the capital.", 0.8),
            ("phi3:3.8b", "Paris, the capital of France.", 0.7),
            ("llama3.2:3b", "London is the capital.", 0.5),
        ])

        judge_output = json.dumps({
            "clusters": [
                {
                    "id": 1,
                    "label": "Paris",
                    "models": ["mistral:7b", "phi3:3.8b"],
                    "key_claims": ["Paris"],
                },
                {
                    "id": 2,
                    "label": "London",
                    "models": ["llama3.2:3b"],
                    "key_claims": ["London"],
                },
            ],
            "analysis": "Majority says Paris.",
        })

        self.mock_client.chat.return_value = {
            "message": {"content": judge_output}
        }

        result = await self.engine.vote(responses)

        assert result is not None
        assert result.consensus >= 0.6
        assert result.winning_model in ["mistral:7b", "phi3:3.8b"]
        assert result.debate_triggered is False
        assert len(result.clusters) == 2

    @pytest.mark.asyncio
    async def test_vote_no_consensus_returns_none(self):
        """50/50 split with equal weights, returns None."""
        responses = self._make_responses([
            ("mistral:7b", "Paris", 0.5),
            ("phi3:3.8b", "London", 0.5),
        ])

        judge_output = json.dumps({
            "clusters": [
                {
                    "id": 1,
                    "label": "Paris",
                    "models": ["mistral:7b"],
                    "key_claims": ["Paris"],
                },
                {
                    "id": 2,
                    "label": "London",
                    "models": ["phi3:3.8b"],
                    "key_claims": ["London"],
                },
            ],
            "analysis": "Split decision.",
        })

        self.mock_client.chat.return_value = {
            "message": {"content": judge_output}
        }

        result = await self.engine.vote(responses)

        # 0.5 / 1.0 = 0.5, below threshold of 0.6
        assert result is None

    @pytest.mark.asyncio
    async def test_single_response_auto_consensus(self):
        """1 response = consensus 1.0."""
        responses = self._make_responses([
            ("mistral:7b", "42 is the answer.", 0.8),
        ])

        result = await self.engine.vote(responses)

        assert result is not None
        assert result.consensus == 1.0
        assert result.winning_model == "mistral:7b"
        assert result.winner == "42 is the answer."

    @pytest.mark.asyncio
    async def test_empty_responses(self):
        """0 responses = empty result."""
        result = await self.engine.vote([])

        assert result is not None
        assert result.winner == ""
        assert result.consensus == 0.0
        assert result.clusters == []
