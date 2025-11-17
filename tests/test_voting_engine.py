"""
Tests for VotingEngine
"""
import pytest
from src.ai_congress.core.voting_engine import VotingEngine


class TestVotingEngine:
    def setup_method(self):
        self.engine = VotingEngine()

    def test_weighted_majority_vote_simple(self):
        responses = ["Paris", "London", "Paris"]
        weights = [0.8, 0.6, 0.9]
        model_names = ["model1", "model2", "model3"]

        winner, confidence, breakdown = self.engine.weighted_majority_vote(
            responses, weights, model_names
        )

        assert winner == "Paris"
        assert confidence > 0.5

    def test_majority_vote(self):
        responses = ["Yes", "No", "Yes"]
        winner, confidence, breakdown = self.engine.majority_vote(responses)

        assert winner == "Yes"

    def test_temperature_ensemble(self):
        responses = ["Answer A", "Answer B", "Answer A"]
        temperatures = [0.3, 0.7, 1.0]

        result = self.engine.temperature_ensemble(responses, temperatures)
        assert result in ["Answer A", "Answer B"]

    def test_calculate_consensus_score(self):
        responses = ["Same", "Same", "Same"]
        weights = [0.5, 0.5, 0.5]

        score = self.engine.calculate_consensus_score(responses, weights)
        assert score == 1.0
