"""
Tests for Emotional Voting Engine: emotion-modulated weighted voting.
"""
import pytest
from src.ai_congress.core.acp.message import PersonalityProfile
from src.ai_congress.core.personality.emotional_voting import EmotionalVotingEngine


class TestEmotionalVotingEngine:
    def test_emotional_weighted_vote_basic(self):
        """Basic voting with personalities should produce a winner with confidence."""
        engine = EmotionalVotingEngine()

        responses = ["Option A", "Option A", "Option B"]
        base_weights = [1.0, 1.0, 1.0]
        personalities = [
            PersonalityProfile(confidence=0.8, stress=1),
            PersonalityProfile(confidence=0.7, stress=2),
            PersonalityProfile(confidence=0.5, stress=5),
        ]
        model_names = ["model1", "model2", "model3"]

        winner, confidence, details = engine.emotional_weighted_vote(
            responses, base_weights, personalities, model_names
        )

        assert winner.strip().lower() == "option a"
        assert confidence > 0.5  # Majority voted A, should be confident
        assert "option a" in details
        assert "option b" in details
        assert details["option a"]["votes"] == 2
        assert details["option b"]["votes"] == 1
        assert len(details["option a"]["models"]) == 2

    def test_emotional_vote_stress_reduces_weight(self):
        """A high-stress personality should have its effective weight reduced."""
        engine = EmotionalVotingEngine()

        # Two responses: A (high stress) vs B (low stress, high confidence)
        responses = ["Option A", "Option B"]
        base_weights = [1.0, 1.0]
        personalities = [
            PersonalityProfile(confidence=0.5, stress=10),  # stressed -> low weight
            PersonalityProfile(confidence=0.9, stress=0),   # calm, confident -> high weight
        ]
        model_names = ["stressed_model", "calm_model"]

        winner, confidence, details = engine.emotional_weighted_vote(
            responses, base_weights, personalities, model_names
        )

        # Option B should win because calm_model has higher effective weight
        assert winner.strip().lower() == "option b"
        assert details["option b"]["weight"] > details["option a"]["weight"]

    def test_emotional_vote_no_personalities_falls_back(self):
        """When no personalities are provided, base weights should be used as-is."""
        engine = EmotionalVotingEngine()

        responses = ["Yes", "Yes", "No"]
        base_weights = [1.0, 1.0, 1.0]
        personalities = [None, None, None]
        model_names = ["m1", "m2", "m3"]

        winner, confidence, details = engine.emotional_weighted_vote(
            responses, base_weights, personalities, model_names
        )

        assert winner.strip().lower() == "yes"
        # With equal weights: 2/3 confidence
        assert confidence == pytest.approx(2.0 / 3.0, rel=1e-3)
        assert details["yes"]["weight"] == pytest.approx(2.0, rel=1e-3)
        assert details["no"]["weight"] == pytest.approx(1.0, rel=1e-3)

    def test_stress_drift_after_disagreement(self):
        """Disagreeing with majority should increase stress based on neuroticism."""
        engine = EmotionalVotingEngine()

        # Neurotic agent (neuroticism=9) disagrees
        profile = PersonalityProfile(neuroticism=9, stress=3)
        engine.apply_emotional_drift(profile, agent_agreed_with_majority=False)

        # tension = max(1, 9 // 3) = 3
        # new stress = min(10, 3 + 3) = 6
        assert profile.stress == 6

        # Less neurotic agent (neuroticism=2) disagrees
        calm_profile = PersonalityProfile(neuroticism=2, stress=3)
        engine.apply_emotional_drift(calm_profile, agent_agreed_with_majority=False)

        # tension = max(1, 2 // 3) = max(1, 0) = 1
        # new stress = min(10, 3 + 1) = 4
        assert calm_profile.stress == 4

    def test_stress_stable_after_agreement(self):
        """Agreeing with majority should reduce stress based on emotional stability."""
        engine = EmotionalVotingEngine()

        # Stable agent (low neuroticism=2) agrees
        stable = PersonalityProfile(neuroticism=2, stress=5)
        engine.apply_emotional_drift(stable, agent_agreed_with_majority=True)

        # relief = max(1, (10 - 2) // 3) = max(1, 2) = 2
        # new stress = max(0, 5 - 2) = 3
        assert stable.stress == 3

        # Already calm agent
        calm = PersonalityProfile(neuroticism=5, stress=1)
        engine.apply_emotional_drift(calm, agent_agreed_with_majority=True)

        # relief = max(1, (10 - 5) // 3) = max(1, 1) = 1
        # new stress = max(0, 1 - 1) = 0
        assert calm.stress == 0
