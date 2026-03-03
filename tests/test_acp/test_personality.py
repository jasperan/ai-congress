"""
Tests for Personality Profiles: ModelPersonalityLoader with Big Five traits and emotional states.
"""
import json
import pytest
from src.ai_congress.core.acp.message import PersonalityProfile
from src.ai_congress.core.personality.profile import ModelPersonalityLoader


class TestModelPersonalityLoader:
    def test_load_personalities_from_file(self, tmp_path):
        """Loader should read JSON config and create PersonalityProfile objects."""
        config = {
            "phi3:3.8b": {
                "openness": 8,
                "conscientiousness": 7,
                "extraversion": 4,
                "agreeableness": 6,
                "neuroticism": 3,
                "confidence": 0.7,
                "stress": 2,
                "engagement": 8,
                "communication_style": "analytical",
            },
            "mistral:7b": {
                "openness": 6,
                "conscientiousness": 8,
                "extraversion": 7,
                "agreeableness": 5,
                "neuroticism": 4,
                "confidence": 0.8,
                "stress": 3,
                "engagement": 7,
                "communication_style": "balanced",
            },
        }
        config_file = tmp_path / "models_personality.json"
        config_file.write_text(json.dumps(config))

        loader = ModelPersonalityLoader(str(config_file))

        phi3 = loader.get_profile("phi3:3.8b")
        assert phi3.openness == 8
        assert phi3.conscientiousness == 7
        assert phi3.extraversion == 4
        assert phi3.agreeableness == 6
        assert phi3.neuroticism == 3
        assert phi3.confidence == 0.7
        assert phi3.stress == 2
        assert phi3.engagement == 8
        assert phi3.communication_style == "analytical"

        mistral = loader.get_profile("mistral:7b")
        assert mistral.openness == 6
        assert mistral.conscientiousness == 8
        assert mistral.confidence == 0.8
        assert mistral.communication_style == "balanced"

    def test_unknown_model_gets_default(self, tmp_path):
        """Requesting an unknown model should return a default PersonalityProfile."""
        config = {
            "phi3:3.8b": {
                "openness": 8,
                "conscientiousness": 7,
                "extraversion": 4,
                "agreeableness": 6,
                "neuroticism": 3,
                "confidence": 0.7,
                "stress": 2,
                "engagement": 8,
                "communication_style": "analytical",
            }
        }
        config_file = tmp_path / "models_personality.json"
        config_file.write_text(json.dumps(config))

        loader = ModelPersonalityLoader(str(config_file))
        default = loader.get_profile("unknown:model")

        # Should get default PersonalityProfile values
        assert default.openness == 5
        assert default.conscientiousness == 5
        assert default.extraversion == 5
        assert default.agreeableness == 5
        assert default.neuroticism == 5
        assert default.confidence == 0.5
        assert default.stress == 0
        assert default.engagement == 5
        assert default.communication_style == "balanced"

    def test_emotional_weight_modifier(self):
        """High confidence + low stress should boost weight; low confidence + high stress should reduce it."""
        loader = ModelPersonalityLoader("/dev/null")

        # High confidence (0.9), low stress (1) -> boost
        high_conf = PersonalityProfile(confidence=0.9, stress=1)
        modifier_high = loader.compute_emotional_weight_modifier(high_conf)
        # confidence_factor = 0.7 + (0.9 * 0.6) = 1.24
        # stress_penalty = 1.0 - (1 / 20.0) = 0.95
        # result = 1.24 * 0.95 = 1.178
        assert modifier_high == pytest.approx(1.178, rel=1e-3)
        assert modifier_high > 1.0  # Should boost

        # Low confidence (0.2), high stress (10) -> reduction
        low_conf = PersonalityProfile(confidence=0.2, stress=10)
        modifier_low = loader.compute_emotional_weight_modifier(low_conf)
        # confidence_factor = 0.7 + (0.2 * 0.6) = 0.82
        # stress_penalty = 1.0 - (10 / 20.0) = 0.5
        # result = 0.82 * 0.5 = 0.41
        assert modifier_low == pytest.approx(0.41, rel=1e-3)
        assert modifier_low < 1.0  # Should reduce

        # Moderate: confidence=0.5, stress=0
        moderate = PersonalityProfile(confidence=0.5, stress=0)
        modifier_mod = loader.compute_emotional_weight_modifier(moderate)
        # confidence_factor = 0.7 + (0.5 * 0.6) = 1.0
        # stress_penalty = 1.0 - (0 / 20.0) = 1.0
        # result = 1.0
        assert modifier_mod == pytest.approx(1.0, rel=1e-3)

    def test_debate_flexibility(self):
        """High agreeableness + low neuroticism should yield high flexibility."""
        loader = ModelPersonalityLoader("/dev/null")

        # Very agreeable and stable
        agreeable = PersonalityProfile(agreeableness=10, neuroticism=0)
        flex_high = loader.debate_flexibility(agreeable)
        # agreeableness_factor = 10 / 10.0 = 1.0
        # stability_factor = (10 - 0) / 10.0 = 1.0
        # result = 1.0 * 0.6 + 1.0 * 0.4 = 1.0
        assert flex_high == pytest.approx(1.0, rel=1e-3)

        # Disagreeable and neurotic
        stubborn = PersonalityProfile(agreeableness=0, neuroticism=10)
        flex_low = loader.debate_flexibility(stubborn)
        # agreeableness_factor = 0 / 10.0 = 0.0
        # stability_factor = (10 - 10) / 10.0 = 0.0
        # result = 0.0 * 0.6 + 0.0 * 0.4 = 0.0
        assert flex_low == pytest.approx(0.0, rel=1e-3)

        # Middle ground
        mid = PersonalityProfile(agreeableness=5, neuroticism=5)
        flex_mid = loader.debate_flexibility(mid)
        # agreeableness_factor = 5 / 10.0 = 0.5
        # stability_factor = (10 - 5) / 10.0 = 0.5
        # result = 0.5 * 0.6 + 0.5 * 0.4 = 0.5
        assert flex_mid == pytest.approx(0.5, rel=1e-3)
