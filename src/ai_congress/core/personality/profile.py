import json
import os
from ..acp.message import PersonalityProfile


class ModelPersonalityLoader:
    """Loads and manages personality profiles for LLM models."""

    def __init__(self, config_path: str):
        self._profiles: dict[str, PersonalityProfile] = {}
        self._load(config_path)

    def _load(self, config_path: str) -> None:
        if not os.path.exists(config_path):
            return
        try:
            with open(config_path) as f:
                content = f.read().strip()
            if not content:
                return
            data = json.loads(content)
        except (json.JSONDecodeError, OSError):
            return
        for model_name, traits in data.items():
            self._profiles[model_name] = PersonalityProfile(**traits)

    def get_profile(self, model_name: str) -> PersonalityProfile:
        return self._profiles.get(model_name, PersonalityProfile())

    def compute_emotional_weight_modifier(self, profile: PersonalityProfile) -> float:
        """High confidence + low stress -> boost (up to 1.3x). Low confidence + high stress -> reduction (down to 0.7x)."""
        confidence_factor = 0.7 + (profile.confidence * 0.6)  # 0.7 to 1.3
        stress_penalty = 1.0 - (profile.stress / 20.0)  # 1.0 to 0.5
        return confidence_factor * stress_penalty

    def debate_flexibility(self, profile: PersonalityProfile) -> float:
        """0.0 (stubborn) to 1.0 (very flexible). High agreeableness + low neuroticism -> more flexible."""
        agreeableness_factor = profile.agreeableness / 10.0
        stability_factor = (10 - profile.neuroticism) / 10.0
        return (agreeableness_factor * 0.6 + stability_factor * 0.4)
