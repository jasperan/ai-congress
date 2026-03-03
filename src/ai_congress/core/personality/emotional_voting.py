from collections import defaultdict
from ..acp.message import PersonalityProfile
from .profile import ModelPersonalityLoader


class EmotionalVotingEngine:
    """Extends voting with emotion-based weight modulation."""

    def emotional_weighted_vote(
        self,
        responses: list[str],
        base_weights: list[float],
        personalities: list[PersonalityProfile | None],
        model_names: list[str],
    ) -> tuple[str, float, dict]:
        loader = ModelPersonalityLoader("/dev/null")

        effective_weights = []
        for base_w, personality in zip(base_weights, personalities):
            if personality:
                modifier = loader.compute_emotional_weight_modifier(personality)
                effective_weights.append(base_w * modifier)
            else:
                effective_weights.append(base_w)

        vote_groups: dict[str, dict] = defaultdict(
            lambda: {"weight": 0.0, "votes": 0, "models": []}
        )
        for resp, weight, name in zip(responses, effective_weights, model_names):
            normalized = resp.strip().lower()
            vote_groups[normalized]["weight"] += weight
            vote_groups[normalized]["votes"] += 1
            vote_groups[normalized]["models"].append(name)
            vote_groups[normalized]["original"] = resp

        if not vote_groups:
            return "", 0.0, {}

        winner_key = max(vote_groups, key=lambda k: vote_groups[k]["weight"])
        winner_data = vote_groups[winner_key]
        total_weight = sum(g["weight"] for g in vote_groups.values())
        confidence = winner_data["weight"] / total_weight if total_weight > 0 else 0.0

        details = {
            k: {"weight": v["weight"], "votes": v["votes"], "models": v["models"]}
            for k, v in vote_groups.items()
        }

        return winner_data["original"], confidence, details

    def apply_emotional_drift(
        self, profile: PersonalityProfile, agent_agreed_with_majority: bool
    ) -> None:
        if agent_agreed_with_majority:
            relief = max(1, (10 - profile.neuroticism) // 3)
            profile.stress = max(0, profile.stress - relief)
        else:
            tension = max(1, profile.neuroticism // 3)
            profile.stress = min(10, profile.stress + tension)
