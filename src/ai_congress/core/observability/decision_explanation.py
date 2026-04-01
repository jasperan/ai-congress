"""Auto-generation of human-readable consensus decision explanations."""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class DecisionExplainer:
    """Generates human-readable explanations of voting and consensus decisions."""

    def explain(
        self,
        vote_result: dict[str, Any],
        debate_transcript: Optional[list] = None,
        role_assignments: Optional[dict[str, str]] = None,
    ) -> str:
        """Generate a human-readable explanation of a consensus decision.

        Args:
            vote_result: Voting result dict with winner, scores, confidence, etc.
            debate_transcript: Optional list of debate round data.
            role_assignments: Optional mapping of model to role name.

        Returns:
            Multi-line explanation string.
        """
        lines: list[str] = ["This answer was chosen because:"]

        # Extract vote data
        winner = vote_result.get("winner", vote_result.get("selected_model", "unknown"))
        scores = vote_result.get("scores", vote_result.get("model_scores", {}))
        confidence = vote_result.get("confidence", vote_result.get("consensus_score", 0.0))
        total_models = vote_result.get("total_models", len(scores))
        votes = vote_result.get("votes", {})

        # Majority analysis
        if scores:
            # Find models that voted for the winner
            supporting_models: list[str] = []
            dissenting_models: list[str] = []
            for model, score in scores.items():
                if model == winner or score == max(scores.values()):
                    supporting_models.append(model)
                elif score < max(scores.values()) * 0.5:
                    dissenting_models.append(model)

            if supporting_models:
                supporter_count = len(supporting_models)
                lines.append(
                    f"- {supporter_count}/{total_models} models agreed on this core position"
                )

            # Show weight information for top supporters
            weight_info = vote_result.get("weights", {})
            if weight_info and supporting_models:
                weighted_parts = []
                for model in supporting_models[:3]:
                    w = weight_info.get(model, 0.0)
                    if w > 0:
                        weighted_parts.append(f"{model} ({w:.2f})")
                if weighted_parts:
                    lines.append(
                        f"- {' and '.join(weighted_parts)} formed the majority"
                    )

        # Debate round info
        if debate_transcript:
            num_rounds = len(debate_transcript)
            # Check for position changes
            revised_models: list[str] = []
            for round_data in debate_transcript:
                for resp in round_data if isinstance(round_data, list) else [round_data]:
                    if isinstance(resp, dict) and resp.get("revised", False):
                        model_name = resp.get("model", "unknown")
                        if model_name not in revised_models:
                            revised_models.append(model_name)

            if revised_models:
                lines.append(
                    f"- After {num_rounds} debate round(s), "
                    f"{', '.join(revised_models)} revised their position to align"
                )
            elif num_rounds > 0:
                lines.append(f"- {num_rounds} debate round(s) were conducted")

        # Confidence
        if confidence:
            lines.append(f"- Confidence: {confidence:.0%}")

        # Dissenting views
        if scores:
            minority_models = [
                m for m, s in scores.items()
                if s < max(scores.values()) * 0.7 and m != winner
            ]
            if minority_models:
                lines.append(
                    f"- Dissenting view supported by: {', '.join(minority_models)}"
                )

        return "\n".join(lines)

    def build_summary(self, result: dict[str, Any]) -> dict[str, Any]:
        """Build a structured summary of the decision.

        Args:
            result: Full voting/debate result dict.

        Returns:
            Dict with explanation_text, key_factors list, and confidence_level string.
        """
        explanation = self.explain(result)
        confidence = result.get("confidence", result.get("consensus_score", 0.0))

        # Determine confidence level label
        if confidence >= 0.8:
            confidence_level = "high"
        elif confidence >= 0.5:
            confidence_level = "medium"
        else:
            confidence_level = "low"

        # Extract key factors
        key_factors: list[str] = []
        scores = result.get("scores", result.get("model_scores", {}))
        if scores:
            total = len(scores)
            winner = result.get("winner", "")
            agreeing = sum(
                1 for s in scores.values() if s >= max(scores.values()) * 0.7
            )
            key_factors.append(f"{agreeing}/{total} models in agreement")

        debate_rounds = result.get("debate_rounds", result.get("num_rounds", 0))
        if debate_rounds:
            key_factors.append(f"{debate_rounds} debate rounds conducted")

        key_factors.append(f"Confidence: {confidence_level} ({confidence:.0%})")

        return {
            "explanation_text": explanation,
            "key_factors": key_factors,
            "confidence_level": confidence_level,
        }
