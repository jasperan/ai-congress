"""A/B testing and evolution of prompt templates."""

import logging
import random

logger = logging.getLogger(__name__)


class PromptEvolution:
    """Manages prompt template variants with performance-based selection.

    Templates are selected using weighted random sampling biased toward
    higher-performing variants. Outcomes update template scores over time.
    """

    def __init__(self) -> None:
        """Initialize with built-in prompt template variants."""
        self._templates: list[dict] = [
            # Pressure prompt variants for debate rounds
            {
                "id": "pressure_v1",
                "type": "pressure",
                "text": (
                    "Consider the other models' responses carefully. "
                    "If you find merit in their arguments, revise your answer. "
                    "If you disagree, strengthen your position with evidence."
                ),
                "successes": 0,
                "trials": 0,
                "score": 0.5,
            },
            {
                "id": "pressure_v2",
                "type": "pressure",
                "text": (
                    "You have seen the other perspectives. Now provide your "
                    "final, most well-reasoned answer. Be specific and concise. "
                    "Address any counterarguments directly."
                ),
                "successes": 0,
                "trials": 0,
                "score": 0.5,
            },
            {
                "id": "pressure_v3",
                "type": "pressure",
                "text": (
                    "The other models have shared their views. Synthesize the "
                    "strongest points from all perspectives into a unified answer. "
                    "Acknowledge where you were wrong if applicable."
                ),
                "successes": 0,
                "trials": 0,
                "score": 0.5,
            },
            # Critique prompt variants
            {
                "id": "critique_v1",
                "type": "critique",
                "text": (
                    "Analyze the following response for accuracy, completeness, "
                    "and reasoning quality. Identify any errors, gaps, or "
                    "unsupported claims."
                ),
                "successes": 0,
                "trials": 0,
                "score": 0.5,
            },
            {
                "id": "critique_v2",
                "type": "critique",
                "text": (
                    "Rate the following response on a scale of 1-10 for: "
                    "(1) factual accuracy, (2) completeness, (3) clarity. "
                    "Provide specific suggestions for improvement."
                ),
                "successes": 0,
                "trials": 0,
                "score": 0.5,
            },
            {
                "id": "critique_v3",
                "type": "critique",
                "text": (
                    "Play devil's advocate against the following response. "
                    "What are the weakest points? What important aspects "
                    "were overlooked? What alternative conclusions are possible?"
                ),
                "successes": 0,
                "trials": 0,
                "score": 0.5,
            },
        ]

    def select_template(self, template_type: str) -> dict:
        """Select a template using weighted random sampling biased toward higher scores.

        Args:
            template_type: The type of template ('pressure' or 'critique').

        Returns:
            Selected template dict with id, type, and text.
        """
        candidates = [t for t in self._templates if t["type"] == template_type]
        if not candidates:
            logger.warning("No templates found for type: %s", template_type)
            return {"id": "fallback", "type": template_type, "text": "Please respond."}

        # Weighted random selection using scores as weights
        weights = [max(t["score"], 0.01) for t in candidates]
        selected = random.choices(candidates, weights=weights, k=1)[0]

        logger.debug("Selected template %s (score: %.3f)", selected["id"], selected["score"])
        return {
            "id": selected["id"],
            "type": selected["type"],
            "text": selected["text"],
        }

    def record_outcome(
        self, template_id: str, consensus_reached: bool, rounds_needed: int
    ) -> None:
        """Record the outcome of using a template.

        Args:
            template_id: The template identifier.
            consensus_reached: Whether consensus was achieved.
            rounds_needed: Number of debate rounds needed.
        """
        for template in self._templates:
            if template["id"] == template_id:
                template["trials"] += 1
                if consensus_reached:
                    template["successes"] += 1
                # Score: success rate penalized by rounds needed
                if template["trials"] > 0:
                    success_rate = template["successes"] / template["trials"]
                    round_penalty = max(0, 1 - (rounds_needed * 0.1))
                    template["score"] = success_rate * round_penalty
                logger.debug(
                    "Template %s updated: score=%.3f (trials=%d, successes=%d)",
                    template_id,
                    template["score"],
                    template["trials"],
                    template["successes"],
                )
                return

        logger.warning("Template not found: %s", template_id)

    def get_template_stats(self) -> dict[str, dict]:
        """Get performance statistics for all templates.

        Returns:
            Dict mapping template IDs to their performance data.
        """
        return {
            t["id"]: {
                "type": t["type"],
                "trials": t["trials"],
                "successes": t["successes"],
                "score": round(t["score"], 3),
                "text_preview": t["text"][:80] + "..." if len(t["text"]) > 80 else t["text"],
            }
            for t in self._templates
        }
