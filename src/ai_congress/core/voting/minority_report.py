"""Minority report generation for voting dissent analysis.

After a vote is decided, this module extracts and formats the strongest
dissenting position so that minority viewpoints are not silently discarded.
"""

import logging

logger = logging.getLogger(__name__)


class MinorityReportGenerator:
    """Extracts and formats minority / dissenting views from vote results."""

    def extract_minority(self, vote_details: dict, winner: str) -> dict:
        """Find the strongest dissenting position after the winner.

        Args:
            vote_details: The vote_details dict returned by VotingEngine
                (mapping normalised response -> {original, weight, votes, models}).
            winner: The winning response text.

        Returns:
            Dict with dissenting_answer, dissenting_models, dissenting_weight,
            and dissenting_ratio. Returns empty values when no minority exists.
        """
        winner_norm = winner.strip().lower()

        # Collect non-winner entries sorted by weight descending
        dissenters = [
            (key, info)
            for key, info in vote_details.items()
            if key != winner_norm
        ]
        dissenters.sort(key=lambda x: x[1]["weight"], reverse=True)

        if not dissenters:
            return {
                "dissenting_answer": None,
                "dissenting_models": [],
                "dissenting_weight": 0.0,
                "dissenting_ratio": 0.0,
            }

        top_dissent_key, top_dissent = dissenters[0]
        total_weight = sum(info["weight"] for info in vote_details.values())
        dissenting_ratio = (
            top_dissent["weight"] / total_weight if total_weight > 0 else 0.0
        )

        return {
            "dissenting_answer": top_dissent["original"],
            "dissenting_models": top_dissent.get("models", []),
            "dissenting_weight": top_dissent["weight"],
            "dissenting_ratio": dissenting_ratio,
        }

    def format_minority_report(
        self, winner: str, confidence: float, minority: dict
    ) -> str:
        """Format a human-readable minority report string.

        Args:
            winner: The winning response text.
            confidence: Confidence score of the winner (0-1).
            minority: Dict produced by extract_minority().

        Returns:
            Formatted report string.
        """
        if not minority.get("dissenting_answer"):
            return (
                f"Consensus: {winner} ({confidence:.0%} confidence). "
                "No dissenting views."
            )

        models_str = ", ".join(minority["dissenting_models"]) or "unknown models"
        return (
            f"Consensus: {winner} ({confidence:.0%} confidence). "
            f"Dissenting view: {minority['dissenting_answer']} "
            f"(supported by {models_str} with "
            f"{minority['dissenting_ratio']:.0%} of vote)"
        )

    def should_surface_minority(
        self, minority: dict, threshold: float = 0.2
    ) -> bool:
        """Decide whether a minority view is significant enough to surface.

        Args:
            minority: Dict produced by extract_minority().
            threshold: Minimum dissenting_ratio to consider significant.

        Returns:
            True if the minority holds more than threshold share of the vote.
        """
        return minority.get("dissenting_ratio", 0.0) > threshold
