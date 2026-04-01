"""Pre-vote coalition formation for grouping similar model responses."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class CoalitionFormation:
    """Groups model responses into coalitions based on similarity before voting.

    Coalitions allow similar responses to pool their voting weight behind
    a single representative, improving consensus quality.
    """

    def form_coalitions(
        self, responses: list[dict], similarity_threshold: float = 0.7
    ) -> list[dict]:
        """Group responses by textual similarity into coalitions.

        Args:
            responses: List of dicts with keys 'model', 'response', 'weight'.
            similarity_threshold: Minimum similarity (0-1) to join a coalition.

        Returns:
            List of coalition dicts with coalition_id, members, representative_response,
            and combined_weight.
        """
        if not responses:
            return []

        coalitions: list[dict] = []
        assigned: set[int] = set()

        for i, resp in enumerate(responses):
            if i in assigned:
                continue

            coalition_members = [resp]
            assigned.add(i)

            for j, other in enumerate(responses):
                if j in assigned:
                    continue
                similarity = self.compute_similarity(
                    resp.get("response", ""), other.get("response", "")
                )
                if similarity >= similarity_threshold:
                    coalition_members.append(other)
                    assigned.add(j)

            coalition = {
                "coalition_id": len(coalitions),
                "members": [m.get("model", "unknown") for m in coalition_members],
                "member_responses": coalition_members,
                "combined_weight": sum(
                    m.get("weight", 1.0) for m in coalition_members
                ),
            }
            coalition["representative_response"] = self.select_representative(
                coalition
            )
            coalitions.append(coalition)

        logger.info(
            "Formed %d coalitions from %d responses", len(coalitions), len(responses)
        )
        return coalitions

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute word-overlap similarity (Jaccard index) between two texts.

        Args:
            text1: First text.
            text2: Second text.

        Returns:
            Similarity score between 0.0 and 1.0.
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 and not words2:
            return 1.0
        union = words1 | words2
        if not union:
            return 0.0
        return len(words1 & words2) / len(union)

    def select_representative(self, coalition: dict) -> str:
        """Select the response from the highest-weighted coalition member.

        Args:
            coalition: Coalition dict containing member_responses.

        Returns:
            The response text of the highest-weighted member.
        """
        members = coalition.get("member_responses", [])
        if not members:
            return ""
        best = max(members, key=lambda m: m.get("weight", 0.0))
        return best.get("response", "")

    def apply_coalition_weights(
        self, coalitions: list[dict], original_weights: dict[str, float]
    ) -> dict[str, float]:
        """Assign each coalition's combined weight to its representative.

        Args:
            coalitions: List of coalition dicts from form_coalitions().
            original_weights: Original per-model weights.

        Returns:
            New weight mapping where representative models carry coalition weight.
        """
        new_weights: dict[str, float] = {}
        for coalition in coalitions:
            members = coalition.get("member_responses", [])
            if not members:
                continue
            representative = max(members, key=lambda m: m.get("weight", 0.0))
            rep_model = representative.get("model", "unknown")
            new_weights[rep_model] = coalition.get("combined_weight", 1.0)
        return new_weights
