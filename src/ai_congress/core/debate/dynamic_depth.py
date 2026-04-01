"""Adaptive debate depth controller.

Decides how many debate rounds to run based on initial consensus level,
and provides early-stopping logic and a temperature schedule that cools
down across rounds to converge on a final answer.
"""

import logging

logger = logging.getLogger(__name__)


class DynamicDebateDepth:
    """Dynamically determine and control debate depth based on consensus."""

    def determine_depth(
        self,
        consensus: float,
        num_models: int,
        conviction_variance: float = 0.0,
    ) -> dict:
        """Decide how many debate rounds are needed.

        Args:
            consensus: Initial consensus score (0-1). Higher = more agreement.
            num_models: Number of participating models.
            conviction_variance: Variance of model conviction/confidence scores.
                High variance (>0.3) indicates strongly opposing views.

        Returns:
            Dict with 'rounds' (int) and 'reason' (str).
        """
        if consensus > 0.8:
            rounds = 0
            reason = "Strong consensus, no debate needed"
        elif consensus > 0.6:
            rounds = 1
            reason = "Moderate consensus, brief critique"
        elif consensus > 0.4:
            rounds = 2
            reason = "Split opinion, standard debate"
        elif consensus > 0.2:
            rounds = 3
            reason = "Low consensus, extended debate"
        else:
            rounds = 4
            reason = "Deep disagreement, extended debate with evidence"

        # High conviction variance means strong opposing views -> extra round
        if conviction_variance > 0.3:
            rounds += 1
            reason += " (+1 for high conviction variance)"

        logger.info(
            "Debate depth: %d rounds (consensus=%.2f, models=%d, var=%.2f) — %s",
            rounds, consensus, num_models, conviction_variance, reason,
        )

        return {"rounds": rounds, "reason": reason}

    def should_continue(
        self,
        round_num: int,
        max_rounds: int,
        current_consensus: float,
        prev_consensus: float,
    ) -> bool:
        """Decide whether to continue debating.

        Stops early if:
            - Consensus exceeds 0.8 (strong agreement reached).
            - Consensus did not improve (delta < 0.05).
            - Maximum rounds have been reached.

        Args:
            round_num: Current round number (0-indexed).
            max_rounds: Maximum allowed rounds.
            current_consensus: Consensus after the latest round.
            prev_consensus: Consensus after the previous round.

        Returns:
            True if debate should continue, False to stop.
        """
        if round_num >= max_rounds:
            logger.debug("Stopping: reached max rounds (%d)", max_rounds)
            return False

        if current_consensus > 0.8:
            logger.debug("Stopping: strong consensus reached (%.2f)", current_consensus)
            return False

        delta = abs(current_consensus - prev_consensus)
        if delta < 0.05:
            logger.debug(
                "Stopping: consensus stalled (delta=%.3f)", delta
            )
            return False

        return True

    def get_temperature_schedule(self, total_rounds: int) -> list[float]:
        """Generate a linearly decreasing temperature schedule.

        Temperature decreases from 0.9 (exploratory) to 0.2 (focused)
        across the specified number of rounds.

        Args:
            total_rounds: Number of debate rounds planned.

        Returns:
            List of temperature values, one per round.
        """
        if total_rounds <= 0:
            return []
        if total_rounds == 1:
            return [0.55]  # midpoint

        start, end = 0.9, 0.2
        step = (start - end) / (total_rounds - 1)
        return [round(start - i * step, 3) for i in range(total_rounds)]
