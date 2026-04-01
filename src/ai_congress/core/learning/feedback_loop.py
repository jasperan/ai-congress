"""User feedback collection and weight modification."""

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class FeedbackCollector:
    """Collects user feedback on model responses and computes approval-based weight modifiers."""

    def __init__(self) -> None:
        """Initialize the feedback collector."""
        self._feedback: list[dict] = []

    def record_feedback(
        self,
        session_id: str,
        model: str,
        feedback: str,
        response_text: str = "",
    ) -> None:
        """Record user feedback for a model response.

        Args:
            session_id: The session identifier.
            model: The model identifier.
            feedback: Either 'positive' or 'negative'.
            response_text: Optional response text for context.
        """
        if feedback not in ("positive", "negative"):
            logger.warning("Invalid feedback value: %s (expected 'positive' or 'negative')", feedback)
            return

        entry = {
            "session_id": session_id,
            "model": model,
            "feedback": feedback,
            "response_text": response_text,
            "timestamp": time.time(),
        }
        self._feedback.append(entry)
        logger.debug("Recorded %s feedback for model %s in session %s", feedback, model, session_id)

    def get_model_feedback_stats(self, model: str) -> dict:
        """Get feedback statistics for a specific model.

        Args:
            model: The model identifier.

        Returns:
            Dict with positive count, negative count, and approval_rate.
        """
        model_feedback = [f for f in self._feedback if f["model"] == model]
        positive = sum(1 for f in model_feedback if f["feedback"] == "positive")
        negative = sum(1 for f in model_feedback if f["feedback"] == "negative")
        total = positive + negative
        approval_rate = positive / total if total > 0 else 0.0

        return {
            "positive": positive,
            "negative": negative,
            "approval_rate": round(approval_rate, 3),
        }

    def get_feedback_weight_modifier(self, model: str) -> float:
        """Compute a weight modifier based on user feedback approval rate.

        Returns 1.0 if insufficient data (fewer than 5 feedbacks).

        Args:
            model: The model identifier.

        Returns:
            Weight modifier (0.8 to 1.1).
        """
        stats = self.get_model_feedback_stats(model)
        total = stats["positive"] + stats["negative"]

        if total < 5:
            return 1.0

        approval = stats["approval_rate"]
        if approval > 0.8:
            return 1.1
        elif approval > 0.6:
            return 1.0
        elif approval > 0.4:
            return 0.9
        else:
            return 0.8

    def get_all_feedback(self) -> list[dict]:
        """Get all recorded feedback entries.

        Returns:
            List of feedback dicts.
        """
        return list(self._feedback)

    def get_session_feedback(self, session_id: str) -> list[dict]:
        """Get all feedback entries for a specific session.

        Args:
            session_id: The session identifier.

        Returns:
            List of feedback dicts for the session.
        """
        return [f for f in self._feedback if f["session_id"] == session_id]
