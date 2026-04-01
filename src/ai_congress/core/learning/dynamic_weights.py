"""Dynamic weight adjustment based on model performance over time."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class DynamicWeightManager:
    """Adjusts model weights dynamically based on win/loss outcomes.

    Uses exponential moving average to blend base benchmark weights
    with observed performance (win rate).
    """

    def __init__(
        self,
        base_weights: Optional[dict[str, float]] = None,
        learning_rate: float = 0.1,
        min_weight: float = 0.1,
        max_weight: float = 1.0,
    ):
        """Initialize the dynamic weight manager.

        Args:
            base_weights: Initial benchmark weights per model (default 0.5 for unknown).
            learning_rate: How fast weights adapt to new outcomes (0-1).
            min_weight: Minimum allowed weight.
            max_weight: Maximum allowed weight.
        """
        self.base_weights: dict[str, float] = dict(base_weights) if base_weights else {}
        self.learning_rate = learning_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        self._current_weights: dict[str, float] = dict(self.base_weights)
        self._outcomes: dict[str, dict[str, int]] = {}  # model -> {wins, total}

    def record_outcome(self, model: str, was_winner: bool) -> None:
        """Record a win or loss for a model.

        Args:
            model: The model identifier.
            was_winner: True if the model's response was selected as winner.
        """
        if model not in self._outcomes:
            self._outcomes[model] = {"wins": 0, "total": 0}
        self._outcomes[model]["total"] += 1
        if was_winner:
            self._outcomes[model]["wins"] += 1

        logger.debug(
            "Model %s outcome: %s (wins: %d/%d)",
            model,
            "win" if was_winner else "loss",
            self._outcomes[model]["wins"],
            self._outcomes[model]["total"],
        )

    def update_weights(self) -> None:
        """Update all model weights using EMA blending with win rate.

        Formula: new_weight = (1 - learning_rate) * old_weight + learning_rate * win_rate
        Weights are clamped to [min_weight, max_weight].
        """
        for model, stats in self._outcomes.items():
            if stats["total"] == 0:
                continue
            win_rate = stats["wins"] / stats["total"]
            old_weight = self._current_weights.get(
                model, self.base_weights.get(model, 0.5)
            )
            new_weight = (1 - self.learning_rate) * old_weight + self.learning_rate * win_rate
            self._current_weights[model] = max(
                self.min_weight, min(self.max_weight, new_weight)
            )

        logger.info("Updated dynamic weights: %s", self._current_weights)

    def get_weight(self, model: str) -> float:
        """Get the current adjusted weight for a model.

        Args:
            model: The model identifier.

        Returns:
            Current weight value.
        """
        return self._current_weights.get(
            model, self.base_weights.get(model, 0.5)
        )

    def get_all_weights(self) -> dict[str, float]:
        """Get all current model weights.

        Returns:
            Dict mapping model names to their current weights.
        """
        all_models = set(self.base_weights.keys()) | set(self._current_weights.keys())
        return {
            model: self.get_weight(model) for model in sorted(all_models)
        }

    def get_performance_stats(self) -> dict[str, dict]:
        """Get per-model performance statistics.

        Returns:
            Dict mapping model names to {win_rate, total_participations, current_weight}.
        """
        stats: dict[str, dict] = {}
        all_models = (
            set(self.base_weights.keys())
            | set(self._current_weights.keys())
            | set(self._outcomes.keys())
        )
        for model in sorted(all_models):
            outcome = self._outcomes.get(model, {"wins": 0, "total": 0})
            total = outcome["total"]
            win_rate = outcome["wins"] / total if total > 0 else 0.0
            stats[model] = {
                "win_rate": round(win_rate, 3),
                "total_participations": total,
                "current_weight": round(self.get_weight(model), 3),
            }
        return stats

    def reset_to_base(self) -> None:
        """Reset all weights to their original benchmark values."""
        self._current_weights = dict(self.base_weights)
        self._outcomes.clear()
        logger.info("Weights reset to base values")
