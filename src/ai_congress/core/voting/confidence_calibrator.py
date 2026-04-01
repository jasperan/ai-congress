"""Confidence calibration for model predictions.

Tracks predicted confidence vs actual accuracy per model and applies
binned calibration to correct over- or under-confident predictions.
"""

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

NUM_BINS = 10
MIN_OBSERVATIONS = 5


class ConfidenceCalibrator:
    """Calibrates model confidence scores based on historical accuracy.

    Uses binned calibration: the [0, 1] confidence range is divided into
    10 equal bins. For each bin the actual accuracy (fraction of correct
    predictions) is tracked. When calibrating a new raw confidence value,
    the bin's empirical accuracy is returned instead of the raw score.
    """

    def __init__(self) -> None:
        # model -> bin_index -> {"correct": int, "total": int}
        self._data: dict[str, dict[int, dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: {"correct": 0, "total": 0})
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bin_index(confidence: float) -> int:
        """Return the bin index (0-9) for a confidence value in [0, 1]."""
        idx = int(confidence * NUM_BINS)
        return min(idx, NUM_BINS - 1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, model: str, predicted_confidence: float, was_correct: bool) -> None:
        """Store an observation of predicted confidence vs actual outcome.

        Args:
            model: Model identifier.
            predicted_confidence: The confidence the model reported (0-1).
            was_correct: Whether the prediction turned out to be correct.
        """
        bin_idx = self._bin_index(predicted_confidence)
        bucket = self._data[model][bin_idx]
        bucket["total"] += 1
        if was_correct:
            bucket["correct"] += 1

        logger.debug(
            "Recorded observation for %s: conf=%.2f bin=%d correct=%s",
            model, predicted_confidence, bin_idx, was_correct,
        )

    def calibrate(self, model: str, raw_confidence: float) -> float:
        """Apply calibration curve to a raw confidence value.

        If fewer than MIN_OBSERVATIONS exist in the corresponding bin the
        raw confidence is returned unchanged.

        Args:
            model: Model identifier.
            raw_confidence: The uncalibrated confidence value (0-1).

        Returns:
            Calibrated confidence value (0-1).
        """
        bin_idx = self._bin_index(raw_confidence)
        bucket = self._data.get(model, {}).get(bin_idx)

        if bucket is None or bucket["total"] < MIN_OBSERVATIONS:
            return raw_confidence

        calibrated = bucket["correct"] / bucket["total"]
        logger.debug(
            "Calibrated %s confidence %.2f -> %.2f (bin %d, n=%d)",
            model, raw_confidence, calibrated, bin_idx, bucket["total"],
        )
        return calibrated

    def get_calibration_stats(self, model: str) -> dict:
        """Return per-bin calibration statistics for a single model.

        Args:
            model: Model identifier.

        Returns:
            Dict mapping bin index to {correct, total, accuracy}.
        """
        stats: dict[int, dict] = {}
        for bin_idx in range(NUM_BINS):
            bucket = self._data.get(model, {}).get(bin_idx)
            if bucket and bucket["total"] > 0:
                stats[bin_idx] = {
                    "correct": bucket["correct"],
                    "total": bucket["total"],
                    "accuracy": bucket["correct"] / bucket["total"],
                    "bin_range": (bin_idx / NUM_BINS, (bin_idx + 1) / NUM_BINS),
                }
        return stats

    def get_all_stats(self) -> dict:
        """Return calibration statistics for all tracked models.

        Returns:
            Dict mapping model name to its per-bin stats.
        """
        return {model: self.get_calibration_stats(model) for model in self._data}
