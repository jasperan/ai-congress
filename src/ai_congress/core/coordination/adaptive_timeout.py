"""Per-model adaptive timeout management using exponential moving averages."""

import logging

logger = logging.getLogger(__name__)


class AdaptiveTimeout:
    """Computes per-model timeouts based on observed latency history.

    Uses exponential moving average (EMA) to smooth latency measurements
    and sets timeouts at 3x the EMA, clamped within configurable bounds.
    """

    def __init__(
        self,
        default_timeout: float = 60.0,
        min_timeout: float = 15.0,
        max_timeout: float = 180.0,
    ):
        """Initialize the adaptive timeout manager.

        Args:
            default_timeout: Timeout (seconds) for models with no history.
            min_timeout: Minimum allowed timeout in seconds.
            max_timeout: Maximum allowed timeout in seconds.
        """
        self.default_timeout = default_timeout
        self.min_timeout = min_timeout
        self.max_timeout = max_timeout
        self._latencies: dict[str, float] = {}  # model -> EMA latency in ms

    def record_latency(self, model: str, latency_ms: float) -> None:
        """Record an observed latency for a model, updating the EMA.

        Args:
            model: The model identifier.
            latency_ms: Observed latency in milliseconds.
        """
        if model in self._latencies:
            self._latencies[model] = 0.8 * self._latencies[model] + 0.2 * latency_ms
        else:
            self._latencies[model] = latency_ms

        logger.debug(
            "Model %s latency recorded: %.1fms (EMA: %.1fms)",
            model,
            latency_ms,
            self._latencies[model],
        )

    def get_timeout(self, model: str) -> float:
        """Get the adaptive timeout for a model in seconds.

        Returns 3x the EMA latency, clamped to [min_timeout, max_timeout].
        Falls back to default_timeout if no history exists.

        Args:
            model: The model identifier.

        Returns:
            Timeout in seconds.
        """
        if model not in self._latencies:
            return self.default_timeout

        ema_ms = self._latencies[model]
        timeout_s = (ema_ms * 3.0) / 1000.0
        clamped = max(self.min_timeout, min(self.max_timeout, timeout_s))
        return clamped

    def get_stats(self) -> dict[str, dict[str, float]]:
        """Get per-model latency statistics and computed timeouts.

        Returns:
            Dict mapping model names to {avg_latency_ms, timeout_s}.
        """
        stats: dict[str, dict[str, float]] = {}
        for model, ema_ms in self._latencies.items():
            stats[model] = {
                "avg_latency_ms": round(ema_ms, 1),
                "timeout_s": round(self.get_timeout(model), 1),
            }
        return stats
