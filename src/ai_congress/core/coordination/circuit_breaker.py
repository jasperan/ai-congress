"""Circuit breaker pattern for model failure management."""

import logging
import time
from typing import Optional

from ...utils.persistence import load_json, save_json

logger = logging.getLogger(__name__)

# Circuit breaker states
STATE_CLOSED = "CLOSED"
STATE_OPEN = "OPEN"
STATE_HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    """Circuit breaker to prevent repeated calls to failing models.

    States:
        CLOSED: Normal operation, requests pass through.
        OPEN: Model is failing, requests are blocked until recovery timeout.
        HALF_OPEN: Testing recovery, limited requests allowed.

    With ``persist_path`` set, per-model state (including the failure
    timestamp) is written on every change, so OPEN breakers survive
    restarts and recovery timing is preserved (3.4.3).
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 1,
        persist_path: Optional[str] = None,
    ):
        """Initialize the circuit breaker.

        Args:
            failure_threshold: Number of consecutive failures before tripping to OPEN.
            recovery_timeout: Seconds to wait in OPEN before transitioning to HALF_OPEN.
            half_open_max_calls: Max calls allowed in HALF_OPEN state before deciding.
            persist_path: Optional JSON file to load/save breaker state.
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self._states: dict[str, dict] = {}
        self.persist_path = persist_path
        if persist_path:
            self._load()

    # ------------------------------------------------------------------
    # Persistence (3.4.3)
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Restore per-model breaker state, preserving failure timestamps."""
        data = load_json(self.persist_path, default=None)
        if not isinstance(data, dict) or not data.get("states"):
            return
        for model, info in data["states"].items():
            if not isinstance(info, dict):
                continue
            self._states[model] = {
                "state": info.get("state", STATE_CLOSED),
                "failure_count": int(info.get("failure_count", 0)),
                "last_failure_time": float(info.get("last_failure_time", 0.0)),
                "half_open_calls": int(info.get("half_open_calls", 0)),
            }
        logger.info("Loaded circuit-breaker state for %d models", len(self._states))

    def _save(self) -> None:
        if not self.persist_path:
            return
        save_json(self.persist_path, {"states": self._states})

    def _get_model_state(self, model: str) -> dict:
        """Get or initialize state for a model."""
        if model not in self._states:
            self._states[model] = {
                "state": STATE_CLOSED,
                "failure_count": 0,
                "last_failure_time": 0.0,
                "half_open_calls": 0,
            }
        return self._states[model]

    def record_success(self, model: str) -> None:
        """Record a successful call, resetting the breaker to CLOSED.

        Args:
            model: The model identifier.
        """
        state = self._get_model_state(model)
        state["state"] = STATE_CLOSED
        state["failure_count"] = 0
        state["half_open_calls"] = 0
        logger.debug("Circuit breaker CLOSED for model %s (success)", model)
        self._save()

    def record_failure(self, model: str) -> None:
        """Record a failed call. Trips to OPEN if threshold exceeded.

        Args:
            model: The model identifier.
        """
        state = self._get_model_state(model)
        state["failure_count"] += 1
        state["last_failure_time"] = time.time()

        if state["failure_count"] >= self.failure_threshold:
            state["state"] = STATE_OPEN
            logger.warning(
                "Circuit breaker OPEN for model %s after %d failures",
                model,
                state["failure_count"],
            )
        self._save()

    def can_execute(self, model: str) -> bool:
        """Check whether a model is available for execution.

        Args:
            model: The model identifier.

        Returns:
            True if the model can accept requests.
        """
        state = self._get_model_state(model)
        current_state = state["state"]

        if current_state == STATE_CLOSED:
            return True

        if current_state == STATE_OPEN:
            elapsed = time.time() - state["last_failure_time"]
            if elapsed >= self.recovery_timeout:
                state["state"] = STATE_HALF_OPEN
                state["half_open_calls"] = 0
                logger.info(
                    "Circuit breaker HALF_OPEN for model %s (recovery timeout elapsed)",
                    model,
                )
                return True
            return False

        if current_state == STATE_HALF_OPEN:
            if state["half_open_calls"] < self.half_open_max_calls:
                state["half_open_calls"] += 1
                return True
            return False

        return False

    def get_state(self, model: str) -> str:
        """Return the current circuit breaker state for a model.

        Args:
            model: The model identifier.

        Returns:
            One of 'CLOSED', 'OPEN', or 'HALF_OPEN'.
        """
        state = self._get_model_state(model)
        return state["state"]

    def get_all_states(self) -> dict[str, dict]:
        """Return circuit breaker states for all tracked models.

        Returns:
            Dict mapping model names to their state dicts.
        """
        return {
            model: {
                "state": info["state"],
                "failure_count": info["failure_count"],
                "last_failure_time": info["last_failure_time"],
                "half_open_calls": info["half_open_calls"],
            }
            for model, info in self._states.items()
        }
