"""Cross-session personality state persistence."""

import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)


class PersonalityPersistence:
    """Persists personality state (stress, confidence, engagement) across sessions.

    Stores per-model JSON files in a configurable directory.
    """

    def __init__(self, storage_dir: str = "data/personality_state") -> None:
        """Initialize personality persistence.

        Args:
            storage_dir: Directory for storing personality state JSON files.
        """
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    def _model_path(self, model: str) -> str:
        """Get the file path for a model's state file."""
        safe_name = model.replace("/", "_").replace(":", "_").replace(".", "_")
        return os.path.join(self.storage_dir, f"{safe_name}.json")

    def save_state(self, model: str, profile_updates: dict[str, Any]) -> None:
        """Save personality state updates for a model.

        Args:
            model: The model identifier.
            profile_updates: Dict of personality state changes (e.g., stress, confidence).
        """
        path = self._model_path(model)
        existing = self.load_state(model)
        existing.update(profile_updates)
        existing["model"] = model

        try:
            with open(path, "w") as f:
                json.dump(existing, f, indent=2)
            logger.debug("Saved personality state for %s", model)
        except OSError as e:
            logger.error("Failed to save personality state for %s: %s", model, e)

    def load_state(self, model: str) -> dict[str, Any]:
        """Load the last saved personality state for a model.

        Args:
            model: The model identifier.

        Returns:
            Dict of personality state, or empty dict if none exists.
        """
        path = self._model_path(model)
        if not os.path.exists(path):
            return {}

        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Failed to load personality state for %s: %s", model, e)
            return {}

    def apply_persisted_state(self, model: str, profile: Any) -> None:
        """Apply persisted state to a personality profile object in-place.

        Modifies the profile's attributes with any saved state values.

        Args:
            model: The model identifier.
            profile: A personality profile object with settable attributes.
        """
        state = self.load_state(model)
        if not state:
            return

        for key, value in state.items():
            if key == "model":
                continue
            if hasattr(profile, key):
                setattr(profile, key, value)
                logger.debug("Applied persisted %s=%s to %s", key, value, model)

    def reset_state(self, model: str) -> None:
        """Delete saved personality state for a model.

        Args:
            model: The model identifier.
        """
        path = self._model_path(model)
        if os.path.exists(path):
            try:
                os.remove(path)
                logger.info("Reset personality state for %s", model)
            except OSError as e:
                logger.error("Failed to reset personality state for %s: %s", model, e)

    def get_all_states(self) -> dict[str, dict[str, Any]]:
        """Get all persisted personality states.

        Returns:
            Dict mapping model names to their personality state dicts.
        """
        states: dict[str, dict[str, Any]] = {}
        if not os.path.isdir(self.storage_dir):
            return states

        for filename in os.listdir(self.storage_dir):
            if not filename.endswith(".json"):
                continue
            path = os.path.join(self.storage_dir, filename)
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                model = data.get("model", filename.replace(".json", ""))
                states[model] = data
            except (json.JSONDecodeError, OSError) as e:
                logger.error("Failed to read %s: %s", filename, e)

        return states
