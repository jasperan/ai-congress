"""Graceful degradation strategies based on available model count."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class GracefulDegradation:
    """Adjusts swarm behavior based on the number of available models.

    Ensures the system remains functional even when most models are
    unavailable, by simplifying debate rounds and voting strategy.
    """

    def determine_mode(self, available_models: int, total_models: int) -> dict[str, Any]:
        """Determine the operational mode based on available model count.

        Args:
            available_models: Number of models currently reachable.
            total_models: Total number of configured models.

        Returns:
            Dict describing the mode, debate_rounds, voting strategy, and any caveats.
        """
        if available_models >= 5:
            mode = {
                "mode": "full",
                "debate_rounds": 3,
                "voting": "semantic",
            }
        elif available_models >= 3:
            mode = {
                "mode": "simplified",
                "debate_rounds": 1,
                "voting": "weighted",
            }
        elif available_models == 2:
            mode = {
                "mode": "comparison",
                "debate_rounds": 0,
                "voting": "weighted",
            }
        elif available_models == 1:
            mode = {
                "mode": "single",
                "debate_rounds": 0,
                "voting": "none",
                "caveat": "Single model response - no consensus verification",
            }
        else:
            mode = {
                "mode": "error",
                "message": "No models available",
            }

        logger.info(
            "Degradation mode: %s (%d/%d models available)",
            mode["mode"],
            available_models,
            total_models,
        )
        return mode

    def apply_degradation(
        self, config: dict[str, Any], swarm_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Apply degradation settings to the swarm configuration.

        Args:
            config: Degradation mode config from determine_mode().
            swarm_config: Current swarm configuration to modify.

        Returns:
            Modified swarm configuration with degradation settings applied.
        """
        modified = dict(swarm_config)

        mode = config.get("mode", "error")

        if mode == "error":
            modified["enabled"] = False
            modified["error_message"] = config.get("message", "No models available")
            return modified

        modified["debate_rounds"] = config.get(
            "debate_rounds", modified.get("debate_rounds", 0)
        )
        modified["voting_strategy"] = config.get(
            "voting", modified.get("voting_strategy", "weighted")
        )

        if "caveat" in config:
            modified["caveat"] = config["caveat"]

        if mode == "single":
            modified["min_models"] = 1
            modified["require_consensus"] = False
        elif mode == "comparison":
            modified["min_models"] = 2
            modified["require_consensus"] = False
        elif mode == "simplified":
            modified["min_models"] = 3
            modified["require_consensus"] = True
        else:
            modified["require_consensus"] = True

        modified["degradation_mode"] = mode
        return modified
