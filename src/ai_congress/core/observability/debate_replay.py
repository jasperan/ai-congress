"""Debate state persistence and replay functionality."""

import json
import logging
import os
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


class DebateReplayManager:
    """Saves and replays debate sessions for analysis and debugging.

    Stores debate artifacts as JSON files with round-by-round snapshots.
    """

    def __init__(self, storage_dir: str = "data/debate_replays") -> None:
        """Initialize the debate replay manager.

        Args:
            storage_dir: Directory for storing debate replay JSON files.
        """
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    def _session_path(self, session_id: str) -> str:
        """Get the file path for a session's debate file."""
        safe_id = session_id.replace("/", "_").replace(":", "_")
        return os.path.join(self.storage_dir, f"{safe_id}.json")

    def save_debate(self, session_id: str, artifact: dict[str, Any]) -> None:
        """Save a debate artifact to disk.

        Args:
            session_id: Unique session identifier.
            artifact: Full debate artifact dict (rounds, votes, responses, etc.).
        """
        path = self._session_path(session_id)
        data = {
            "session_id": session_id,
            "saved_at": time.time(),
            "artifact": artifact,
        }
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
            logger.info("Saved debate replay for session %s", session_id)
        except OSError as e:
            logger.error("Failed to save debate for %s: %s", session_id, e)

    def load_debate(self, session_id: str) -> dict[str, Any]:
        """Load a saved debate artifact.

        Args:
            session_id: The session identifier.

        Returns:
            The saved debate data dict, or empty dict if not found.
        """
        path = self._session_path(session_id)
        if not os.path.exists(path):
            logger.warning("No debate replay found for session %s", session_id)
            return {}

        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Failed to load debate for %s: %s", session_id, e)
            return {}

    def list_debates(self, limit: int = 50) -> list[dict[str, Any]]:
        """List recent saved debates with metadata.

        Args:
            limit: Maximum number of debates to return.

        Returns:
            List of debate metadata dicts, sorted by save time (newest first).
        """
        debates: list[dict[str, Any]] = []
        if not os.path.isdir(self.storage_dir):
            return debates

        for filename in os.listdir(self.storage_dir):
            if not filename.endswith(".json"):
                continue
            path = os.path.join(self.storage_dir, filename)
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                artifact = data.get("artifact", {})
                debates.append({
                    "session_id": data.get("session_id", filename.replace(".json", "")),
                    "saved_at": data.get("saved_at", 0),
                    "num_rounds": len(artifact.get("rounds", [])),
                    "num_models": len(artifact.get("models", [])),
                    "query": artifact.get("query", artifact.get("prompt", ""))[:100],
                })
            except (json.JSONDecodeError, OSError):
                continue

        debates.sort(key=lambda d: d.get("saved_at", 0), reverse=True)
        return debates[:limit]

    def get_round_snapshot(
        self, session_id: str, round_num: int
    ) -> dict[str, Any]:
        """Get the state snapshot for a specific debate round.

        Args:
            session_id: The session identifier.
            round_num: Zero-indexed round number.

        Returns:
            Round state dict, or empty dict if not found.
        """
        data = self.load_debate(session_id)
        if not data:
            return {}

        artifact = data.get("artifact", {})
        rounds = artifact.get("rounds", [])

        if 0 <= round_num < len(rounds):
            return rounds[round_num]

        logger.warning(
            "Round %d not found in session %s (has %d rounds)",
            round_num,
            session_id,
            len(rounds),
        )
        return {}

    def format_replay_timeline(self, artifact: dict[str, Any]) -> list[dict[str, Any]]:
        """Format a debate artifact as a chronological timeline of events.

        Args:
            artifact: The debate artifact dict.

        Returns:
            List of timeline event dicts with type, description, and data.
        """
        timeline: list[dict[str, Any]] = []

        # Query event
        query = artifact.get("query", artifact.get("prompt", ""))
        if query:
            timeline.append({
                "type": "query",
                "description": f"Query submitted: {query[:100]}",
                "data": {"query": query},
            })

        # Round events
        for i, round_data in enumerate(artifact.get("rounds", [])):
            responses = round_data.get("responses", [])
            model_names = [r.get("model", "unknown") for r in responses]
            timeline.append({
                "type": "round",
                "description": f"Round {i + 1}: {len(responses)} responses from {', '.join(model_names)}",
                "data": round_data,
            })

        # Vote event
        vote_result = artifact.get("vote_result", artifact.get("result", {}))
        if vote_result:
            winner = vote_result.get("winner", vote_result.get("selected_model", ""))
            timeline.append({
                "type": "vote",
                "description": f"Vote completed: winner = {winner}",
                "data": vote_result,
            })

        return timeline
