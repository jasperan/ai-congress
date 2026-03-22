"""
ELO Performance Tracker for AI Congress models.

Tracks model performance across voting sessions using standard ELO ratings.
Thread-safe, persists to disk with atomic writes, and caps history at 10k entries.
"""

import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional


# Defaults
DEFAULT_K = 32
DEFAULT_RATING = 1200
MAX_HISTORY = 10_000
DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "data", "elo_ratings.json"
)


class ELOTracker:
    """Tracks ELO ratings for models participating in AI Congress voting sessions."""

    def __init__(
        self,
        data_path: Optional[str] = None,
        k_factor: int = DEFAULT_K,
        default_rating: float = DEFAULT_RATING,
    ):
        self.data_path = os.path.abspath(data_path or DEFAULT_DATA_PATH)
        self.k_factor = k_factor
        self.default_rating = default_rating
        self._lock = threading.RLock()

        # Internal state
        self.ratings: Dict[str, float] = {}
        self.domain_ratings: Dict[str, Dict[str, float]] = {}
        self.match_history: List[dict] = []

        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_match(
        self,
        winner: str,
        participants: List[str],
        domain: str = "general",
    ) -> Dict[str, float]:
        """Record a match outcome and return ELO deltas for each participant.

        The winner gains rating points; all other participants lose points.
        For N participants, the winner plays (N-1) virtual 1v1 matches against
        each loser, and each loser plays one virtual match against the winner.
        """
        if winner not in participants:
            raise ValueError(f"Winner '{winner}' must be in participants list")
        if len(participants) < 2:
            raise ValueError("Need at least 2 participants")

        with self._lock:
            # Ensure all participants have ratings
            for p in participants:
                if p not in self.ratings:
                    self.ratings[p] = self.default_rating
                if domain not in self.domain_ratings:
                    self.domain_ratings[domain] = {}
                if p not in self.domain_ratings[domain]:
                    self.domain_ratings[domain][p] = self.default_rating

            deltas: Dict[str, float] = {p: 0.0 for p in participants}
            losers = [p for p in participants if p != winner]

            for loser in losers:
                # Global ratings
                g_delta_w, g_delta_l = self._compute_elo_delta(
                    self.ratings[winner], self.ratings[loser]
                )
                self.ratings[winner] += g_delta_w
                self.ratings[loser] += g_delta_l
                deltas[winner] += g_delta_w
                deltas[loser] += g_delta_l

                # Domain ratings
                d_delta_w, d_delta_l = self._compute_elo_delta(
                    self.domain_ratings[domain][winner],
                    self.domain_ratings[domain][loser],
                )
                self.domain_ratings[domain][winner] += d_delta_w
                self.domain_ratings[domain][loser] += d_delta_l

            # Record history entry
            entry = {
                "winner": winner,
                "participants": participants,
                "domain": domain,
                "deltas": deltas,
                "ratings_snapshot": {p: self.ratings[p] for p in participants},
                "timestamp": time.time(),
            }
            self.match_history.append(entry)

            # Cap history
            if len(self.match_history) > MAX_HISTORY:
                self.match_history = self.match_history[-MAX_HISTORY:]

            self._save()

        return deltas

    def get_leaderboard(self) -> List[dict]:
        """Return models sorted by ELO descending."""
        with self._lock:
            board = []
            for model, rating in self.ratings.items():
                wins = sum(
                    1 for m in self.match_history if m["winner"] == model
                )
                total = sum(
                    1 for m in self.match_history if model in m["participants"]
                )
                board.append({
                    "model": model,
                    "elo": round(rating, 2),
                    "wins": wins,
                    "matches": total,
                    "win_rate": round(wins / total, 4) if total > 0 else 0.0,
                })
            board.sort(key=lambda x: x["elo"], reverse=True)
            return board

    def get_model_trend(self, model: str, last_n: int = 50) -> List[float]:
        """Return the last N ELO snapshots for a model (oldest first)."""
        with self._lock:
            snapshots = []
            for entry in self.match_history:
                if model in entry.get("ratings_snapshot", {}):
                    snapshots.append(entry["ratings_snapshot"][model])
            return snapshots[-last_n:]

    def get_stats(self) -> dict:
        """Return full statistics: ratings, domain ratings, match count."""
        with self._lock:
            return {
                "ratings": dict(self.ratings),
                "domain_ratings": {
                    d: dict(r) for d, r in self.domain_ratings.items()
                },
                "match_count": len(self.match_history),
                "leaderboard": self.get_leaderboard(),
            }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load ratings from disk. No-op if file doesn't exist."""
        if not os.path.exists(self.data_path):
            return
        try:
            with open(self.data_path, "r") as f:
                data = json.load(f)
            self.ratings = data.get("ratings", {})
            self.domain_ratings = data.get("domain_ratings", {})
            self.match_history = data.get("match_history", [])
        except (json.JSONDecodeError, OSError):
            # Corrupted file; start fresh
            pass

    def _save(self) -> None:
        """Persist ratings to disk with atomic write (tempfile + rename)."""
        data = {
            "ratings": self.ratings,
            "domain_ratings": self.domain_ratings,
            "match_history": self.match_history,
        }
        dir_path = os.path.dirname(self.data_path)
        os.makedirs(dir_path, exist_ok=True)

        fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, self.data_path)
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------
    # ELO math
    # ------------------------------------------------------------------

    def _compute_elo_delta(
        self, rating_winner: float, rating_loser: float
    ) -> tuple:
        """Compute ELO deltas for a single 1v1 outcome.

        Returns (delta_winner, delta_loser).
        """
        expected_winner = 1.0 / (
            1.0 + 10.0 ** ((rating_loser - rating_winner) / 400.0)
        )
        expected_loser = 1.0 - expected_winner

        delta_winner = self.k_factor * (1.0 - expected_winner)
        delta_loser = self.k_factor * (0.0 - expected_loser)

        return delta_winner, delta_loser
