"""Tests for the ELO Performance Tracker."""

import json
import os
import tempfile

import pytest

from src.ai_congress.core.learning.elo_tracker import ELOTracker


@pytest.fixture
def tmp_data_path(tmp_path):
    """Return a temporary file path for ELO data."""
    return str(tmp_path / "elo_ratings.json")


@pytest.fixture
def tracker(tmp_data_path):
    """Create a fresh ELOTracker with a temp data file."""
    return ELOTracker(data_path=tmp_data_path)


# ------------------------------------------------------------------
# Basic ELO math
# ------------------------------------------------------------------


class TestBasicELO:
    def test_equal_ratings_one_wins(self, tracker):
        """When two models start at equal ratings, the winner should gain ~16 points."""
        deltas = tracker.record_match("alpha", ["alpha", "beta"])

        # Winner gains, loser loses by the same absolute amount
        assert deltas["alpha"] > 0
        assert deltas["beta"] < 0
        assert abs(deltas["alpha"] + deltas["beta"]) < 1e-9

        # With K=32 and equal ratings, expected score is 0.5, so delta = 32*(1-0.5) = 16
        assert abs(deltas["alpha"] - 16.0) < 1e-9

    def test_strong_beats_weak_small_delta(self, tracker):
        """When a strong model beats a weak one, the delta should be small."""
        # Manually set up unequal ratings
        tracker.ratings["strong"] = 1600
        tracker.ratings["weak"] = 1000

        deltas = tracker.record_match("strong", ["strong", "weak"])

        # Strong was expected to win, so the gain is small
        assert deltas["strong"] > 0
        assert deltas["strong"] < 5  # Much less than 16

    def test_weak_beats_strong_large_delta(self, tracker):
        """When a weak model beats a strong one, the delta should be large."""
        tracker.ratings["strong"] = 1600
        tracker.ratings["weak"] = 1000

        deltas = tracker.record_match("weak", ["strong", "weak"])

        # Weak wasn't expected to win, so the gain is large
        assert deltas["weak"] > 25  # Much more than 16

    def test_winner_must_be_participant(self, tracker):
        with pytest.raises(ValueError, match="must be in participants"):
            tracker.record_match("ghost", ["alpha", "beta"])

    def test_need_at_least_two_participants(self, tracker):
        with pytest.raises(ValueError, match="at least 2"):
            tracker.record_match("alpha", ["alpha"])


# ------------------------------------------------------------------
# Multi-player
# ------------------------------------------------------------------


class TestMultiPlayer:
    def test_multi_player_match(self, tracker):
        """A match with 3+ participants: winner plays N-1 virtual 1v1s."""
        participants = ["a", "b", "c"]
        deltas = tracker.record_match("a", participants)

        # Winner gains from two matches
        assert deltas["a"] > 0
        # Each loser loses from one match
        assert deltas["b"] < 0
        assert deltas["c"] < 0

        # Winner's gain should be roughly 2 * 16 = 32 (not exact because
        # ratings shift between each virtual 1v1 within the same match)
        assert 30.0 < deltas["a"] < 34.0

    def test_four_player_match(self, tracker):
        """Four participants: winner plays 3 virtual 1v1s."""
        participants = ["a", "b", "c", "d"]
        deltas = tracker.record_match("a", participants)

        assert deltas["a"] > 0
        # Roughly 3 * 16 = 48, with some drift from sequential rating updates
        assert 44.0 < deltas["a"] < 50.0


# ------------------------------------------------------------------
# Leaderboard
# ------------------------------------------------------------------


class TestLeaderboard:
    def test_leaderboard_sorted_descending(self, tracker):
        """Leaderboard should be sorted by ELO descending."""
        tracker.record_match("a", ["a", "b"])
        tracker.record_match("a", ["a", "c"])
        tracker.record_match("b", ["b", "c"])

        board = tracker.get_leaderboard()
        elos = [entry["elo"] for entry in board]
        assert elos == sorted(elos, reverse=True)

    def test_leaderboard_fields(self, tracker):
        """Each leaderboard entry should have the right fields."""
        tracker.record_match("a", ["a", "b"])
        board = tracker.get_leaderboard()

        for entry in board:
            assert "model" in entry
            assert "elo" in entry
            assert "wins" in entry
            assert "matches" in entry
            assert "win_rate" in entry

    def test_leaderboard_win_rate(self, tracker):
        """Win rate should match wins / matches."""
        tracker.record_match("a", ["a", "b"])
        tracker.record_match("b", ["a", "b"])
        tracker.record_match("a", ["a", "b"])

        board = tracker.get_leaderboard()
        a_entry = next(e for e in board if e["model"] == "a")
        assert a_entry["wins"] == 2
        assert a_entry["matches"] == 3
        assert abs(a_entry["win_rate"] - 2.0 / 3.0) < 1e-3


# ------------------------------------------------------------------
# Model trend
# ------------------------------------------------------------------


class TestModelTrend:
    def test_trend_accuracy(self, tracker):
        """Trend should reflect the ELO snapshots after each match."""
        tracker.record_match("a", ["a", "b"])
        tracker.record_match("b", ["a", "b"])
        tracker.record_match("a", ["a", "b"])

        trend = tracker.get_model_trend("a")
        assert len(trend) == 3
        # First match: a wins from 1200, gains 16 -> 1216
        assert abs(trend[0] - 1216.0) < 1e-9

    def test_trend_last_n(self, tracker):
        """last_n should limit the returned snapshots."""
        for _ in range(10):
            tracker.record_match("a", ["a", "b"])

        trend = tracker.get_model_trend("a", last_n=3)
        assert len(trend) == 3

    def test_trend_unknown_model(self, tracker):
        """Unknown model returns empty list."""
        trend = tracker.get_model_trend("nonexistent")
        assert trend == []


# ------------------------------------------------------------------
# Persistence
# ------------------------------------------------------------------


class TestPersistence:
    def test_save_and_load(self, tmp_data_path):
        """Data should survive save + load cycle."""
        t1 = ELOTracker(data_path=tmp_data_path)
        t1.record_match("a", ["a", "b"])
        rating_a = t1.ratings["a"]

        # Create a new tracker that loads from the same file
        t2 = ELOTracker(data_path=tmp_data_path)
        assert t2.ratings["a"] == rating_a
        assert len(t2.match_history) == 1

    def test_atomic_write_creates_dir(self, tmp_path):
        """If the data directory doesn't exist, _save should create it."""
        nested = str(tmp_path / "deep" / "nested" / "elo.json")
        t = ELOTracker(data_path=nested)
        t.record_match("a", ["a", "b"])

        assert os.path.exists(nested)

    def test_corrupted_file_handled(self, tmp_data_path):
        """A corrupted JSON file should not crash the tracker."""
        os.makedirs(os.path.dirname(tmp_data_path), exist_ok=True)
        with open(tmp_data_path, "w") as f:
            f.write("NOT VALID JSON {{{")

        t = ELOTracker(data_path=tmp_data_path)
        assert t.ratings == {}
        assert t.match_history == []


# ------------------------------------------------------------------
# History cap
# ------------------------------------------------------------------


class TestHistoryCap:
    def test_cap_at_10000(self, tmp_data_path):
        """Match history should never exceed MAX_HISTORY entries."""
        from unittest.mock import patch
        from src.ai_congress.core.learning import elo_tracker as elo_mod

        # Temporarily lower cap to avoid slow disk writes in tests
        original_cap = elo_mod.MAX_HISTORY
        elo_mod.MAX_HISTORY = 100
        try:
            t = ELOTracker(data_path=tmp_data_path)

            # Pre-load history just under the cap
            t.match_history = [
                {"winner": "a", "participants": ["a", "b"]}
            ] * 99
            t.record_match("a", ["a", "b"])
            assert len(t.match_history) == 100

            # One more should still stay at cap
            t.record_match("a", ["a", "b"])
            assert len(t.match_history) == 100
        finally:
            elo_mod.MAX_HISTORY = original_cap


# ------------------------------------------------------------------
# Domain-specific ratings
# ------------------------------------------------------------------


class TestDomainRatings:
    def test_domain_tracking(self, tracker):
        """Domain-specific ratings should be tracked separately."""
        tracker.record_match("a", ["a", "b"], domain="math")
        tracker.record_match("b", ["a", "b"], domain="writing")

        stats = tracker.get_stats()
        assert "math" in stats["domain_ratings"]
        assert "writing" in stats["domain_ratings"]

        # a should be higher in math, b higher in writing
        assert stats["domain_ratings"]["math"]["a"] > stats["domain_ratings"]["math"]["b"]
        assert stats["domain_ratings"]["writing"]["b"] > stats["domain_ratings"]["writing"]["a"]

    def test_domain_independent_of_global(self, tracker):
        """Domain ratings should change independently of global ratings."""
        tracker.record_match("a", ["a", "b"], domain="code")

        global_a = tracker.ratings["a"]
        domain_a = tracker.domain_ratings["code"]["a"]

        # Both should have moved from 1200, and by the same amount
        # (since they started equal), but they're tracked in separate dicts
        assert global_a == domain_a
        assert global_a != 1200

    def test_get_stats_includes_domains(self, tracker):
        """get_stats should include domain_ratings."""
        tracker.record_match("a", ["a", "b"], domain="science")
        stats = tracker.get_stats()

        assert "domain_ratings" in stats
        assert "science" in stats["domain_ratings"]
        assert "ratings" in stats
        assert "match_count" in stats
        assert stats["match_count"] == 1
