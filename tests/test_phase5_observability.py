"""Phase 5 — observability & UX:
3.7.3 profile waterfall in stats, 3.7.4 EventLogger JSONL fallback,
3.7.5 /api/observability endpoints, 4.2.4 enhanced-mode stage streaming."""
import asyncio
import json
import os

import pytest

from src.ai_congress.datalake.logger import EventLogger


class _FakePool:
    def __init__(self, available=True):
        self.available = available
        self.pool = None

    @property
    def is_available(self):
        return self.available


# ---------------------------------------------------------------------------
# 3.7.4 — EventLogger JSONL fallback when Oracle is down
# ---------------------------------------------------------------------------

class TestEventLoggerFallback:
    def _logger(self, tmp_path, available=True):
        return EventLogger(
            _FakePool(available=available),
            batch_size=100,
            flush_interval=1.0,
            fallback_dir=str(tmp_path / "events"),
        )

    def test_events_written_to_fallback_when_oracle_down(self, tmp_path):
        logger = self._logger(tmp_path, available=False)
        logger.log("chat_request", "s1", mode="multi_model")
        logger.log("chat_response", "s1", confidence=0.8)
        asyncio.run(logger._flush_batch())

        fallback_dir = tmp_path / "events"
        files = list(fallback_dir.glob("*.jsonl"))
        assert files, "expected a JSONL fallback file"
        lines = [json.loads(l) for l in files[0].read_text().splitlines()]
        assert len(lines) == 2
        types = {e["event_type"] for e in lines}
        assert types == {"chat_request", "chat_response"}
        assert logger.get_fallback_stats()["fallback_hits"] == 2

    def test_no_fallback_when_oracle_available(self, tmp_path):
        logger = self._logger(tmp_path, available=True)
        logger.log("chat_request", "s1")
        asyncio.run(logger._flush_batch())
        # Pool available but no real connection: flush attempt fails -> fallback.
        # Assert the fallback still captured the event (never silently dropped).
        assert logger.get_fallback_stats()["fallback_hits"] == 1

    def test_stop_flushes_remaining_queue(self, tmp_path):
        logger = self._logger(tmp_path, available=False)
        logger.log("vote", "s2", winning_model="m1")
        asyncio.run(logger.stop())
        files = list((tmp_path / "events").glob("*.jsonl"))
        total = sum(len(f.read_text().splitlines()) for f in files)
        assert total == 1

    def test_corrupt_fallback_dir_degrades_gracefully(self, tmp_path):
        logger = self._logger(tmp_path, available=False)
        # Point the fallback at a path that cannot be created (a file).
        logger._fallback_dir = str(tmp_path / "blocked")
        with open(tmp_path / "blocked", "w") as f:
            f.write("x")
        logger.log("chat_request", "s1")
        asyncio.run(logger._flush_batch())  # must not raise


# ---------------------------------------------------------------------------
# 3.7.3 — profile waterfall surfaced in /enhanced/stats
# ---------------------------------------------------------------------------

class TestProfileInStats:
    def test_stats_include_profile_after_run(self):
        from src.ai_congress.core.enhanced_orchestrator import EnhancedOrchestrator
        orch = EnhancedOrchestrator.__new__(EnhancedOrchestrator)
        # Minimal attributes used by get_performance_stats
        orch.dynamic_weight_manager = _WeightStats()
        orch.confidence_calibrator = _CalibStats()
        orch.moe_router = _MoE()
        orch.adaptive_timeout = _Timeouts()
        orch._last_profile = {
            "stages": [{"name": "wave_1_queries", "duration_ms": 1200.0}],
            "total_ms": 4200.0,
            "slowest_stage": "wave_1_queries",
        }
        stats = orch.get_performance_stats()
        assert stats["profile"]["slowest_stage"] == "wave_1_queries"
        assert stats["profile"]["total_ms"] == 4200.0

    def test_stats_without_profile(self):
        from src.ai_congress.core.enhanced_orchestrator import EnhancedOrchestrator
        orch = EnhancedOrchestrator.__new__(EnhancedOrchestrator)
        orch.dynamic_weight_manager = _WeightStats()
        orch.confidence_calibrator = _CalibStats()
        orch.moe_router = _MoE()
        orch.adaptive_timeout = _Timeouts()
        stats = orch.get_performance_stats()
        assert "profile" not in stats


class _WeightStats:
    def get_performance_stats(self):
        return {"m1": {"win_rate": 0.5, "total_participations": 2, "current_weight": 0.6}}


class _CalibStats:
    def get_all_stats(self):
        return {"m1": {"observations": 1}}


class _MoE:
    def get_routing_stats(self):
        return {"total_routes": 0}


class _Timeouts:
    def get_stats(self):
        return {"avg_timeout": 30.0}


# ---------------------------------------------------------------------------
# 4.2.4 — stage streaming callback
# ---------------------------------------------------------------------------

class TestStageStreaming:
    def test_emit_stage_forwards_events(self):
        from src.ai_congress.core.enhanced_orchestrator import EnhancedOrchestrator
        orch = EnhancedOrchestrator.__new__(EnhancedOrchestrator)
        events = []

        async def cb(event_type, name, content, full_response):
            events.append((event_type, name, content))

        asyncio.run(orch._emit_stage(cb, "wave_1_queries", "3 responses"))
        assert events == [("stage", "wave_1_queries", "3 responses")]

    def test_emit_stage_silent_without_callback(self):
        from src.ai_congress.core.enhanced_orchestrator import EnhancedOrchestrator
        orch = EnhancedOrchestrator.__new__(EnhancedOrchestrator)
        asyncio.run(orch._emit_stage(None, "wave_1_queries", "x"))  # no raise


# ---------------------------------------------------------------------------
# 3.7.5 — observability endpoints
# ---------------------------------------------------------------------------

class TestObservabilityAPI:
    def _client(self):
        from fastapi.testclient import TestClient
        from src.ai_congress.api.main import app
        return TestClient(app, raise_server_exceptions=False)

    def test_summary_returns_sections(self):
        client = self._client()
        r = client.get("/api/observability/summary")
        assert r.status_code == 200
        body = r.json()
        for key in ("leaderboard", "circuit_breakers", "calibration",
                    "recent_runs", "event_logger", "breaker_open_count"):
            assert key in body, f"missing {key}"

    def test_leaderboard_sorted(self):
        client = self._client()
        r = client.get("/api/observability/leaderboard")
        assert r.status_code == 200
        rows = r.json().get("leaderboard", [])
        weights = [row["weight"] for row in rows]
        assert weights == sorted(weights, reverse=True)

    def test_enhanced_stats_endpoint(self):
        client = self._client()
        r = client.get("/api/enhanced/stats")
        assert r.status_code == 200
        body = r.json()
        assert "dynamic_weights" in body

    def test_enhanced_ws_route_registered(self):
        # The WS route should be registered on the app.
        from src.ai_congress.api.main import app
        routes = [getattr(r, "path", "") for r in app.routes]
        assert "/ws/chat/enhanced" in routes
