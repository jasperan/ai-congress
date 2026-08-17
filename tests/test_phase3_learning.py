"""Phase 3 — learning persistence + model-agnostic bootstrap:
3.5.1 live-catalog bootstrap, 3.5.2 persisted dynamic weights + calibration,
3.5.3 feedback->weight bridge, 3.4.3 persisted circuit-breaker state."""
import json
import os

import pytest

from src.ai_congress.core.coordination.circuit_breaker import (
    STATE_OPEN,
    CircuitBreaker,
)
from src.ai_congress.core.learning.dynamic_weights import DynamicWeightManager
from src.ai_congress.core.learning.feedback_loop import FeedbackCollector
from src.ai_congress.core.voting.confidence_calibrator import ConfidenceCalibrator


# ---------------------------------------------------------------------------
# 3.5.2 — dynamic weights persist across instances
# ---------------------------------------------------------------------------

class TestDynamicWeightPersistence:
    def test_update_weights_writes_state_file(self, tmp_path):
        path = str(tmp_path / "weights.json")
        mgr = DynamicWeightManager(base_weights={"m1": 0.5}, persist_path=path)
        mgr.record_outcome("m1", was_winner=True)
        mgr.update_weights()
        data = json.loads(open(path).read())
        assert data["outcomes"]["m1"]["wins"] == 1
        assert "m1" in data["current_weights"]

    def test_reload_restores_learned_weights(self, tmp_path):
        path = str(tmp_path / "weights.json")
        mgr = DynamicWeightManager(base_weights={"m1": 0.5}, persist_path=path)
        mgr.record_outcome("m1", was_winner=True)
        # Force accumulated wins so EMA moves the weight away from base
        for _ in range(10):
            mgr.record_outcome("m1", was_winner=True)
        mgr.update_weights()
        learned = mgr.get_weight("m1")
        assert learned > 0.5

        # New instance, same path — must restore the learned weight
        reloaded = DynamicWeightManager(base_weights={"m1": 0.5}, persist_path=path)
        assert reloaded.get_weight("m1") == pytest.approx(learned)
        assert reloaded.get_performance_stats()["m1"]["total_participations"] == 11

    def test_missing_file_returns_defaults(self, tmp_path):
        mgr = DynamicWeightManager(
            base_weights={"m1": 0.5},
            persist_path=str(tmp_path / "does-not-exist.json"),
        )
        assert mgr.get_weight("m1") == 0.5


# ---------------------------------------------------------------------------
# 3.5.2 — confidence calibrator persists
# ---------------------------------------------------------------------------

class TestCalibratorPersistence:
    def test_calibration_survives_restart(self, tmp_path):
        path = str(tmp_path / "calibration.json")
        cal = ConfidenceCalibrator(persist_path=path)
        for _ in range(10):
            cal.record("m1", predicted_confidence=0.9, was_correct=True)
        assert os.path.exists(path)

        reloaded = ConfidenceCalibrator(persist_path=path)
        # Bin 9 accuracy is 1.0 after 10 correct high-confidence records
        assert reloaded._data["m1"][9] == {"correct": 10, "total": 10}


# ---------------------------------------------------------------------------
# 3.5.3 — feedback bridge into weight updates + domain tagging
# ---------------------------------------------------------------------------

class TestFeedbackBridge:
    def test_positive_feedback_raises_weight(self, tmp_path):
        path = str(tmp_path / "weights.json")
        mgr = DynamicWeightManager(base_weights={"m1": 0.5}, persist_path=path)
        before = mgr.get_weight("m1")
        mgr.apply_feedback("m1", positive=True)
        assert mgr.get_weight("m1") > before

    def test_negative_feedback_lowers_weight(self, tmp_path):
        path = str(tmp_path / "w.json")
        mgr = DynamicWeightManager(base_weights={"m1": 0.9}, persist_path=path)
        before = mgr.get_weight("m1")
        mgr.apply_feedback("m1", positive=False)
        assert mgr.get_weight("m1") < before

    def test_feedback_collector_persists_with_domain(self, tmp_path):
        path = str(tmp_path / "feedback.json")
        fc = FeedbackCollector(persist_path=path)
        fc.record_feedback("sess-1", "m1", "positive", "great answer", domain="factual")
        fc.record_feedback("sess-1", "m1", "negative", "bad answer", domain="analytical")

        reloaded = FeedbackCollector(persist_path=path)
        entries = reloaded.get_all_feedback()
        assert len(entries) == 2
        assert entries[0]["domain"] == "factual"
        assert entries[1]["domain"] == "analytical"
        stats = reloaded.get_model_feedback_stats("m1")
        assert stats == {"positive": 1, "negative": 1, "approval_rate": 0.5}


# ---------------------------------------------------------------------------
# 3.4.3 — circuit-breaker state persists across restarts
# ---------------------------------------------------------------------------

class TestCircuitBreakerPersistence:
    def test_open_breaker_restored_after_restart(self, tmp_path):
        path = str(tmp_path / "breaker.json")
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=3600, persist_path=path)
        cb.record_failure("bad-model")
        cb.record_failure("bad-model")
        assert cb.get_state("bad-model") == STATE_OPEN
        assert os.path.exists(path)

        # A fresh instance must still see the OPEN breaker (no retry storm).
        reloaded = CircuitBreaker(failure_threshold=2, recovery_timeout=3600, persist_path=path)
        assert reloaded.get_state("bad-model") == STATE_OPEN
        # Within the recovery window requests are blocked.
        assert reloaded.can_execute("bad-model") is False

    def test_success_persists_closed_state(self, tmp_path):
        path = str(tmp_path / "breaker.json")
        cb = CircuitBreaker(persist_path=path)
        cb.record_failure("m2")
        cb.record_failure("m2")
        cb.record_failure("m2")
        assert cb.get_state("m2") == STATE_OPEN
        cb.record_success("m2")

        reloaded = CircuitBreaker(persist_path=path)
        assert reloaded.get_state("m2") == "CLOSED"

    def test_corrupt_file_degrades_gracefully(self, tmp_path):
        path = str(tmp_path / "breaker.json")
        with open(path, "w") as f:
            f.write("{not json!!")
        cb = CircuitBreaker(persist_path=path)
        assert cb.get_state("anything") == "CLOSED"


# ---------------------------------------------------------------------------
# 3.5.1 — model-agnostic bootstrap from the live catalog
# ---------------------------------------------------------------------------

class _FakeOllamaClient:
    def __init__(self, models):
        self._models = models

    async def list_models(self):
        return self._models


class TestCatalogBootstrap:
    def _registry(self, ollama_models):
        from src.ai_congress.core.model_registry import ModelRegistry
        from src.ai_congress.utils.config_loader import OllamaConfig
        r = ModelRegistry(OllamaConfig())
        r.ollama_client = _FakeOllamaClient(ollama_models)
        return r

    @pytest.mark.asyncio
    async def test_live_models_get_neutral_weight(self):
        r = self._registry([
            {"name": "qwen3.5:9b", "model": "qwen3.5:9b", "size": 1, "digest": "x"},
            {"name": "gemma3:4b", "model": "gemma3:4b", "size": 1, "digest": "y"},
        ])
        r.weights = {"stale-ghost": 0.98}  # not in the catalog
        loaded = await r.list_available_models()
        assert len(loaded) == 2
        assert r.get_model_weight("qwen3.5:9b") == 0.5  # neutral bootstrap
        assert r.get_model_weight("gemma3:4b") == 0.5
        # Stale key purged so learning starts on real models
        assert "stale-ghost" not in r.weights

    @pytest.mark.asyncio
    async def test_benchmark_applies_only_to_installed_models(self, tmp_path):
        r = self._registry([
            {"name": "qwen3.5:9b", "model": "qwen3.5:9b", "size": 1, "digest": "x"},
        ])
        await r.list_available_models()
        bench = tmp_path / "bench.json"
        bench.write_text(json.dumps({
            "qwen3.5:9b": {"accuracy": 0.9},
            "deepseek-r1:671b": {"accuracy": 0.98},  # not installed
        }))
        await r.load_benchmark_weights(str(bench))
        assert r.get_model_weight("qwen3.5:9b") == pytest.approx(0.9)
        assert r.get_model_weight("deepseek-r1:671b") == 0.5  # default, not ghost weight

    @pytest.mark.asyncio
    async def test_empty_catalog_does_not_wipe_weights(self):
        r = self._registry([])
        r.weights = {"qwen3.5:9b": 0.7}
        await r.list_available_models()
        # Ollama outage: weights preserved
        assert r.get_model_weight("qwen3.5:9b") == 0.7