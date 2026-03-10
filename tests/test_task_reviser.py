"""Tests for TaskReviser — mid-debate sub-query amendment."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.ai_congress.core.coordination.task_reviser import TaskReviser, RevisionSignal
from src.ai_congress.core.acp.run_context import ImplementationRun


class TestRevisionSignal:

    def test_should_revise_high_divergence(self):
        signal = RevisionSignal(
            divergence_score=0.8,
            coverage_score=0.9,
            avg_confidence=0.7,
            divergence_threshold=0.4,
            confidence_threshold=0.5,
        )
        assert signal.should_revise is True
        assert "divergence" in signal.reason.lower()

    def test_should_revise_low_coverage(self):
        signal = RevisionSignal(
            divergence_score=0.2,
            coverage_score=0.2,
            avg_confidence=0.7,
            divergence_threshold=0.4,
            confidence_threshold=0.5,
        )
        assert signal.should_revise is True
        assert "coverage" in signal.reason.lower()

    def test_should_revise_low_confidence(self):
        signal = RevisionSignal(
            divergence_score=0.2,
            coverage_score=0.8,
            avg_confidence=0.3,
            divergence_threshold=0.4,
            confidence_threshold=0.5,
        )
        assert signal.should_revise is True
        assert "confidence" in signal.reason.lower()

    def test_no_revision_needed(self):
        signal = RevisionSignal(
            divergence_score=0.2,
            coverage_score=0.8,
            avg_confidence=0.7,
            divergence_threshold=0.4,
            confidence_threshold=0.5,
        )
        assert signal.should_revise is False


class TestTaskReviser:

    def _make_run(self, sub_queries=None) -> ImplementationRun:
        run = ImplementationRun(query="What is the capital of France?")
        run.sub_queries = sub_queries or [
            {"text": "What is the capital city?", "source": "planner", "revision": 0},
            {"text": "What are key facts about it?", "source": "planner", "revision": 0},
        ]
        return run

    def _make_responses(self, texts: list[str]) -> list[dict]:
        models = ["phi3:3.8b", "mistral:7b", "llama3.2:3b"]
        return [
            {"model": models[i % len(models)], "response": text, "success": True}
            for i, text in enumerate(texts)
        ]

    def test_compute_coverage_high(self):
        reviser = TaskReviser(ollama_client=MagicMock())
        sub_queries = [
            {"text": "capital city", "source": "planner", "revision": 0},
        ]
        responses = self._make_responses(["The capital city of France is Paris."])
        coverage = reviser._compute_coverage(sub_queries, responses)
        assert coverage > 0.5

    def test_compute_coverage_low(self):
        reviser = TaskReviser(ollama_client=MagicMock())
        sub_queries = [
            {"text": "quantum entanglement effects", "source": "planner", "revision": 0},
        ]
        responses = self._make_responses(["The capital city of France is Paris."])
        coverage = reviser._compute_coverage(sub_queries, responses)
        assert coverage < 0.5

    def test_compute_divergence_identical(self):
        reviser = TaskReviser(ollama_client=MagicMock())
        responses = self._make_responses([
            "Paris is the capital of France.",
            "Paris is the capital of France.",
        ])
        divergence = reviser._compute_divergence(responses)
        assert divergence == 0.0

    def test_compute_divergence_different(self):
        reviser = TaskReviser(ollama_client=MagicMock())
        responses = self._make_responses([
            "Quantum mechanics describes subatomic particles.",
            "The Eiffel Tower is in Paris, a beautiful city.",
        ])
        divergence = reviser._compute_divergence(responses)
        assert divergence > 0.5

    @pytest.mark.asyncio
    async def test_assess_returns_signal(self):
        reviser = TaskReviser(ollama_client=MagicMock())
        run = self._make_run()
        responses = self._make_responses([
            "Paris is the capital of France.",
            "Paris is the capital of France, with the Eiffel Tower.",
            "France's capital is Paris.",
        ])
        signal = await reviser.assess(run, responses)
        assert isinstance(signal, RevisionSignal)

    @pytest.mark.asyncio
    async def test_revise_calls_planner(self):
        mock_client = AsyncMock()
        mock_client.chat.return_value = {
            "message": {"content": "- What is the official capital?\n- What makes it historically significant?"},
            "success": True,
        }
        reviser = TaskReviser(ollama_client=mock_client)
        run = self._make_run()
        signal = RevisionSignal(
            divergence_score=0.8,
            coverage_score=0.3,
            avg_confidence=0.4,
            divergence_threshold=0.4,
            confidence_threshold=0.5,
        )
        revised = await reviser.revise(run, signal, "phi3:3.8b")
        assert len(revised) == 2
        assert revised[0]["revision"] == 1
        assert "previous" in revised[0]

    @pytest.mark.asyncio
    async def test_revise_empty_response_keeps_originals(self):
        mock_client = AsyncMock()
        mock_client.chat.return_value = {
            "message": {"content": "No sub-questions needed."},
            "success": True,
        }
        reviser = TaskReviser(ollama_client=mock_client)
        run = self._make_run()
        original_sq = list(run.sub_queries)
        signal = RevisionSignal(
            divergence_score=0.8, coverage_score=0.3, avg_confidence=0.4,
            divergence_threshold=0.4, confidence_threshold=0.5,
        )
        result = await reviser.revise(run, signal, "phi3:3.8b")
        assert len(result) == len(original_sq)
        assert result[0]["text"] == original_sq[0]["text"]

    @pytest.mark.asyncio
    async def test_revise_preserves_content_with_leading_dash(self):
        mock_client = AsyncMock()
        mock_client.chat.return_value = {
            "message": {"content": "- -interesting topic about capitals\n- Another question"},
            "success": True,
        }
        reviser = TaskReviser(ollama_client=mock_client)
        run = self._make_run()
        signal = RevisionSignal(
            divergence_score=0.8, coverage_score=0.3, avg_confidence=0.4,
            divergence_threshold=0.4, confidence_threshold=0.5,
        )
        revised = await reviser.revise(run, signal, "phi3:3.8b")
        # Content starting with dash should be preserved after prefix removal
        assert revised[0]["text"] == "-interesting topic about capitals"

    def test_revisions_remaining(self):
        reviser = TaskReviser(ollama_client=MagicMock(), max_revisions=2)
        run = self._make_run()
        assert reviser.revisions_remaining(run) == 2

        run.sub_queries = [
            {"text": "q1", "source": "p", "revision": 1},
        ]
        assert reviser.revisions_remaining(run) == 1

    def test_revisions_remaining_exhausted(self):
        reviser = TaskReviser(ollama_client=MagicMock(), max_revisions=1)
        run = self._make_run()
        run.sub_queries = [
            {"text": "q1", "source": "p", "revision": 1},
        ]
        assert reviser.revisions_remaining(run) == 0
