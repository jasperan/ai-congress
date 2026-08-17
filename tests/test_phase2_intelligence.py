"""Phase 2 — dormant intelligence wiring: 3.1.1 reasoning-mode prompts,
3.1.4 AgentMemory into the enhanced pipeline, 3.1.6 self-consistency gating,
3.3.3 evidence-grounded deliberation, 3.5.5 prompt-evolution A/B templates."""
import asyncio

import pytest

from src.ai_congress.core.deliberation import DeliberationConfig, DeliberationOrchestrator
from src.ai_congress.core.enhanced_pipeline import run_initial_response_wave
from src.ai_congress.core.intelligence.agent_memory import AgentMemory
from src.ai_congress.core.intelligence.role_prompts import (
    MODE_INSTRUCTIONS,
    get_mode_instruction,
)
from src.ai_congress.core.learning.prompt_evolution import PromptEvolution
from src.ai_congress.core.precedent.precedent_injector import PrecedentAction
from src.ai_congress.core.acp.supervisor import SupervisedTask, RestartPolicy


# ---------------------------------------------------------------------------
# 3.1.1 — reasoning_mode actually changes the prompt
# ---------------------------------------------------------------------------

class TestModeInstructions:
    def test_map_has_all_modes(self):
        assert set(MODE_INSTRUCTIONS) == {"direct", "cot", "react"}

    def test_cot_instruction(self):
        assert "step by step" in get_mode_instruction("cot").lower()

    def test_react_instruction(self):
        assert "tools" in get_mode_instruction("react").lower()

    def test_direct_and_unknown_are_empty(self):
        assert get_mode_instruction("direct") == ""
        assert get_mode_instruction("not-a-mode") == ""


class _StubTaskResult:
    def __init__(self, agent_id, result, success=True):
        self.agent_id = agent_id
        self.result = result
        self.success = success


class _FakeSupervisor:
    """Records task args; runs coro factories directly."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.calls = []

    async def supervise_all(self, tasks):
        results = []
        for t in tasks:
            try:
                result = await t.coro_factory(*t.args, **t.kwargs)
                self.calls.append({
                    "agent_id": t.agent_id,
                    "prompt": t.args[1],
                    "system_prompt": t.kwargs.get("system_prompt", ""),
                    "result": result,
                })
                results.append(_StubTaskResult(t.agent_id, result))
            except Exception as e:
                results.append(_StubTaskResult(t.agent_id, None, success=False))
        return results


class _StubRuntime:
    """Minimal runtime surface used by run_initial_response_wave."""

    def __init__(self, response_text="model answer"):
        self.goal_engine = None
        self.response_text = response_text
        self.supervisor = _FakeSupervisor(self)
        self.circuit_breaker = _StubCircuitBreaker()

        class _Timeout:
            def get_timeout(self, model):
                return 30.0
            def record_latency(self, model, ms):
                pass

        class _Adaptive:
            def __init__(self):
                self.adaptive = _Timeout()
            def get_timeout(self, model):
                return self.adaptive.get_timeout(model)
            def record_latency(self, model, ms):
                pass

        self.adaptive_timeout = _Adaptive()

    async def _throttled_query(self, model, prompt, temperature, system_prompt="", timeout=60.0, **kw):
        return {
            "model": model,
            "response": f"{self.response_text}-{model}",
            "temperature": temperature,
            "success": True,
            "latency_ms": 5.0,
        }


class _StubCircuitBreaker:
    def record_success(self, model):
        pass
    def record_failure(self, model):
        pass


def _make_run(query="test question"):
    from src.ai_congress.core.acp.run_context import ImplementationRun
    return ImplementationRun(query=query)


class TestInitialResponseWave:
    def _wave(self, runtime, run, **kwargs):
        from src.ai_congress.core.acp.roles import RoleAssignment, AgentRole
        defaults = dict(
            runtime=runtime, run=run, available_models=["m1"],
            role_assignments={AgentRole.WORKER: [RoleAssignment(model_name="m1", role=AgentRole.WORKER, score=0.5)]},
            effective_prompt="the base prompt", temperature=0.7,
            precedent_action=PrecedentAction.NO_PRECEDENT, cited_precedents=[],
            reasoning_mode="direct", memory_context="",
        )
        defaults.update(kwargs)
        return asyncio.run(run_initial_response_wave(**defaults))

    def test_cot_mode_instruction_appended(self):
        runtime = _StubRuntime()
        run = _make_run()
        self._wave(runtime, run, reasoning_mode="cot")
        prompt = runtime.supervisor.calls[0]["prompt"]
        assert prompt.startswith("the base prompt")
        assert "Think step by step and show your work." in prompt

    def test_direct_mode_no_instruction(self):
        runtime = _StubRuntime()
        run = _make_run()
        self._wave(runtime, run, reasoning_mode="direct")
        prompt = runtime.supervisor.calls[0]["prompt"]
        assert prompt == "the base prompt"


    def test_memory_context_injected_into_wave(self):
        runtime = _StubRuntime()
        run = _make_run()
        memory = AgentMemory()
        memory.add_exchange("What is the capital of France?", "Paris is the capital.")
        ctx = memory.build_memory_context("capital of France")
        self._wave(runtime, run, memory_context=ctx)
        prompt = runtime.supervisor.calls[0]["prompt"]
        assert "Paris is the capital." in prompt


# ---------------------------------------------------------------------------
# 3.1.4 — AgentMemory wired into the enhanced pipeline
# ---------------------------------------------------------------------------

class TestAgentMemoryRecall:
    def test_add_and_recall(self):
        m = AgentMemory()
        m.add_exchange("What is the capital of France?", "Paris is the capital.")
        recalled = m.recall_relevant("capital of France", top_k=1)
        assert recalled and recalled[0]["response"] == "Paris is the capital."
        assert recalled[0]["source"] in ("short_term", "long_term")

    def test_build_memory_context_format(self):
        m = AgentMemory()
        m.add_exchange("Explain transformers", "Attention is all you need.")
        ctx = m.build_memory_context("explain transformers")
        assert "Relevant past exchanges" in ctx
        assert "Attention is all you need" in ctx

    def test_no_relevant_memory_returns_empty(self):
        m = AgentMemory()
        assert m.build_memory_context("totally unrelated topic") == ""


# ---------------------------------------------------------------------------
# 3.1.6 — self-consistency gating
# ---------------------------------------------------------------------------

class TestSelfConsistency:
    def _make_orchestrator(self, monkeypatch, min_agreement=0.5, samples=2):
        import tempfile, os, json
        from src.ai_congress.utils.config_loader import (
            IntelligenceConfig, OllamaConfig,
        )
        from src.ai_congress.core.model_registry import ModelRegistry
        from src.ai_congress.core.voting_engine import VotingEngine
        from src.ai_congress.core.enhanced_orchestrator import EnhancedOrchestrator
        from src.ai_congress.core.personality.profile import ModelPersonalityLoader

        cfg = IntelligenceConfig()
        cfg.self_consistency.min_agreement = min_agreement
        cfg.self_consistency.samples = samples
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        with open(path, "w") as f:
            f.write(json.dumps({}))
        try:
            orch = EnhancedOrchestrator(
                ModelRegistry(OllamaConfig()),
                VotingEngine(),
                _StubRuntime(),
                ModelPersonalityLoader(path),
            )
        finally:
            os.unlink(path)
        orch.intelligence = cfg
        return orch

    def test_skipped_when_agreement_high(self, monkeypatch):
        orch = self._make_orchestrator(monkeypatch, min_agreement=0.5)
        votes = {"winner": "A", "confidence": 0.8, "agreement_ratio": 0.9}

        async def fake_query(*a, **k):
            raise AssertionError("should not resample on high agreement")

        monkeypatch.setattr(orch, "_query_model", fake_query)
        w, c, v = asyncio.run(orch._self_consistency_pass(
            ["m1"], {"m1": 0.5}, "prompt", "A", 0.8, votes, _make_run()))
        assert (w, c) == ("A", 0.8)
        assert v is votes

    def test_disabled_never_resamples(self, monkeypatch):
        orch = self._make_orchestrator(monkeypatch, min_agreement=0.99)
        orch.intelligence.self_consistency.enabled = False

        async def fake_query(*a, **k):
            raise AssertionError("disabled must not resample")

        monkeypatch.setattr(orch, "_query_model", fake_query)
        votes = {"winner": "A", "confidence": 0.4, "agreement_ratio": 0.2}
        w, c, _ = asyncio.run(orch._self_consistency_pass(
            ["m1"], {"m1": 0.5}, "prompt", "A", 0.4, votes, _make_run()))
        assert (w, c) == ("A", 0.4)

    def test_adopts_resampled_winner_when_confidence_rises(self, monkeypatch):
        orch = self._make_orchestrator(monkeypatch, min_agreement=0.5, samples=2)
        calls = {"n": 0}

        async def fake_query(model, prompt, temperature=0.7, **k):
            calls["n"] += 1
            calls["temp"] = temperature
            # Both samples agree on a stronger answer, overriding low-confidence A
            return {"model": model, "response": "B", "success": True}

        monkeypatch.setattr(orch, "_query_model", fake_query)

        class FakeEnsemble:
            def ensemble_vote(self, **kw):
                return {"winner": "B", "confidence": 0.9, "agreement_ratio": 1.0}

        monkeypatch.setattr(orch, "ensemble_voter", FakeEnsemble())
        votes = {"winner": "A", "confidence": 0.3, "agreement_ratio": 0.2}
        w, c, v = asyncio.run(orch._self_consistency_pass(
            ["m1", "m2"], {"m1": 0.5, "m2": 0.5}, "prompt", "A", 0.3, votes, _make_run()))
        assert w == "B" and c == 0.9
        assert calls["n"] == 4  # 2 models x 2 samples
        assert calls["temp"] == pytest.approx(0.9)

    def test_keeps_original_when_no_confidence_gain(self, monkeypatch):
        orch = self._make_orchestrator(monkeypatch, min_agreement=0.5, samples=1)
        async def fake_query(model, prompt, temperature=0.7, **k):
            return {"model": model, "response": "C", "success": True}
        monkeypatch.setattr(orch, "_query_model", fake_query)

        class FakeEnsemble:
            def ensemble_vote(self, **kw):
                return {"winner": "C", "confidence": 0.2, "agreement_ratio": 1.0}

        monkeypatch.setattr(orch, "ensemble_voter", FakeEnsemble())
        votes = {"winner": "A", "confidence": 0.5, "agreement_ratio": 0.1}
        w, c, _ = asyncio.run(orch._self_consistency_pass(
            ["m1"], {"m1": 0.5}, "prompt", "A", 0.5, votes, _make_run()))
        assert (w, c) == ("A", 0.5)


# ---------------------------------------------------------------------------
# 3.5.5 — prompt_evolution A/B routing
# ---------------------------------------------------------------------------

class TestPromptEvolutionWiring:
    def test_select_template_and_record_outcome(self):
        pe = PromptEvolution()
        tpl = pe.select_template("pressure")
        assert tpl["id"].startswith("pressure_") and tpl["text"]
        stats_before = pe.get_template_stats()[tpl["id"]]["score"]
        pe.record_outcome(tpl["id"], consensus_reached=True, rounds_needed=1)
        stats_after = pe.get_template_stats()[tpl["id"]]
        assert stats_after["trials"] == 1
        assert stats_after["successes"] == 1
        assert stats_after["score"] > 0

    def test_deliberation_round2_uses_pressure_template(self, monkeypatch):
        pe = PromptEvolution()

        def fake_select(template_type):
            return {"id": "pressure_vX", "type": template_type,
                    "text": "INJECTED-PRESSURE-TEXT"}

        monkeypatch.setattr(pe, "select_template", fake_select)
        captured = {}

        async def query_fn(agent, messages, temperature):
            captured["prompt"] = messages[-1]["content"]
            return {"response": "member reply", "success": True}

        orch = DeliberationOrchestrator(
            query_fn=query_fn,
            prompt_evolution=pe,
            config=DeliberationConfig(engagement_re_prompt=False),
        )
        agents = [
            {"role": "member_1", "name": "a1", "model": "m1", "system_prompt": "sys"},
            {"role": "member_2", "name": "a2", "model": "m2", "system_prompt": "sys"},
        ]
        round1 = _make_round("round1")
        asyncio.run(orch.run_round2(agents, "question?", round1, 0.7))
        assert "INJECTED-PRESSURE-TEXT" in captured["prompt"]
        assert orch._round2_template_id == "pressure_vX"

    def test_deliberation_records_outcome_per_run(self, monkeypatch):
        pe = PromptEvolution()
        monkeypatch.setattr(
            pe, "select_template",
            lambda t: {"id": "pressure_vY", "type": t, "text": "P-TEXT"},
        )
        record_calls = []
        monkeypatch.setattr(pe, "record_outcome", lambda *a, **k: record_calls.append((a, k)))

        async def query_fn(agent, messages, temperature):
            user = messages[-1]["content"]
            if "RESTATE:" in user and "ALT_FRAMING:" in user:
                return {"response": "RESTATE: ok\nALT_FRAMING: alt", "success": True}
            if "Round 2" in user:
                return {"response": "engage a1 and a2 by name.", "success": True}
            if "Round 3" in user:
                return {"response": "final: recommend Y.", "success": True}
            return {"response": "independent analysis.", "success": True}

        orch = DeliberationOrchestrator(query_fn=query_fn, prompt_evolution=pe)
        agents = [
            {"role": "member_1", "name": "a1", "model": "m1", "system_prompt": "sys"},
            {"role": "member_2", "name": "a2", "model": "m2", "system_prompt": "sys"},
        ]
        asyncio.run(orch.run(agents, "question?", temperature=0.7))
        assert record_calls, "record_outcome must be called once per run"
        args, kwargs = record_calls[0]
        assert args == ("pressure_vY",)
        assert kwargs["rounds_needed"] == 3
        assert "consensus_reached" in kwargs


# ---------------------------------------------------------------------------
# 3.3.3 — evidence-grounded deliberation rounds
# ---------------------------------------------------------------------------

class TestEvidenceGroundedDeliberation:
    def _query_fn(self):
        async def fn(agent, messages, temperature):
            user = messages[-1]["content"]
            if "RESTATE:" in user and "ALT_FRAMING:" in user:
                return {"response": "RESTATE: ok\nALT_FRAMING: alt", "success": True}
            if "Round 2" in user:
                return {"response": "engage a1 and a2 by name.", "success": True}
            if "Round 3" in user:
                return {"response": "final: evidence supports X.", "success": True}
            return {"response": "independent analysis with evidence.", "success": True}
        return fn

    class _FakeSearchEngine:
        async def search(self, query, max_results=None, **kw):
            return [
                {"title": "R1", "snippet": "economic growth evidence snippet"},
                {"title": "R2", "snippet": "inflation data snippet"},
            ]

    def test_evidence_injected_into_rounds(self):
        captured = {"prompts": []}

        async def fn(agent, messages, temperature):
            user = messages[-1]["content"]
            captured["prompts"].append(user)
            if "RESTATE:" in user and "ALT_FRAMING:" in user:
                return {"response": "RESTATE: ok\nALT_FRAMING: alt", "success": True}
            if "Round 2" in user:
                return {"response": "engage a1 and a2 by name.", "success": True}
            if "Round 3" in user:
                return {"response": "final: evidence supports X.", "success": True}
            return {"response": "independent analysis with evidence.", "success": True}

        cfg = DeliberationConfig(evidence_grounded=True)
        orch = DeliberationOrchestrator(
            query_fn=fn, config=cfg, evidence_engine=self._FakeSearchEngine(),
        )
        agents = [
            {"role": "member_1", "name": "a1", "model": "m1", "system_prompt": "sys"},
            {"role": "member_2", "name": "a2", "model": "m2", "system_prompt": "sys"},
        ]
        result = asyncio.run(orch.run(agents, "Is inflation rising?", temperature=0.7))

        # Round 1 prompt = independent analysis block containing the evidence
        round1_prompt = next(p for p in captured["prompts"] if "independent analysis" in p.lower())
        assert "search evidence" in round1_prompt.lower()
        assert "economic growth evidence snippet" in round1_prompt
        # Round 3 prompt = final-position block carrying the cross-check
        round3_prompt = next(p for p in captured["prompts"] if "final position" in p.lower())
        assert "cross-check" in round3_prompt.lower()

        # Round-3 outputs carry evidence alignment scores
        for out in result.final_positions:
            assert "evidence_alignment" in out
            assert 0.0 <= out["evidence_alignment"] <= 1.0

        assert result.metadata["evidence_grounded"] is True

    def test_no_evidence_engine_skips_gracefully(self):
        async def fn(agent, messages, temperature):
            return {"response": "plain answer", "success": True}

        cfg = DeliberationConfig(evidence_grounded=True, restate_gate_enabled=False,
                                 dissent_quota_enabled=False)
        orch = DeliberationOrchestrator(query_fn=fn, config=cfg, evidence_engine=None)
        agents = [
            {"role": "member_1", "name": "a1", "model": "m1", "system_prompt": "sys"},
            {"role": "member_2", "name": "a2", "model": "m2", "system_prompt": "sys"},
        ]
        result = asyncio.run(orch.run(agents, "question?", temperature=0.7))
        assert len(result.rounds) == 3
        assert result.metadata["evidence_grounded"] is False


def _make_round(name):
    from src.ai_congress.core.deliberation import RoundResult, _enforce_word_limit
    outputs = [
        {"agent": "a1", "role": "member_1", "model": "m1", "response": "r1-a1", "success": True},
        {"agent": "a2", "role": "member_2", "model": "m2", "response": "r1-a2", "success": True},
    ]
    for o in outputs:
        o["response"] = _enforce_word_limit(o["response"], 400)
    return RoundResult(name=name, outputs=outputs)