"""
Phase 6 hardening tests.

- 4.6.3  mock-Ollama fixture (defined in conftest.py) used across hermetic tests
- 4.6.4  golden tests for the delicate deliberation string logic
- 4.6.5  property-style voting invariants (hand-rolled, no hypothesis dep)
- 4.6.7  eval harness runs as tests with a JSON report artifact
- 3.5.6  eval harness + benchmark auto-update behavior
"""
import asyncio
import json
import random
from pathlib import Path

import pytest

from ai_congress.core.consensus_detector import pick_steelman_targets
from ai_congress.core.deliberation import (
    _count_engagement,
    _check_engagement,
    _enforce_word_limit,
    _parse_restate,
    _peer_tokens,
    ENGAGEMENT_REPROMPT_TEMPLATE,
)
from ai_congress.core.deliberation_verdict import (
    extract_next_steps,
    extract_unresolved,
    format_deliberation_verdict,
)
from ai_congress.utils import evals as evals_mod

# ── 4.6.4 Golden tests: deliberation string logic ───────────────────────


class TestEnforceWordLimit:
    def test_short_text_untouched(self):
        assert _enforce_word_limit("hello world", 100) == "hello world"

    def test_exactly_at_limit(self):
        text = " ".join(f"w{i}" for i in range(5))
        assert _enforce_word_limit(text, 5) == text

    def test_truncates_and_marks(self):
        text = " ".join(f"w{i}" for i in range(10))
        result = _enforce_word_limit(text, 4)
        assert result == "w0 w1 w2 w3 [...]"
        assert result.endswith("[...]")

    def test_empty_text(self):
        assert _enforce_word_limit("", 10) == "" or _enforce_word_limit(None or "", 10) == ""


class TestParseRestate:
    def test_golden_full_format(self):
        raw = "Here is the question again.\nRESTATE: The real issue is budget allocation.\nALT_FRAMING: A scheduling problem in disguise."
        restate, alt = _parse_restate(raw)
        assert restate == "The real issue is budget allocation."
        assert alt == "A scheduling problem in disguise."

    def test_golden_restate_only(self):
        raw = "RESTATE: We must decide whether to expand the plant."
        restate, alt = _parse_restate(raw)
        assert restate == "We must decide whether to expand the plant."
        assert alt == ""

    def test_golden_alt_only(self):
        raw = "ALT_FRAMING: This is actually a hiring question."
        restate, alt = _parse_restate(raw)
        assert restate == "" or "hiring" in alt
        assert alt == "This is actually a hiring question."

    def test_unstructured_fallback_first_line(self):
        raw = "I think the core problem is coordination.\nSecond line ignored."
        restate, alt = _parse_restate(raw)
        assert restate.startswith("I think the core problem is coordination.")
        assert alt == ""

    def test_golden_lowercase_markers(self):
        raw = "restate: lowercase marker works too\nALT_FRAMING: and the alt"
        restate, alt = _parse_restate(raw)
        assert restate == "lowercase marker works too"
        assert "and the alt" in alt


class TestPeerTokens:
    def test_strips_model_suffix(self):
        agent = {"name": "Skeptic@qwen3.5:9b", "role": "Skeptic"}
        tokens = _peer_tokens(agent)
        assert "skeptic" in tokens
        assert "qwen3.5:9b" not in tokens
        assert "skeptic@qwen3.5:9b" not in tokens

    def test_role_fallback(self):
        assert _peer_tokens({"name": "", "role": "Devil's Advocate"}) == ["devil's advocate"]

    def test_no_duplicate_tokens(self):
        tokens = _peer_tokens({"name": "chair", "role": "Chair"})
        assert tokens == ["chair"]


class TestEngagementCounting:
    def make_agents(self):
        return [
            {"name": "member_1@a", "role": "Analyst"},
            {"name": "member_2@b", "role": "Skeptic"},
            {"name": "member_3@c", "role": "Chair"},
        ]

    def test_counts_compliance(self):
        agents = self.make_agents()
        engaged = {  # as _check_engagement returns: agent -> peer tokens mentioned
            "member_1@a": ["member_2", "member_3"],   # 2 peers -> compliant
            "member_2@b": ["member_1", "member_3"],   # 2 peers -> compliant
            "member_3@c": [],                            # 0 peers -> not
        }
        compliant, total = _count_engagement(engaged, min_peers=2)
        assert (compliant, total) == (2, 3)

    def test_self_mention_not_counted(self):
        # _check_engagement skips the output's own peer (self-mention)
        agents = self.make_agents()
        outputs = [{"agent": "member_1@a", "response": "member_1 agrees with itself."}]
        engaged = _check_engagement(outputs, peers=agents, min_peers=2)
        assert engaged == {"member_1@a": []}
        compliant, total = _count_engagement(engaged, min_peers=2)
        assert compliant == 0 and total == 1


class TestSteelmanPick:
    def _report(self, positions, pairwise):
        from ai_congress.core.consensus_detector import ConsensusReport

        return ConsensusReport(
            agreement_ratio=0.4,
            mean_similarity=0.4,
            premature=False,
            method="lexical",
            pairwise=pairwise,
            outlier_index=0,
        )

    def test_picks_lowest_agreement(self):
        positions = [
            {"agent": "a", "success": True, "response": "pos a"},
            {"agent": "b", "success": True, "response": "pos b"},
            {"agent": "c", "success": True, "response": "pos c"},
        ]
        # agent 0 agrees least with the others (lowest pairwise sims)
        report = self._report(
            positions, [(0, 1, 0.2), (0, 2, 0.3), (1, 2, 0.9)]
        )
        picks = pick_steelman_targets(report, count=2)
        assert len(picks) == 2
        assert set(picks) <= {0, 1, 2}
        assert 0 in picks  # the dissenter must steelman

    def test_returns_valid_indices_only(self):
        positions = [
            {"agent": "a", "success": True, "response": "pos a"},
            {"agent": "b", "success": True, "response": "pos b"},
        ]
        report = self._report(positions, [(0, 1, 0.3)])
        picks = pick_steelman_targets(report, count=5)
        assert len(picks) == 2
        assert set(picks) == {0, 1}

    def test_empty_pairwise_returns_empty(self):
        positions = [{"agent": "a", "success": True, "response": "pos a"}]
        report = self._report(positions, [])
        assert pick_steelman_targets(report, count=2) == []


class TestUnresolvedExtraction:
    def test_marker_lines(self):
        positions = [
            {
                "success": True,
                "response": (
                    "Would make me wrong: if the market shifts to "
                    "decentralized storage within the quarter."
                ),
            },
        ]
        unresolved = extract_unresolved(positions)
        assert len(unresolved) == 1
        assert "market shifts" in unresolved[0]

    def test_question_mark_surface(self):
        positions = [{"success": True, "response": "What happens if rates rise next quarter?"}]
        unresolved = extract_unresolved(positions)
        assert any("rates rise" in u for u in unresolved)

    def test_empty_positions(self):
        assert extract_unresolved([]) == []
        assert extract_unresolved([{"success": False, "response": ""}]) == []


class TestNextSteps:
    def test_recommendation_line(self):
        positions = [{"success": True, "response": "I recommend rebalancing the portfolio."}]
        steps = extract_next_steps(positions)
        assert any("rebalancing" in s for s in steps)

    def test_first_sentence_fallback(self):
        positions = [{"success": True, "response": "We should run the pilot. Then scale."}]
        steps = extract_next_steps(positions)
        assert any("pilot" in s for s in steps)


class TestVerdictGolden:
    def test_golden_verdict_ordering(self):
        question = "Should the company expand to Asia?"
        positions = [
            {
                "agent": "a",
                "role": "Analyst",
                "success": True,
                "response": "I recommend a staged rollout. Would make me wrong if "
                "regulatory approvals trail the timeline.",
            },
            {
                "agent": "b",
                "role": "Skeptic",
                "success": True,
                "response": "I recommend deferring. Would make me wrong if local "
                "demand is confirmed early.",
            },
        ]
        verdict = format_deliberation_verdict(
            question=question,
            final_positions=positions,
            restate={
                "warning": "The council restated the question as a market-entry decision.",
                "restates": [{"agent": "a", "alt_framing": "Is this about timing or location?"}],
            },
            dissent_report={"agreement_ratio": 0.4, "dissenting": ["b"]},
            steelman=[],
            final_answer="Staged rollout",
        )
        # The verdict LEADS with what the council does not know
        assert verdict.index("## Unresolved Questions") < verdict.index("## Final Positions")
        assert "Question-Reframing Warning" in verdict
        assert "timing or location" in verdict
        assert "Staged rollout" in verdict


# ── 4.6.5 Voting invariants (property-style) ────────────────────────────


class TestVotingInvariants:
    def test_winner_is_a_member_of_responses(self):
        from ai_congress.core.voting_engine import VotingEngine

        engine = VotingEngine()
        responses = ["alpha", "beta", "gamma"]
        weights = [0.8, 0.6, 0.9]
        winner, confidence, _ = engine.weighted_majority_vote(responses, weights)
        assert winner in responses

    def test_confidence_is_weight_share(self):
        from ai_congress.core.voting_engine import VotingEngine

        engine = VotingEngine()
        responses = ["alpha", "beta", "gamma"]
        weights = [0.9, 0.3, 0.4]
        winner, confidence, details = engine.weighted_majority_vote(responses, weights)
        # confidence == winner's normalized weight share, always in [0, 1]
        expected = max(weights) / sum(weights)
        assert abs(confidence - expected) < 1e-9
        assert 0.0 <= confidence <= 1.0
        # details carry the raw per-response weights
        winner_entry = max(details.values(), key=lambda d: d["weight"])
        assert winner_entry["original"] == winner

    def test_semantic_similarity_bounded_and_symmetric(self):
        from ai_congress.utils.semantic import text_similarity

        for _ in range(25):
            a = "the quick brown fox jumps over the lazy dog"
            b = "a fast brown fox leaps across a sleepy hound"
            s_ab = text_similarity(a, b)
            s_ba = text_similarity(b, a)
            assert 0.0 <= s_ab <= 1.0
            assert abs(s_ab - s_ba) < 1e-9

    def test_rank_responses_order_invariance(self):
        from ai_congress.core.voting_engine import VotingEngine

        engine = VotingEngine()
        responses = ["alpha", "beta", "gamma", "delta"]
        weights = [0.9, 0.5, 0.7, 0.2]
        model_names = ["m1", "m2", "m3", "m4"]

        def rank_ids(pairs):
            news = [p[0] for p in pairs]
            new_weights = [p[1] for p in pairs]
            ranked = engine.rank_responses(news, new_weights, model_names)
            return [(r["original"], round(r["weight"], 5)) for r in ranked]

        base = rank_ids(list(zip(responses, weights)))
        for _ in range(25):
            pairs = list(zip(responses, weights))
            random.shuffle(pairs)
            assert rank_ids(pairs) == base


# ── 4.6.3 / 4.6.7 Evals as tests with mock Ollama ───────────────────────


class EvalFakeClient:
    """Deterministic fake client: returns ground truth for known questions."""

    def __init__(self):
        self.calls = []

    async def generate(self, *, prompt=None, model=None, stream=False, temperature=None, **kwargs):
        self.calls.append((model, prompt))
        if "capital of France" in (prompt or ""):
            return {"response": "The capital of France is Paris.", "model": model}
        if "HTML stand for" in (prompt or ""):
            return {"response": "HTML stands for HyperText Markup Language.", "model": model}
        if "created the Python" in (prompt or ""):
            return {"response": "Python was created by Guido van Rossum.", "model": model}
        if "requests per minute" in (prompt or ""):
            return {"response": "60 requests per minute on average.", "model": model}
        return {"response": "I do not know the answer to that.", "model": model}


@pytest.mark.unit
class TestEvalHarness:
    @pytest.fixture(autouse=True)
    def isolate_report_dir(self, monkeypatch):
        """Tighten the eval set to the questions the fake client knows."""
        known_ids = ["capital-france", "html-purpose", "python-creator", "req-time"]
        monkeypatch.setattr(
            evals_mod, "EVAL_SET", [i for i in evals_mod.EVAL_SET if i["id"] in known_ids]
        )
        yield

    def test_run_evals_scores_and_reports(self, tmp_path):
        client = EvalFakeClient()
        report = asyncio.run(evals_mod.run_evals(client, ["model-a"], report_dir=str(tmp_path)))
        assert report["model_count"] == 1
        assert report["question_count"] == 4
        top = report["ranked"][0]
        assert top["model"] == "model-a"
        assert top["correct"] == 4  # all four answered correctly
        artifact = Path(tmp_path) / "eval_report.json"
        assert artifact.exists()
        loaded = json.loads(artifact.read_text())
        assert loaded["ranked"][0]["accuracy"] > 0.9

    def test_timeout_scores_zero(self, tmp_path):
        class Flaky(EvalFakeClient):
            async def generate(self, **kwargs):
                raise asyncio.TimeoutError("slow")

        report = asyncio.run(evals_mod.run_evals(Flaky(), ["model-a"], report_dir=str(tmp_path)))
        assert report["ranked"][0]["accuracy"] == 0.0
        assert report["ranked"][0]["timeouts"] == 4

    def test_benchmark_update_folds_without_clobber(self, tmp_path):
        import ai_congress.utils.evals as ev

        # seed a tiny benchmark table in tmp dir
        bench = tmp_path / "models_benchmark.json"
        bench.write_text(json.dumps({"phi3:3.8b": {"accuracy": 0.69, "mmlu": 0.69}}))
        report = {
            "generated_at": "2026-01-01",
            "question_count": 2,
            "model_count": 1,
            "ranked": [{"model": "phi3:3.8b", "accuracy": 0.75, "correct": 3, "total": 4, "avg_score": 0.7, "timeouts": 0, "errors": 0}],
            "results": [],
        }
        updates = ev.compute_benchmark_update(report, blend=0.2, benchmark_path=str(bench))
        assert "phi3:3.8b" in updates
        # 0.8 * 0.69 (existing) + 0.2 * 0.75 = 0.702 — blended, not replaced
        assert abs(updates["phi3:3.8b"]["new"] - (0.8 * 0.69 + 0.2 * 0.75)) < 1e-6
        # persisted back and sorted by accuracy
        persisted = json.loads(bench.read_text())
        assert persisted["phi3:3.8b"]["accuracy"] == updates["phi3:3.8b"]["new"]
        assert persisted["phi3:3.8b"].get("last_eval_accuracy") == 0.75

    def test_score_response_blend(self):
        assert evals_mod._score_response("Paris", "Paris", ["paris"]) > 0.8
        assert evals_mod._score_response("no idea", "Paris", ["paris"]) < 0.5

    def test_summarize_contains_models(self):
        report = {
            "question_count": 2,
            "model_count": 1,
            "ranked": [{"model": "m", "accuracy": 0.5, "correct": 1, "total": 2, "avg_score": 0.5, "timeouts": 0, "errors": 0}],
        }
        text = evals_mod.summarize(report)
        assert "m" in text and "50%" in text


# ── 4.9.3–4.9.6 Security hardening ──────────────────────────────────────


class TestRagInjectionHardening:
    def test_warning_appended_to_context(self):
        from ai_congress.api.security import harden_rag_context

        ctx = "Source 1: Some facts about the economy."
        hardened = harden_rag_context(ctx)
        assert "DOCUMENT DATA, not instructions" in hardened
        assert ctx in hardened

    def test_warning_is_idempotent(self):
        from ai_congress.api.security import harden_rag_context

        once = harden_rag_context("facts")
        twice = harden_rag_context(once)
        assert once == twice
        assert once.count("DOCUMENT DATA, not instructions") == 1

    def test_empty_context_passthrough(self):
        from ai_congress.api.security import harden_rag_context, harden_deliberation_evidence

        assert harden_rag_context("") == ""
        assert harden_deliberation_evidence("") == ""

    def test_rag_engine_template_keeps_warning_out_of_template(self):
        from ai_congress.api.security import RAG_UNTRUSTED_DOCS_WARNING

        # The warning must not be inside the user-overridable template string
        assert "UNTRUSTED" not in ("Based on the following context")
        assert bool(RAG_UNTRUSTED_DOCS_WARNING.strip())


class TestApiKeyGuard:
    def test_disabled_guard_allows_all(self):
        from ai_congress.api.security import create_require_api_key

        dep = create_require_api_key(False, "")
        result = asyncio.run(dep(None))
        assert result is None

    def test_enabled_guard_rejects_missing_key(self):
        from ai_congress.api.security import create_require_api_key

        dep = create_require_api_key(True, "sekrit")
        with pytest.raises(Exception) as exc:  # HTTPException in FastAPI runtime
            asyncio.run(dep(_FakeRequest(headers={})))
        assert exc.value.status_code == 401

    def test_enabled_guard_accepts_correct_key(self):
        from ai_congress.api.security import create_require_api_key

        dep = create_require_api_key(True, "sekrit")
        result = asyncio.run(dep(_FakeRequest(headers={"X-API-Key": "sekrit"})))
        assert result is None

    def test_enabled_guard_rejects_wrong_key(self):
        from ai_congress.api.security import create_require_api_key

        dep = create_require_api_key(True, "sekrit")
        with pytest.raises(Exception) as exc:
            asyncio.run(dep(_FakeRequest(headers={"X-API-Key": "wrong"})))
        assert exc.value.status_code == 401


class _FakeRequest:
    def __init__(self, headers=None, client=None):
        self.headers = headers or {}
        self.client = client


class TestRateLimiter:
    def test_allows_within_window(self):
        from ai_congress.api.security import RateLimiter

        limiter = RateLimiter(limit=3, window_s=60, limit_loopback=True)
        req = _FakeRequest(client=None)
        for _ in range(3):
            allowed, remaining = limiter.is_allowed(req)
            assert allowed is True
        allowed, remaining = limiter.is_allowed(req)
        assert allowed is False
        assert remaining == 0

    def test_loopback_exempt_by_default(self):
        from ai_congress.api.security import RateLimiter

        limiter = RateLimiter(limit=1, window_s=60)
        request = _FakeRequest(client=_FakeClient(host="127.0.0.1"))
        allowed, _ = limiter.is_allowed(request)
        assert allowed is True
        # exhaust
        allowed, _ = limiter.is_allowed(request)
        assert allowed is True  # still exempt (loopback)

    def test_window_resets(self):
        from ai_congress.api.security import RateLimiter

        limiter = RateLimiter(limit=1, window_s=0.05, limit_loopback=True)
        req = _FakeRequest(client=None)
        allowed, _ = limiter.is_allowed(req)
        assert allowed is True
        allowed, _ = limiter.is_allowed(req)
        assert allowed is False
        import time as _time
        _time.sleep(0.06)
        allowed, _ = limiter.is_allowed(req)
        assert allowed is True

    def test_forwarded_for_client_ip(self):
        from ai_congress.api.security import RateLimiter

        limiter = RateLimiter(limit=2, window_s=60, limit_loopback=False)
        req = _FakeRequest(client=_FakeClient(host="127.0.0.1"), headers={"x-forwarded-for": "203.0.113.9"})
        allowed, _ = limiter.is_allowed(req)
        assert allowed is True
        allowed, _ = limiter.is_allowed(req)
        assert allowed is True
        allowed, _ = limiter.is_allowed(req)
        assert allowed is False  # 3rd hit exceeds limit of 2 by that IP


class _FakeClient:
    def __init__(self, host="127.0.0.1"):
        self.host = host


class TestSpendGovernor:
    def test_local_ollama_uncapped(self):
        from ai_congress.api.security import SpendGovernor

        g = SpendGovernor(max_per_run=1, max_per_session=10)
        g.begin_run("r1")
        assert g.try_acquire("r1", backend="ollama") is True
        assert g.try_acquire("r1", backend="ollama") is True  # free

    def test_cloud_capped_per_run(self):
        from ai_congress.api.security import SpendGovernor

        g = SpendGovernor(max_per_run=2, max_per_session=100)
        g.begin_run("r1")
        assert g.try_acquire("r1", backend="pi") is True
        assert g.try_acquire("r1", backend="pi") is True
        assert g.try_acquire("r1", backend="pi") is False  # run cap hit

    def test_session_cap_global(self):
        from ai_congress.api.security import SpendGovernor

        g = SpendGovernor(max_per_run=5, max_per_session=3)
        g.begin_run("r1")
        assert g.try_acquire("r1", backend="pi") is True
        g.begin_run("r2")
        assert g.try_acquire("r2", backend="pi") is True
        assert g.try_acquire("r2", backend="pi") is True
        assert g.try_acquire("r2", backend="pi") is False  # session cap hit

    def test_usage_tracking(self):
        from ai_congress.api.security import SpendGovernor

        g = SpendGovernor()
        g.begin_run("r1")
        g.try_acquire("r1", backend="pi")
        assert g.run_usage("r1") == 1
        assert g.session_stats()["session_used"] == 1

    def test_openai_client_honors_governor(self):
        # The governor is enforced inside OpenAIClient.chat: when the cap is
        # reached the client returns an empty response rather than spending.
        from ai_congress.api.security import SpendGovernor
        from ai_congress.core.openai_client import OpenAIClient

        governor = SpendGovernor(max_per_run=0, max_per_session=0)
        client = OpenAIClient.__new__(OpenAIClient)  # uninitialized — no HTTP
        client.spend_governor = governor
        client.run_id = "r1"
        client.max_tokens = 4096
        client.max_retries = 3

        result = asyncio.run(client.chat(model="deepseek-v4-flash", messages=[{"role": "user", "content": "hi"}]))
        assert result.get("error") == "spend_limit"
        assert result["message"]["content"] == ""


class TestMockOllamaFixture:
    def test_canned_success(self, mock_ollama_client):
        mock_ollama_client.responses["hello"] = {"response": "hi back", "model": "mock"}
        result = asyncio.run(mock_ollama_client.generate(prompt="say hello", model="m"))
        assert result["response"] == "hi back"

    def test_canned_response_dict_not_mutated(self, mock_ollama_client):
        canned = {"response": "original", "model": "mock"}
        mock_ollama_client.responses["x"] = canned
        result = asyncio.run(mock_ollama_client.generate(prompt="x", model="m"))
        result["response"] = "mutated"
        assert mock_ollama_client.responses["x"]["response"] == "original"

    def test_canned_timeout(self, mock_ollama_client):
        async def check():
            mock_ollama_client.responses["ping"] = mock_ollama_client.TIMEOUT
            with pytest.raises(asyncio.TimeoutError):
                await mock_ollama_client.generate(prompt="ping", model="m")

        asyncio.run(check())

    def test_default_response(self, mock_ollama_client):
        result = asyncio.run(mock_ollama_client.generate(prompt="anything", model="m"))
        assert result["response"] == "mock answer"

    def test_calls_recorded(self, mock_ollama_client):
        asyncio.run(mock_ollama_client.generate(prompt="q", model="m", options={"temperature": 0.5}))
        assert mock_ollama_client.calls[0]["model"] == "m"
        assert mock_ollama_client.calls[0]["options"]["temperature"] == 0.5