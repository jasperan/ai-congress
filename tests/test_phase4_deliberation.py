"""Phase 4 — deliberation exposure:
4.2.3 /api/deliberation route + triads + replays, 3.3.1 Round-2 engagement
compliance, 4.2.1 verdict formatter moved out of VotingEngine (issue #11)."""
import asyncio

import pytest
from unittest.mock import AsyncMock, patch

from src.ai_congress.core.deliberation import (
    DeliberationConfig,
    DeliberationOrchestrator,
    _check_engagement,
    _count_engagement,
    _peer_tokens,
)
from src.ai_congress.core.deliberation_verdict import (
    extract_next_steps,
    extract_unresolved,
    format_deliberation_verdict,
)
from src.ai_congress.core.voting_engine import VotingEngine


# ---------------------------------------------------------------------------
# 4.2.1 — verdict formatter moved out of VotingEngine (issue #11)
# ---------------------------------------------------------------------------

class TestVerdictModule:
    def _positions(self):
        return [
            {"agent": "Skeptic", "role": "skeptic", "response": (
                "I recommend holding rates; my key assumption is inflation "
                "stays sticky. What would make me wrong? A labor shock."
            ), "success": True},
            {"agent": "Optimist", "role": "optimist", "response": (
                "I recommend a cut in Q3. Next step: watch the jobs report."
            ), "success": True},
        ]

    def test_format_verdict_sections(self):
        verdict = format_deliberation_verdict(
            question="Should we cut rates?",
            final_positions=self._positions(),
            final_answer="Hold rates.",
            dissent_report={"agreement_ratio": 0.9, "method": "embedding"},
        )
        assert "## Unresolved Questions" in verdict
        assert "## Recommended Next Steps" in verdict
        assert "## Final Positions" in verdict
        assert "## Weighted Majority" in verdict
        assert "Hold rates." in verdict
        assert "Skeptic" in verdict and "Optimist" in verdict

    def test_verdict_leads_with_unresolved(self):
        verdict = format_deliberation_verdict(
            question="q", final_positions=self._positions(), final_answer="x",
        )
        assert verdict.index("## Unresolved Questions") < verdict.index("## Weighted Majority")

    def test_steelman_and_warning_sections(self):
        verdict = format_deliberation_verdict(
            question="q",
            final_positions=self._positions(),
            final_answer="x",
            restate={
                "warning": "question-reframing warning: 3 of 3 reframed differently",
                "restates": [{"agent": "Skeptic", "alt_framing": "Is the question the timeline?"}],
            },
            steelman=[{"agent": "Skeptic", "response": "The best case against cutting is..."}],
        )
        assert "## Question-Reframing Warning" in verdict
        assert "## Steelmanned Dissent" in verdict
        assert "Is the question the timeline?" in verdict

    def test_voting_engine_delegates(self):
        # Backward compat: VotingEngine.deliberation_verdict still works.
        positions = self._positions()
        expected = format_deliberation_verdict(
            question="q", final_positions=positions, final_answer="x",
        )
        assert VotingEngine().deliberation_verdict(
            question="q", final_positions=positions, final_answer="x",
        ) == expected

    def test_extract_unresolved_and_steps(self):
        positions = self._positions()
        unresolved = extract_unresolved(positions)
        assert any("labor shock" in u for u in unresolved)
        steps = extract_next_steps(positions)
        assert any("hold" in s.lower() for s in steps)
        assert any("jobs report" in s.lower() for s in steps)


# ---------------------------------------------------------------------------
# 3.3.1 — Round-2 engagement compliance
# ---------------------------------------------------------------------------

AGENTS = [
    {"role": "Skeptic", "name": "Skeptic@qwen3.5:9b", "model": "qwen3.5:9b", "system_prompt": "s"},
    {"role": "Optimist", "name": "Optimist@qwen3.5:9b", "model": "qwen3.5:9b", "system_prompt": "s"},
    {"role": "Pragmatist", "name": "Pragmatist@qwen3.5:9b", "model": "qwen3.5:9b", "system_prompt": "s"},
]


class TestEngagement:
    def test_peer_tokens_strip_model_suffix(self):
        tokens = _peer_tokens(AGENTS[0])
        assert "skeptic" in tokens
        assert "qwen3.5:9b" not in tokens

    def test_check_engagement_counts_named_peers(self):
        outputs = [
            {"agent": "Skeptic@qwen3.5:9b", "response": (
                "Optimist argues X, and Pragmatist counters with Y. I side with Optimist."
            )},
            {"agent": "Optimist@qwen3.5:9b", "response": (
                "I agree with nothing here."  # engages nobody
            )},
            {"agent": "Pragmatist@qwen3.5:9b", "response": (
                "Building on Skeptic, I add Z. Optimist's point is weak."
            )},
        ]
        engaged = _check_engagement(outputs, AGENTS, min_peers=2)
        assert set(engaged["Skeptic@qwen3.5:9b"]) >= {"optimist", "pragmatist"}
        assert engaged["Optimist@qwen3.5:9b"] == []
        assert set(engaged["Pragmatist@qwen3.5:9b"]) >= {"skeptic", "optimist"}
        compliant, total = _count_engagement(engaged, min_peers=2)
        assert (compliant, total) == (2, 3)

    def test_self_mention_does_not_count(self):
        outputs = [{"agent": "Skeptic@qwen3.5:9b", "response": "Skeptic thinks this is wrong."}]
        engaged = _check_engagement(outputs, AGENTS, min_peers=2)
        assert engaged["Skeptic@qwen3.5:9b"] == []

    async def _run_round2(self, responses, config=None):
        """Run Round 2 with a stub query_fn returning the given responses in order."""
        queue = list(responses)

        async def query_fn(agent, messages, temperature):
            text = queue.pop(0) if queue else "(no response)"
            return {"response": text, "success": True}

        orch = DeliberationOrchestrator(
            query_fn=query_fn,
            config=config or DeliberationConfig(engagement_required=True),
        )
        from src.ai_congress.core.deliberation import RoundResult
        round1 = RoundResult(name="round1", outputs=[
            {"agent": a["name"], "role": a["role"], "response": "R1", "success": True}
            for a in AGENTS
        ])
        return await orch.run_round2(AGENTS, "question?", round1, 0.7), orch

    def test_round2_reprompts_non_compliant_once(self):
        # First 3 responses ignore peers entirely; the retries comply.
        round2, orch = asyncio.run(self._run_round2([
            "no names here",
            "no names here either",
            "still nothing",
            # retries (comply)
            "Optimist and Pragmatist both claim X; I refine it",
            "Skeptic and Pragmatist disagree; I add a third view",
            "Skeptic and Optimist miss the cost; here it is",
        ]))
        outputs = round2.outputs
        for o in outputs:
            assert o["engagement"]["re_prompted"] is True
            assert o["engagement"]["compliant"] is True
        assert orch.config.engagement_min_peers == 2

    def test_round2_no_reprompt_when_compliant(self):
        round2, _ = asyncio.run(self._run_round2([
            "Optimist says X and Pragmatist says Y; I agree with Optimist",
            "Skeptic says A and Pragmatist says B; I refine A",
            "Skeptic says C and Optimist says D; I dissent from D",
        ]))
        for o in round2.outputs:
            assert o["engagement"]["re_prompted"] is False
            assert o["engagement"]["compliant"] is True

    def test_engagement_disabled_config(self):
        round2, _ = asyncio.run(self._run_round2(
            ["nothing", "nothing", "nothing"],
            config=DeliberationConfig(engagement_required=False),
        ))
        for o in round2.outputs:
            assert o["engagement"]["compliant"] is True  # bypassed
            assert o["engagement"]["re_prompted"] is False

    def test_engagement_metadata_in_run(self):
        async def query_fn(agent, messages, temperature):
            if "Round 2" in messages[-1]["content"]:
                me = (agent.get("name") or "").split("@")[0].lower()
                others = [a["role"] for a in AGENTS if a is not agent]
                return {"response": (
                    f"{others[0]} claims X, {others[1]} refutes it. I add: Y."
                ), "success": True}
            return {"response": "some analysis", "success": True}

        orch = DeliberationOrchestrator(
            query_fn=query_fn,
            config=DeliberationConfig(
                restate_gate_enabled=False,
                dissent_quota_enabled=False,
                engagement_min_peers=2,
            ),
        )
        result = asyncio.run(orch.run(AGENTS, "question?"))
        meta = result.metadata["engagement_compliance"]
        assert meta["total"] == 3
        assert meta["compliant"] >= 2  # Round-2 texts name 2+ peers


# ---------------------------------------------------------------------------
# 4.2.3 — API surface (hermetic: swarm stubbed)
# ---------------------------------------------------------------------------

class TestDeliberationAPI:
    def _client(self):
        from fastapi.testclient import TestClient
        from src.ai_congress.api.main import app
        return TestClient(app, raise_server_exceptions=False)

    def test_triads_list_and_detail(self):
        client = self._client()
        r = client.get("/api/triads")
        assert r.status_code == 200
        triads = r.json().get("triads", [])
        assert len(triads) >= 20  # config/triads.json ships 20
        detail = client.get(f"/api/triads/{triads[0]['name']}")
        assert detail.status_code == 200
        assert "roles" in detail.json() or "models" in detail.json()

    def test_deliberation_requires_agents(self):
        client = self._client()
        r = client.post("/api/deliberation", json={
            "question": "Should the fed cut rates?", "models": ["only-one"],
        })
        assert r.status_code == 400
        assert "at least 2" in r.json()["detail"].lower()

    def test_deliberation_unknown_triad(self):
        client = self._client()
        r = client.post("/api/deliberation", json={
            "question": "q?", "triad": "no-such-triad",
        })
        assert r.status_code == 400

    def test_deliberation_runs_and_returns_verdict(self):
        fake_result = {
            "mode": "deliberation",
            "responses": [{"model": "m1", "agent": "Skeptic", "response": "hold", "success": True}],
            "rounds": [{"name": "round1", "outputs": []}],
            "restate": None, "dissent_report": None, "steelman": None,
            "final_answer": "hold rates", "verdict": "## Weighted Majority\nhold rates",
            "confidence": 0.8, "vote_breakdown": {}, "agents_used": ["Skeptic"],
            "metadata": {}, "engagement_compliance": None,
        }
        with patch("src.ai_congress.api.routes.deliberation.swarm") as fake_swarm:
            fake_swarm.inference_backend = "ollama"
            fake_swarm.deliberation_swarm = AsyncMock(return_value=fake_result)
            client = self._client()
            r = client.post("/api/deliberation", json={
                "question": "q?", "models": ["m1", "m2"],
            })
        assert r.status_code == 200
        body = r.json()
        assert body["verdict"] == "## Weighted Majority\nhold rates"
        assert body["mode"] == "deliberation"
        assert len(body["agents"]) == 2

    def test_replays_list_and_missing(self):
        client = self._client()
        r = client.get("/api/replays")
        assert r.status_code == 200
        assert "replays" in r.json()
        missing = client.get("/api/replays/definitely-missing")
        assert missing.status_code == 404

    def test_chat_mode_deliberation(self):
        fake_result = {
            "mode": "deliberation", "responses": [], "rounds": [],
            "restate": None, "dissent_report": None, "steelman": None,
            "final_answer": "hold", "verdict": "## Final Positions\n- x",
            "confidence": 0.7, "vote_breakdown": {}, "agents_used": [],
            "metadata": {}, "engagement_compliance": None,
        }
        with patch("src.ai_congress.api.routes.chat.swarm") as fake_swarm:
            fake_swarm.inference_backend = "ollama"
            fake_swarm.deliberation_swarm = AsyncMock(return_value=fake_result)
            client = self._client()
            r = client.post("/api/chat", json={
                "prompt": "q?", "models": ["m1", "m2"], "mode": "deliberation",
            })
        assert r.status_code == 200
        assert r.json()["verdict"] == "## Final Positions\n- x"
