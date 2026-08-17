"""
Phase 7 strategic tests.

- 3.1.9 MemGPT-style memory: JSON persistence (short-term) + precedent-store
  semantic recall when available
- 3.2.7 bargaining consensus: mediator-guided negotiation, utility scoring,
  convergence, persistence-free deterministic behavior
"""
import asyncio
import json
import os

import pytest

from ai_congress.core.intelligence.agent_memory import AgentMemory
from ai_congress.core.negotiation import BargainingSession, Proposal, _utility_from_weights


# ── 3.1.9 MemGPT-style memory persistence ───────────────────────────────


class StubPrecedentStore:
    """Minimal stand-in for the Oracle-backed precedent store."""

    def __init__(self, hits=None):
        self.hits = hits or []

    async def search(self, query, top_k=3):
        return self.hits


class TestMemoryPersistence:
    def test_exchanges_persist_to_file(self, tmp_path):
        path = str(tmp_path / "memory.json")
        mem = AgentMemory(persist_path=path)
        mem.add_exchange("what is the fed rate", "The fed rate is 4.5%")
        mem.add_exchange("who is chair of the fed", "Jerome Powell")
        assert os.path.exists(path)
        with open(path) as f:
            saved = json.load(f)
        assert len(saved["short_term"]) == 2

    def test_restores_after_reload(self, tmp_path):
        path = str(tmp_path / "memory.json")
        mem = AgentMemory(persist_path=path)
        mem.add_exchange("what is the fed rate", "The fed rate is 4.5%")
        mem.add_exchange("who is chair of the fed", "Jerome Powell")

        fresh = AgentMemory(persist_path=path)
        assert fresh.get_stats()["short_term_count"] == 2
        recalled = asyncio.run(fresh.recall_relevant("what is the fed rate"))
        assert len(recalled) >= 1
        assert "4.5%" in recalled[0]["response"]

    def test_long_term_promotion_persists(self, tmp_path):
        path = str(tmp_path / "memory.json")
        mem = AgentMemory(persist_path=path)
        mem.add_exchange("recommend a strategy for market entry", "Phased rollout")
        # near-duplicate query promotes the exchange to long-term
        mem.add_exchange("recommend a strategy for market entry please", "Staged expansion")

        fresh = AgentMemory(persist_path=path)
        stats = fresh.get_stats()
        assert stats["long_term_count"] >= 1
        assert stats["short_term_count"] == 2

    def test_missing_file_graceful(self, tmp_path):
        mem = AgentMemory(persist_path=str(tmp_path / "nope.json"))
        assert mem.get_stats()["short_term_count"] == 0
        assert asyncio.run(mem.recall_relevant("anything")) == []

    def test_unwritable_dir_degrades(self):
        mem = AgentMemory(persist_path="/dev/null/not-a-dir/memory.json")
        # must not raise
        mem.add_exchange("q", "a")
        assert mem.get_stats()["short_term_count"] == 1

    def test_precedent_recall_pages_semantic(self):
        store = StubPrecedentStore([
            {"query": "rate decision history", "response": "Fed cut rates in 2019 after a slowdown.",
             "similarity": 0.8},
            {"query": "unrelated", "response": "Cooking times for rice.", "similarity": 0.05},
        ])
        mem = AgentMemory(precedent_store=store)
        recalled = asyncio.run(mem.recall_relevant("what happened last time rates changed?"))
        # only the high-similarity precedent surfaces
        assert any("Fed cut rates" in r["response"] for r in recalled)
        assert not any("Cooking" in r["response"] for r in recalled)
        assert any(r["source"] == "precedent" for r in recalled)

    def test_memory_context_builds(self):
        mem = AgentMemory()
        mem.add_exchange("capital of france", "Paris")
        ctx = asyncio.run(mem.build_memory_context("capital of france?"))
        assert "Paris" in ctx
        assert "Relevant past exchanges" in ctx

    def test_clear_short_term(self):
        mem = AgentMemory()
        mem.add_exchange("q", "a")
        mem.clear_short_term()
        assert mem.get_stats()["short_term_count"] == 0


# ── 3.2.7 Bargaining consensus ──────────────────────────────────────────


class FakeBargainingClient:
    """Scripted member responses: converge by round 2."""

    def __init__(self):
        self.calls = []

    async def generate(self, *, prompt=None, model=None, stream=False, temperature=None, **kwargs):
        self.calls.append((model, prompt))
        r1 = (
            f"POSITION: model {model} proposes a staged rollout with strict controls.\n"
            "INSISTENCE: 0.9"
        )
        r2 = (
            f"POSITION: model {model} accepts the compromise with a review checkpoint.\n"
            "INSISTENCE: 0.4"
        )
        if "FINAL round" in (prompt or ""):
            return {"response": "POSITION: agreed compromise with checkpoint.\nINSISTENCE: 0.2", "model": model}
        return {"response": r1 if "PROPOSALS SO FAR" not in (prompt or "") else r2, "model": model}


class TestUtilityScoring:
    def test_priority_weights_drive_score(self):
        scorer = _utility_from_weights({"cost": 0.8, "speed": 0.5, "quality": 0.3})
        assert scorer("we must cut cost aggressively") > scorer("focus on quality craftsmanship")
        assert scorer("focus on quality craftsmanship") > 0.0
        assert 0.0 <= scorer("anything") <= 1.0

    def test_empty_weights_neutral(self):
        scorer = _utility_from_weights({})
        assert scorer("anything") == 0.5

    def test_parse_proposal(self):
        session = BargainingSession(client=None)
        prop = session._parse_proposal(
            "POSITION: raise rates by 25bp\nINSISTENCE: 0.7", "a", "m", 0.5
        )
        assert prop.position == "raise rates by 25bp"
        assert prop.demand_level == 0.7

    def test_parse_proposal_insistence_clamped(self):
        session = BargainingSession(client=None)
        prop = session._parse_proposal(
            "POSITION: hold steady\nINSISTENCE: 99", "a", "m", 0.5
        )
        assert prop.demand_level == 1.0


class TestBargainingFlow:
    def test_converges_by_round_two(self):
        session = BargainingSession(
            client=FakeBargainingClient(),
            max_rounds=3,
            tolerance=0.75,
            priority_weights={"rollout": 0.9},
        )
        agents = [{"name": "a", "model": "m1"}, {"name": "b", "model": "m2"}]
        result = asyncio.run(session.negotiate(agents, "Should we expand?"))
        assert result["strategy"] == "bargaining"
        assert result["settled"] is True
        assert len(result["rounds"]) >= 1
        # every round carries proposals with parsed positions
        first = result["rounds"][0]
        assert len(first["proposals"]) == 2
        assert all(p["position"] for p in first["proposals"])
        assert "consensus" in result and result["consensus"]["text"]

    def test_round_cap_limits_iterations(self):
        session = BargainingSession(client=FakeBargainingClient(), max_rounds=2, tolerance=0.99)
        agents = [{"name": "a", "model": "m1"}]
        result = asyncio.run(session.negotiate(agents, "q?"))
        assert len(result["rounds"]) == 2
        assert result["settled"] is False

    def test_member_timeout_abstains(self):
        class TimeoutClient(FakeBargainingClient):
            async def generate(self, **kwargs):
                raise asyncio.TimeoutError("slow")

        session = BargainingSession(client=TimeoutClient(), max_rounds=1)
        agents = [{"name": "a", "model": "m1"}]
        result = asyncio.run(session.negotiate(agents, "q?"))
        # abstain proposal parses but converges to 0 — never 'settled'
        assert any("abstain" in p["position"] for p in result["rounds"][0]["proposals"])
        assert result["settled"] is False

    def test_mediator_builds_weighted_compromise(self):
        session = BargainingSession(client=None)
        proposals = [
            Proposal(agent="a", model="m1", position="cut costs by 20%", utility=0.9, demand_level=0.9),
            Proposal(agent="b", model="m2", position="hire more engineers", utility=0.5, demand_level=0.4),
        ]
        compromise = session._mediate(proposals, 1)
        assert "cut costs" in compromise["text"]
        assert 0.0 <= compromise["convergence"] <= 1.0
        assert len(compromise["concessions"]) == 2

    def test_negotiate_api_shape(self):
        session = BargainingSession(client=FakeBargainingClient(), max_rounds=1)
        agents = [{"name": "a", "model": "m1"}]
        result = asyncio.run(session.negotiate(agents, "q?"))
        assert {"strategy", "consensus", "rounds", "duration_s", "settled"} <= set(result.keys())