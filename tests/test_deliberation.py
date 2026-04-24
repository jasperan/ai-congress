"""
Tests for the 3-round DeliberationOrchestrator (Problem Restate Gate,
cross-examination, dissent quota + steelman pass).
"""
import asyncio

import pytest

from src.ai_congress.core.deliberation import (
    DeliberationConfig,
    DeliberationOrchestrator,
    RESTATE_PROMPT,
    _count_divergent_restates,
    _parse_restate,
)


def _make_query_fn(script=None):
    """Build a deterministic fake query_fn that echoes role+round.

    script: optional dict keyed by (role, round_key) -> response string.
    """
    script = script or {}

    async def fn(agent, messages, temperature):
        role = agent.get("role") or agent.get("name") or "member"
        user = messages[-1]["content"]
        if "RESTATE:" in user and "ALT_FRAMING:" in user:
            key = (role, "restate")
            default = f"RESTATE: {role} sees the question clearly.\nALT_FRAMING: maybe user wants {role}-centric take."
            return {"response": script.get(key, default), "success": True}
        if "Round 2" in user:
            return {"response": script.get((role, "r2"), f"{role} engages peers: refined."), "success": True}
        if "Round 3" in user:
            return {"response": script.get((role, "r3"), f"{role} final: recommend X."), "success": True}
        if "opposing view" in user or "steelman" in user.lower():
            return {"response": script.get((role, "steel"), f"{role} steelman: opposite is strong."), "success": True}
        return {"response": script.get((role, "r1"), f"{role} r1 analysis."), "success": True}

    return fn


@pytest.mark.unit
class TestRestateParsing:
    def test_parse_well_formed(self):
        raw = "RESTATE: user wants X\nALT_FRAMING: perhaps Y"
        restate, alt = _parse_restate(raw)
        assert restate == "user wants X"
        assert alt == "perhaps Y"

    def test_parse_missing_alt(self):
        raw = "RESTATE: something"
        restate, alt = _parse_restate(raw)
        assert restate == "something"
        assert alt == ""

    def test_parse_noise_fallback(self):
        raw = "just some text without markers"
        restate, alt = _parse_restate(raw)
        assert restate != ""
        assert alt == ""


@pytest.mark.unit
class TestRestateGateDivergence:
    def test_no_divergence_when_identical(self):
        restates = [
            {"restate": "same thing", "alt_framing": "same alt"},
            {"restate": "same thing", "alt_framing": "same alt"},
            {"restate": "same thing", "alt_framing": "same alt"},
        ]
        assert _count_divergent_restates(restates, threshold=0.55) == 0

    def test_divergence_when_all_different(self):
        restates = [
            {"restate": "alpha beta gamma delta", "alt_framing": ""},
            {"restate": "xxxx yyyy zzzz wwww", "alt_framing": ""},
            {"restate": "11 22 33 44 55 66", "alt_framing": ""},
        ]
        assert _count_divergent_restates(restates, threshold=0.55) >= 2

    def test_empty_returns_zero(self):
        assert _count_divergent_restates([], threshold=0.55) == 0


@pytest.mark.unit
class TestDeliberationOrchestrator:
    def test_rejects_single_agent(self):
        orc = DeliberationOrchestrator(query_fn=_make_query_fn(), config=DeliberationConfig())
        with pytest.raises(ValueError):
            asyncio.run(orc.run(agents=[{"role": "only"}], question="q"))

    def test_three_rounds_run(self):
        agents = [
            {"role": "alpha", "model": "m1", "system_prompt": "..."},
            {"role": "beta", "model": "m2", "system_prompt": "..."},
            {"role": "gamma", "model": "m3", "system_prompt": "..."},
        ]
        config = DeliberationConfig(
            embedding_model="",
            consensus_threshold=0.99,     # don't fire steelman
            dissent_quota_enabled=False,
            restate_gate_enabled=False,
        )
        orc = DeliberationOrchestrator(query_fn=_make_query_fn(), config=config)
        result = asyncio.run(orc.run(agents=agents, question="q"))
        assert [r.name for r in result.rounds] == ["round1", "round2", "round3"]
        assert len(result.final_positions) == 3
        assert result.restate is None
        assert result.steelman is None

    def test_steelman_fires_on_premature_consensus(self):
        agents = [
            {"role": "alpha", "model": "m1"},
            {"role": "beta", "model": "m2"},
            {"role": "gamma", "model": "m3"},
        ]
        # Every agent gives the same Round-1 text to force high pairwise similarity.
        script = {
            ("alpha", "r1"): "monolith is correct here always",
            ("beta", "r1"): "monolith is correct here always",
            ("gamma", "r1"): "monolith is correct here always",
        }
        config = DeliberationConfig(
            embedding_model="",
            consensus_threshold=0.1,
            pairwise_threshold=0.5,
            restate_gate_enabled=False,
            dissent_quota_enabled=True,
            dissent_steelman_count=2,
        )
        orc = DeliberationOrchestrator(query_fn=_make_query_fn(script), config=config)
        result = asyncio.run(orc.run(agents=agents, question="q"))
        assert result.dissent_report is not None
        assert result.dissent_report.premature is True
        assert result.steelman is not None
        assert len(result.steelman.outputs) == 2

    def test_word_limits_enforced(self):
        agents = [
            {"role": "alpha"},
            {"role": "beta"},
        ]
        long_text = " ".join(["word"] * 1000)
        script = {
            ("alpha", "r1"): long_text,
            ("beta", "r1"): long_text,
        }
        config = DeliberationConfig(
            embedding_model="",
            restate_gate_enabled=False,
            dissent_quota_enabled=False,
            round1_word_limit=50,
            round2_word_limit=25,
            round3_word_limit=10,
        )
        orc = DeliberationOrchestrator(query_fn=_make_query_fn(script), config=config)
        result = asyncio.run(orc.run(agents=agents, question="q"))
        r1 = result.rounds[0].outputs[0]["response"]
        assert r1.endswith("[...]")
        assert len(r1.split()) <= 55  # 50 + the truncation suffix

    def test_restate_warning_fires_when_divergent(self):
        agents = [
            {"role": "alpha"},
            {"role": "beta"},
            {"role": "gamma"},
        ]
        # Force each agent to restate the question very differently.
        script = {
            ("alpha", "restate"): "RESTATE: alpha thinks the user wants apples.\nALT_FRAMING: apples only",
            ("beta", "restate"): "RESTATE: beta thinks the user wants orchestras and violins.\nALT_FRAMING: music first",
            ("gamma", "restate"): "RESTATE: gamma thinks the user wants rocket launches.\nALT_FRAMING: space exploration",
        }
        config = DeliberationConfig(
            embedding_model="",
            restate_gate_enabled=True,
            restate_divergence_min=3,
            dissent_quota_enabled=False,
        )
        orc = DeliberationOrchestrator(query_fn=_make_query_fn(script), config=config)
        result = asyncio.run(orc.run(agents=agents, question="q"))
        assert result.restate is not None
        assert result.restate.warning is not None
        assert "question-reframing" in result.restate.warning
