"""
Tests for the triad loader and the deliberation verdict format.
"""
import pytest

from src.ai_congress.core.triads import (
    TriadError,
    describe_triad,
    list_triads,
    load_triads,
    resolve_triad,
)
from src.ai_congress.core.voting_engine import VotingEngine


@pytest.mark.unit
class TestTriadLoader:
    def test_has_20_triads(self):
        names = list_triads()
        assert len(names) == 20

    def test_expected_triads_present(self):
        expected = {
            "architecture", "strategy", "ethics", "debugging", "innovation",
            "conflict", "complexity", "risk", "shipping", "product",
            "founder", "ai", "ai-product", "decision", "unknowns",
            "market-entry", "system-design", "reframing", "ai-frontier",
            "blind-spots",
        }
        actual = set(list_triads())
        assert expected == actual

    def test_every_triad_has_three_agents(self):
        data = load_triads()
        for name, entry in data["triads"].items():
            agents = entry.get("agents", [])
            assert len(agents) == 3, f"{name} has {len(agents)} agents, expected 3"

    def test_every_triad_archetype_resolves(self):
        data = load_triads()
        archetypes = data["archetypes"]
        for name, entry in data["triads"].items():
            for a in entry["agents"]:
                assert a["role"] in archetypes, f"{name} uses unknown role '{a['role']}'"

    def test_resolve_debugging_triad(self):
        agents = resolve_triad("debugging")
        roles = [a["role"] for a in agents]
        assert roles == ["empiricist", "questioner", "formalist"]
        for a in agents:
            assert a["system_prompt"]
            assert a["model"]

    def test_fallback_when_model_unavailable(self):
        agents = resolve_triad(
            "debugging",
            fallback_model="qwen3.5:9b",
            available_models=["qwen3.5:9b"],
        )
        assert all(a["model"] == "qwen3.5:9b" for a in agents)

    def test_unknown_triad_raises(self):
        with pytest.raises(TriadError):
            resolve_triad("does-not-exist")

    def test_describe_triad(self):
        info = describe_triad("architecture")
        assert info["name"] == "architecture"
        assert len(info["roles"]) == 3
        assert info["description"]


@pytest.mark.unit
class TestDeliberationVerdict:
    def _sample_positions(self):
        return [
            {
                "agent": "classifier@qwen3.5:9b",
                "role": "classifier",
                "model": "qwen3.5:9b",
                "response": (
                    "Recommend a modular monolith first. "
                    "The assumption that would make me wrong is team size above 30."
                ),
                "success": True,
            },
            {
                "agent": "empiricist@qwen3:4b",
                "role": "empiricist",
                "model": "qwen3:4b",
                "response": (
                    "I propose delaying the split for 6 months. "
                    "Unclear whether the boundaries are stable. "
                    "What if our load pattern changes entirely?"
                ),
                "success": True,
            },
            {
                "agent": "formalist@gemma4:latest",
                "role": "formalist",
                "model": "gemma4:latest",
                "response": (
                    "Action: formalize service boundaries before splitting. "
                    "Assumption: current boundaries reflect real coupling."
                ),
                "success": True,
            },
        ]

    def test_verdict_leads_with_unresolved(self):
        v = VotingEngine()
        verdict = v.deliberation_verdict(
            question="monolith or microservices?",
            final_positions=self._sample_positions(),
            final_answer="Recommend a modular monolith first.",
        )
        # Leads with Unresolved Questions (since no restate warning)
        first_heading = verdict.split("\n")[0]
        assert first_heading.startswith("## Unresolved Questions")

    def test_verdict_shows_restate_warning_first(self):
        v = VotingEngine()
        verdict = v.deliberation_verdict(
            question="q",
            final_positions=self._sample_positions(),
            restate={
                "warning": "question-reframing warning: 3 of 3 diverged",
                "divergent_count": 3,
                "restates": [{"agent": "alpha", "alt_framing": "alt A"}],
            },
            final_answer="x",
        )
        assert verdict.startswith("## Question-Reframing Warning")
        assert "Alternative framings proposed" in verdict

    def test_verdict_places_majority_last(self):
        v = VotingEngine()
        verdict = v.deliberation_verdict(
            question="q",
            final_positions=self._sample_positions(),
            final_answer="monolith wins",
        )
        assert verdict.index("## Final Positions") < verdict.index("## Weighted Majority")

    def test_verdict_surfaces_assumption_lines(self):
        v = VotingEngine()
        verdict = v.deliberation_verdict(
            question="q",
            final_positions=self._sample_positions(),
            final_answer="x",
        )
        # An 'assumption' line from any agent should land in Unresolved Questions
        assert "assumption" in verdict.lower()

    def test_verdict_includes_steelman_section(self):
        v = VotingEngine()
        verdict = v.deliberation_verdict(
            question="q",
            final_positions=self._sample_positions(),
            steelman=[{"agent": "alpha", "response": "steelman response here"}],
            dissent_report={"agreement_ratio": 0.95, "method": "lexical"},
            final_answer="x",
        )
        assert "## Steelmanned Dissent" in verdict
        assert "steelman response here" in verdict
        assert "Round-1 agreement: 0.95" in verdict
