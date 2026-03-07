"""Tests for typed ACP message payload contracts"""
import pytest
from src.ai_congress.core.acp.message import (
    ResponsePayload, VotePayload, HandoffPayload, HeartbeatPayload,
)


class TestResponsePayload:
    def test_creation(self):
        p = ResponsePayload(text="Paris is the capital", confidence=0.95, reasoning_mode="cot")
        assert p.text == "Paris is the capital"
        assert p.confidence == 0.95
        assert p.reasoning_mode == "cot"
        assert p.evidence == []
        assert p.alignment_score == 1.0

    def test_with_evidence(self):
        p = ResponsePayload(
            text="Answer", confidence=0.8, reasoning_mode="react",
            evidence=["source1", "source2"], alignment_score=0.9,
        )
        assert len(p.evidence) == 2
        assert p.alignment_score == 0.9


class TestVotePayload:
    def test_creation(self):
        p = VotePayload(chosen_cluster=0, conviction=0.85, rationale="Most accurate")
        assert p.chosen_cluster == 0
        assert p.conviction == 0.85
        assert p.rationale == "Most accurate"
        assert p.dissent_notes is None

    def test_with_dissent(self):
        p = VotePayload(chosen_cluster=1, conviction=0.6, rationale="Closest", dissent_notes="Cluster 0 had merit")
        assert p.dissent_notes == "Cluster 0 had merit"


class TestHandoffPayload:
    def test_creation(self):
        p = HandoffPayload(
            task_type="calculate",
            context="Need to verify 2+2",
            constraints=["must_be_precise"],
        )
        assert p.task_type == "calculate"
        assert p.deadline_seconds == 60.0
        assert p.escalation_path == []

    def test_with_escalation(self):
        p = HandoffPayload(
            task_type="summarize", context="Long text",
            constraints=[], escalation_path=["Chair", "Speaker"],
        )
        assert len(p.escalation_path) == 2


class TestHeartbeatPayload:
    def test_creation(self):
        p = HeartbeatPayload(
            state="ready", health="ok", tokens_remaining=5000, mission_alignment=0.95,
        )
        assert p.state == "ready"
        assert p.health == "ok"
        assert p.reflection is None
        assert p.tokens_remaining == 5000
        assert p.mission_alignment == 0.95
