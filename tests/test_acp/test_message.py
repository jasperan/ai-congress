"""
Tests for ACP Core Dataclasses: PersonalityProfile, AgentIdentity, ACPMessage
"""
import pytest
from src.ai_congress.core.acp.message import PersonalityProfile, AgentIdentity, ACPMessage


class TestPersonalityProfile:
    def test_personality_profile_defaults(self):
        profile = PersonalityProfile()
        assert profile.openness == 5
        assert profile.conscientiousness == 5
        assert profile.extraversion == 5
        assert profile.agreeableness == 5
        assert profile.neuroticism == 5
        assert profile.risk_tolerance == 5
        assert profile.empathy_level == 5
        assert profile.leadership == 5
        assert profile.stress == 0
        assert profile.confidence == 0.5
        assert profile.engagement == 5
        assert profile.communication_style == "balanced"

    def test_personality_profile_custom(self):
        profile = PersonalityProfile(
            openness=9,
            conscientiousness=3,
            extraversion=8,
            agreeableness=2,
            neuroticism=7,
            risk_tolerance=10,
            empathy_level=1,
            leadership=6,
            stress=4,
            confidence=0.9,
            engagement=8,
            communication_style="assertive",
        )
        assert profile.openness == 9
        assert profile.conscientiousness == 3
        assert profile.extraversion == 8
        assert profile.agreeableness == 2
        assert profile.neuroticism == 7
        assert profile.risk_tolerance == 10
        assert profile.empathy_level == 1
        assert profile.leadership == 6
        assert profile.stress == 4
        assert profile.confidence == 0.9
        assert profile.engagement == 8
        assert profile.communication_style == "assertive"


class TestAgentIdentity:
    def test_agent_identity_minimal(self):
        agent = AgentIdentity(name="analyst", role="researcher")
        assert agent.name == "analyst"
        assert agent.role == "researcher"
        assert agent.status == "active"
        assert agent.personality is None
        assert agent.capabilities == []
        assert isinstance(agent.last_active, float)

    def test_agent_identity_with_personality(self):
        profile = PersonalityProfile(extraversion=9, leadership=8)
        agent = AgentIdentity(
            name="leader",
            role="coordinator",
            personality=profile,
            capabilities=["summarize", "vote"],
        )
        assert agent.name == "leader"
        assert agent.role == "coordinator"
        assert agent.personality is not None
        assert agent.personality.extraversion == 9
        assert agent.personality.leadership == 8
        assert "summarize" in agent.capabilities
        assert "vote" in agent.capabilities


class TestACPMessage:
    def test_acp_message_creation(self):
        sender = AgentIdentity(name="agent1", role="analyst")
        msg = ACPMessage(
            sender=sender,
            channel="direct",
            msg_type="opinion",
            payload={"text": "I think option A is best"},
            recipient="agent2",
        )
        assert msg.sender.name == "agent1"
        assert msg.channel == "direct"
        assert msg.msg_type == "opinion"
        assert msg.payload["text"] == "I think option A is best"
        assert msg.recipient == "agent2"
        assert msg.coordination_level == "moderate"
        assert isinstance(msg.id, str)
        assert len(msg.id) > 0
        assert isinstance(msg.timestamp, float)

    def test_acp_message_broadcast(self):
        sender = AgentIdentity(name="moderator", role="coordinator")
        msg = ACPMessage(
            sender=sender,
            channel="broadcast",
            msg_type="announcement",
            payload={"text": "Voting round begins"},
        )
        assert msg.channel == "broadcast"
        assert msg.recipient is None
        assert msg.metadata == {}

    def test_acp_message_vote(self):
        sender = AgentIdentity(name="voter1", role="analyst")
        msg = ACPMessage(
            sender=sender,
            channel="broadcast",
            msg_type="vote",
            payload={"choice": "option_b", "confidence": 0.85},
            coordination_level="chatty",
            metadata={"round": 1},
        )
        assert msg.msg_type == "vote"
        assert msg.payload["choice"] == "option_b"
        assert msg.payload["confidence"] == 0.85
        assert msg.coordination_level == "chatty"
        assert msg.metadata["round"] == 1
