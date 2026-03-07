"""Tests for enhanced ACP Message Bus (persistence, context, org-aware routing)"""
import pytest
from src.ai_congress.core.acp.message import AgentIdentity, ACPMessage, ChannelType
from src.ai_congress.core.acp.message_bus import ACPMessageBus
from src.ai_congress.core.acp.audit_trail import AuditTrail
from src.ai_congress.core.acp.org_chart import OrgChart
from src.ai_congress.core.acp.roles import AgentRole


class TestMessageBusPersistence:
    def setup_method(self):
        self.audit_trail = AuditTrail()
        self.bus = ACPMessageBus(audit_trail=self.audit_trail)
        self.agent1 = AgentIdentity(name="agent1", role="analyst")
        self.bus.register_agent("agent1")
        self.bus.register_agent("agent2")

    def test_send_records_audit_event(self):
        msg = ACPMessage(
            sender=self.agent1, channel="broadcast",
            msg_type="opinion", payload={"text": "hello"},
        )
        self.bus.send(msg)
        timeline = self.audit_trail.get_agent_timeline("agent1")
        assert len(timeline) == 1
        assert timeline[0].event_type.value == "comm.message_sent"

    def test_send_without_audit_trail_still_works(self):
        bus = ACPMessageBus()
        bus.register_agent("a")
        bus.register_agent("b")
        sender = AgentIdentity(name="a", role="x")
        msg = ACPMessage(sender=sender, channel="broadcast", msg_type="test", payload={})
        bus.send(msg)
        assert len(bus.get_messages("b")) == 1


class TestMessageBusContext:
    def setup_method(self):
        self.bus = ACPMessageBus()
        self.agent1 = AgentIdentity(name="agent1", role="analyst")
        self.agent2 = AgentIdentity(name="agent2", role="reviewer")
        self.bus.register_agent("agent1")
        self.bus.register_agent("agent2")

    def test_get_context_returns_sent_messages(self):
        for i in range(5):
            msg = ACPMessage(
                sender=self.agent1, channel="broadcast",
                msg_type="opinion", payload={"i": i},
            )
            self.bus.send(msg)
        context = self.bus.get_context("agent1")
        assert len(context) == 5

    def test_get_context_respects_limit(self):
        for i in range(10):
            msg = ACPMessage(
                sender=self.agent1, channel="broadcast",
                msg_type="opinion", payload={"i": i},
            )
            self.bus.send(msg)
        context = self.bus.get_context("agent1", limit=3)
        assert len(context) == 3
        assert context[0].payload["i"] == 7

    def test_get_context_empty_for_unknown_agent(self):
        assert self.bus.get_context("unknown") == []


class TestMessageBusOrgRouting:
    def setup_method(self):
        self.chart = OrgChart()
        self.chart.define_default_structure()
        self.chart.appoint("speaker", "Speaker of the House")
        self.chart.appoint("researcher", "Committee Chair: Research")
        self.chart.appoint("critic", "Committee Chair: Oversight")

        self.bus = ACPMessageBus(org_chart=self.chart)
        self.bus.register_agent("speaker")
        self.bus.register_agent("researcher")
        self.bus.register_agent("critic")

        self.researcher_id = AgentIdentity(name="researcher", role="planner")
        self.critic_id = AgentIdentity(name="critic", role="critic")

    def test_send_to_supervisor(self):
        msg = ACPMessage(
            sender=self.researcher_id, channel="direct",
            msg_type="escalation", payload={"text": "need help"},
        )
        self.bus.send_to_supervisor(msg)
        speaker_msgs = self.bus.get_messages("speaker")
        assert len(speaker_msgs) == 1
        assert speaker_msgs[0].payload["text"] == "need help"

    def test_send_to_supervisor_no_org_chart(self):
        bus = ACPMessageBus()
        bus.register_agent("a")
        sender = AgentIdentity(name="a", role="x")
        msg = ACPMessage(sender=sender, channel="direct", msg_type="test", payload={})
        bus.send_to_supervisor(msg)

    def test_send_to_reports(self):
        speaker_id = AgentIdentity(name="speaker", role="judge")
        msg = ACPMessage(
            sender=speaker_id, channel="direct",
            msg_type="directive", payload={"text": "focus on accuracy"},
        )
        self.bus.send_to_reports(msg)
        researcher_msgs = self.bus.get_messages("researcher")
        critic_msgs = self.bus.get_messages("critic")
        assert len(researcher_msgs) == 1
        assert len(critic_msgs) == 1

    def test_escalate(self):
        msg = ACPMessage(
            sender=self.researcher_id, channel="direct",
            msg_type="escalation", payload={"need": "can_veto"},
        )
        resolved = self.bus.escalate(msg, required_authority="can_veto")
        speaker_msgs = self.bus.get_messages("speaker")
        assert len(speaker_msgs) == 1
        assert resolved is True

    def test_escalate_no_one_has_authority(self):
        msg = ACPMessage(
            sender=self.researcher_id, channel="direct",
            msg_type="escalation", payload={},
        )
        resolved = self.bus.escalate(msg, required_authority="nonexistent")
        assert resolved is False
