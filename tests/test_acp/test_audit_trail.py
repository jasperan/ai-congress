"""Tests for ACP Audit Trail"""
import time
import pytest
from src.ai_congress.core.acp.audit_trail import AuditTrail, AuditEvent, AuditEventType


class TestAuditEventType:
    def test_lifecycle_events_exist(self):
        assert AuditEventType.AGENT_REGISTERED == "agent.registered"
        assert AuditEventType.AGENT_HEARTBEAT == "agent.heartbeat"
        assert AuditEventType.AGENT_ROTATED == "agent.rotated"

    def test_goal_events_exist(self):
        assert AuditEventType.MISSION_LOADED == "mission.loaded"
        assert AuditEventType.OBJECTIVE_ASSIGNED == "goal.objective_assigned"
        assert AuditEventType.ALIGNMENT_CHECK == "goal.alignment_check"
        assert AuditEventType.MISALIGNMENT_FLAG == "goal.misalignment_flag"

    def test_query_events_exist(self):
        assert AuditEventType.QUERY_RECEIVED == "query.received"
        assert AuditEventType.QUERY_DECOMPOSED == "query.decomposed"
        assert AuditEventType.RESPONSE_GENERATED == "response.generated"

    def test_vote_events_exist(self):
        assert AuditEventType.VOTE_CAST == "vote.cast"
        assert AuditEventType.DEBATE_ROUND == "debate.round"
        assert AuditEventType.CONSENSUS_REACHED == "consensus.reached"
        assert AuditEventType.MINORITY_DISSENT == "vote.minority_dissent"
        assert AuditEventType.VETO_EXERCISED == "vote.veto"

    def test_comm_events_exist(self):
        assert AuditEventType.MESSAGE_SENT == "comm.message_sent"
        assert AuditEventType.HANDOFF_DELEGATED == "comm.handoff_delegated"
        assert AuditEventType.HANDOFF_COMPLETED == "comm.handoff_completed"
        assert AuditEventType.ESCALATION == "comm.escalation"

    def test_gov_events_exist(self):
        assert AuditEventType.BUDGET_WARNING == "gov.budget_warning"
        assert AuditEventType.BUDGET_EXHAUSTED == "gov.budget_exhausted"
        assert AuditEventType.CONFIG_CHANGED == "gov.config_changed"


class TestAuditEvent:
    def test_creation_with_defaults(self):
        event = AuditEvent(
            event_type=AuditEventType.AGENT_REGISTERED,
            agent_name="phi3:3.8b",
            payload={"role": "worker"},
        )
        assert event.event_type == AuditEventType.AGENT_REGISTERED
        assert event.agent_name == "phi3:3.8b"
        assert event.payload == {"role": "worker"}
        assert event.caused_by is None
        assert event.run_id is None
        assert event.position is None
        assert event.mission_alignment is None
        assert event.tokens_consumed == 0
        assert isinstance(event.id, str)
        assert isinstance(event.timestamp, float)

    def test_creation_with_causal_link(self):
        parent = AuditEvent(
            event_type=AuditEventType.QUERY_RECEIVED,
            agent_name="system",
            payload={"query": "What is 2+2?"},
        )
        child = AuditEvent(
            event_type=AuditEventType.RESPONSE_GENERATED,
            agent_name="phi3:3.8b",
            payload={"text": "4"},
            caused_by=parent.id,
            run_id="run-123",
        )
        assert child.caused_by == parent.id
        assert child.run_id == "run-123"


class TestAuditTrail:
    def setup_method(self):
        self.trail = AuditTrail()

    def test_record_returns_event_id(self):
        event = AuditEvent(
            event_type=AuditEventType.AGENT_REGISTERED,
            agent_name="phi3:3.8b",
            payload={},
        )
        event_id = self.trail.record(event)
        assert event_id == event.id

    def test_record_stores_event(self):
        event = AuditEvent(
            event_type=AuditEventType.AGENT_REGISTERED,
            agent_name="phi3:3.8b",
            payload={},
        )
        self.trail.record(event)
        timeline = self.trail.get_agent_timeline("phi3:3.8b")
        assert len(timeline) == 1
        assert timeline[0].id == event.id

    def test_get_causal_chain(self):
        e1 = AuditEvent(
            event_type=AuditEventType.QUERY_RECEIVED,
            agent_name="system",
            payload={"query": "test"},
        )
        e2 = AuditEvent(
            event_type=AuditEventType.RESPONSE_GENERATED,
            agent_name="phi3:3.8b",
            payload={"text": "answer"},
            caused_by=e1.id,
        )
        e3 = AuditEvent(
            event_type=AuditEventType.VOTE_CAST,
            agent_name="phi3:3.8b",
            payload={"cluster": 0},
            caused_by=e2.id,
        )
        self.trail.record(e1)
        self.trail.record(e2)
        self.trail.record(e3)

        chain = self.trail.get_causal_chain(e3.id)
        assert len(chain) == 3
        assert chain[0].id == e1.id
        assert chain[1].id == e2.id
        assert chain[2].id == e3.id

    def test_get_causal_chain_single_event(self):
        e1 = AuditEvent(
            event_type=AuditEventType.MISSION_LOADED,
            agent_name="system",
            payload={},
        )
        self.trail.record(e1)
        chain = self.trail.get_causal_chain(e1.id)
        assert len(chain) == 1

    def test_get_agent_timeline_filters_by_agent(self):
        e1 = AuditEvent(event_type=AuditEventType.AGENT_REGISTERED, agent_name="phi3:3.8b", payload={})
        e2 = AuditEvent(event_type=AuditEventType.AGENT_REGISTERED, agent_name="mistral:7b", payload={})
        e3 = AuditEvent(event_type=AuditEventType.VOTE_CAST, agent_name="phi3:3.8b", payload={})
        self.trail.record(e1)
        self.trail.record(e2)
        self.trail.record(e3)

        phi3_timeline = self.trail.get_agent_timeline("phi3:3.8b")
        assert len(phi3_timeline) == 2
        mistral_timeline = self.trail.get_agent_timeline("mistral:7b")
        assert len(mistral_timeline) == 1

    def test_get_agent_timeline_respects_limit(self):
        for i in range(10):
            self.trail.record(AuditEvent(
                event_type=AuditEventType.AGENT_HEARTBEAT,
                agent_name="phi3:3.8b",
                payload={"tick": i},
            ))
        timeline = self.trail.get_agent_timeline("phi3:3.8b", limit=3)
        assert len(timeline) == 3
        assert timeline[0].payload["tick"] == 7

    def test_get_run_trace(self):
        e1 = AuditEvent(event_type=AuditEventType.QUERY_RECEIVED, agent_name="system", payload={}, run_id="run-1")
        e2 = AuditEvent(event_type=AuditEventType.RESPONSE_GENERATED, agent_name="phi3", payload={}, run_id="run-1")
        e3 = AuditEvent(event_type=AuditEventType.AGENT_HEARTBEAT, agent_name="phi3", payload={}, run_id=None)
        self.trail.record(e1)
        self.trail.record(e2)
        self.trail.record(e3)
        trace = self.trail.get_run_trace("run-1")
        assert len(trace) == 2

    def test_get_token_usage_all_agents(self):
        self.trail.record(AuditEvent(
            event_type=AuditEventType.RESPONSE_GENERATED, agent_name="phi3", payload={}, tokens_consumed=100,
        ))
        self.trail.record(AuditEvent(
            event_type=AuditEventType.RESPONSE_GENERATED, agent_name="mistral", payload={}, tokens_consumed=200,
        ))
        self.trail.record(AuditEvent(
            event_type=AuditEventType.AGENT_HEARTBEAT, agent_name="phi3", payload={}, tokens_consumed=50,
        ))
        usage = self.trail.get_token_usage()
        assert usage["phi3"] == 150
        assert usage["mistral"] == 200

    def test_get_token_usage_single_agent(self):
        self.trail.record(AuditEvent(
            event_type=AuditEventType.RESPONSE_GENERATED, agent_name="phi3", payload={}, tokens_consumed=100,
        ))
        self.trail.record(AuditEvent(
            event_type=AuditEventType.RESPONSE_GENERATED, agent_name="mistral", payload={}, tokens_consumed=200,
        ))
        usage = self.trail.get_token_usage(agent_name="phi3")
        assert usage["phi3"] == 100
        assert "mistral" not in usage

    def test_get_token_usage_since_timestamp(self):
        old_event = AuditEvent(
            event_type=AuditEventType.RESPONSE_GENERATED, agent_name="phi3", payload={}, tokens_consumed=100,
        )
        old_event.timestamp = time.time() - 3600
        self.trail.record(old_event)
        new_event = AuditEvent(
            event_type=AuditEventType.RESPONSE_GENERATED, agent_name="phi3", payload={}, tokens_consumed=50,
        )
        self.trail.record(new_event)
        since = time.time() - 60
        usage = self.trail.get_token_usage(since=since)
        assert usage.get("phi3", 0) == 50

    def test_export_replay(self):
        e1 = AuditEvent(event_type=AuditEventType.QUERY_RECEIVED, agent_name="system", payload={"q": "test"}, run_id="run-1")
        e2 = AuditEvent(event_type=AuditEventType.CONSENSUS_REACHED, agent_name="system", payload={"winner": "4"}, run_id="run-1")
        self.trail.record(e1)
        self.trail.record(e2)
        replay = self.trail.export_replay("run-1")
        assert replay["run_id"] == "run-1"
        assert len(replay["events"]) == 2
        assert replay["events"][0]["event_type"] == "query.received"
