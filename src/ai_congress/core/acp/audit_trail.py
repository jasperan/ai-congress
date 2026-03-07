"""Audit Trail - Immutable, causally-linked decision provenance for all agent actions."""

import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class AuditEventType(str, Enum):
    # Lifecycle
    AGENT_REGISTERED = "agent.registered"
    AGENT_HEARTBEAT = "agent.heartbeat"
    AGENT_ROTATED = "agent.rotated"
    # Goal alignment
    MISSION_LOADED = "mission.loaded"
    OBJECTIVE_ASSIGNED = "goal.objective_assigned"
    ALIGNMENT_CHECK = "goal.alignment_check"
    MISALIGNMENT_FLAG = "goal.misalignment_flag"
    # Query processing
    QUERY_RECEIVED = "query.received"
    QUERY_DECOMPOSED = "query.decomposed"
    RESPONSE_GENERATED = "response.generated"
    # Voting & debate
    VOTE_CAST = "vote.cast"
    DEBATE_ROUND = "debate.round"
    CONSENSUS_REACHED = "consensus.reached"
    MINORITY_DISSENT = "vote.minority_dissent"
    VETO_EXERCISED = "vote.veto"
    # Communication
    MESSAGE_SENT = "comm.message_sent"
    HANDOFF_DELEGATED = "comm.handoff_delegated"
    HANDOFF_COMPLETED = "comm.handoff_completed"
    ESCALATION = "comm.escalation"
    # Governance
    BUDGET_WARNING = "gov.budget_warning"
    BUDGET_EXHAUSTED = "gov.budget_exhausted"
    CONFIG_CHANGED = "gov.config_changed"


@dataclass
class AuditEvent:
    event_type: AuditEventType
    agent_name: str
    payload: dict[str, Any]
    position: Optional[str] = None
    caused_by: Optional[str] = None
    run_id: Optional[str] = None
    mission_alignment: Optional[float] = None
    tokens_consumed: int = 0
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)


class AuditTrail:
    """Append-only audit log with causal linking."""

    def __init__(self):
        self._events: list[AuditEvent] = []
        self._index_by_id: dict[str, AuditEvent] = {}

    def record(self, event: AuditEvent) -> str:
        self._events.append(event)
        self._index_by_id[event.id] = event
        return event.id

    def get_causal_chain(self, event_id: str) -> list[AuditEvent]:
        chain = []
        current_id = event_id
        while current_id and current_id in self._index_by_id:
            event = self._index_by_id[current_id]
            chain.append(event)
            current_id = event.caused_by
        chain.reverse()
        return chain

    def get_agent_timeline(self, agent_name: str, limit: int = 50) -> list[AuditEvent]:
        matching = [e for e in self._events if e.agent_name == agent_name]
        return matching[-limit:]

    def get_run_trace(self, run_id: str) -> list[AuditEvent]:
        return [e for e in self._events if e.run_id == run_id]

    def get_token_usage(self, agent_name: str = None, since: float = None) -> dict[str, int]:
        usage: dict[str, int] = defaultdict(int)
        for event in self._events:
            if agent_name and event.agent_name != agent_name:
                continue
            if since and event.timestamp < since:
                continue
            if event.tokens_consumed > 0:
                usage[event.agent_name] += event.tokens_consumed
        return dict(usage)

    def export_replay(self, run_id: str) -> dict:
        events = self.get_run_trace(run_id)
        return {
            "run_id": run_id,
            "events": [
                {
                    "id": e.id,
                    "event_type": e.event_type.value,
                    "agent_name": e.agent_name,
                    "payload": e.payload,
                    "timestamp": e.timestamp,
                    "caused_by": e.caused_by,
                    "tokens_consumed": e.tokens_consumed,
                }
                for e in events
            ],
        }
