"""
ImplementationRun - Isolated execution context for swarm queries.

Provides scoped state management for multi-step query sessions,
including document chunks, intermediate answers, and per-agent state.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DEBATING = "debating"
    VOTING = "voting"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentState:
    """Per-agent state within a run."""
    model_name: str
    role: str = "worker"
    responses: list[str] = field(default_factory=list)
    position_history: list[int] = field(default_factory=list)
    conviction_score: float = 0.0
    stall_count: int = 0
    last_active: float = field(default_factory=time.time)


@dataclass
class ImplementationRun:
    """Isolated execution context for a swarm query."""
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    query: str = ""
    status: RunStatus = RunStatus.PENDING
    agent_states: dict[str, AgentState] = field(default_factory=dict)
    sub_queries: list[dict] = field(default_factory=list)
    intermediate_results: list[dict] = field(default_factory=list)
    final_result: Optional[dict] = None
    anchored_responses: dict[str, str] = field(default_factory=dict)  # hash -> response
    turn_count: int = 0
    max_turns: int = 10
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    event_log: list[dict] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def log_event(self, event: str, agent: str = "", detail: str = ""):
        self.event_log.append({
            "timestamp": time.time(),
            "turn": self.turn_count,
            "event": event,
            "agent": agent,
            "detail": detail,
        })

    def register_agent(self, model_name: str, role: str = "worker") -> AgentState:
        state = AgentState(model_name=model_name, role=role)
        self.agent_states[model_name] = state
        self.log_event("AGENT_REGISTERED", model_name, f"role={role}")
        return state

    def record_response(self, model_name: str, response: str, anchor: str = ""):
        if model_name in self.agent_states:
            self.agent_states[model_name].responses.append(response)
            self.agent_states[model_name].last_active = time.time()
        if anchor:
            self.anchored_responses[anchor] = response
        self.log_event("RESPONSE", model_name, f"len={len(response)}")

    def advance_turn(self):
        self.turn_count += 1
        self.log_event("TURN_ADVANCE", detail=f"turn={self.turn_count}")

    def complete(self, result: dict):
        self.status = RunStatus.COMPLETED
        self.final_result = result
        self.completed_at = time.time()
        self.log_event("RUN_COMPLETE", detail=f"turns={self.turn_count}")

    def fail(self, error: str):
        self.status = RunStatus.FAILED
        self.completed_at = time.time()
        self.log_event("RUN_FAILED", detail=error)

    @property
    def duration_seconds(self) -> float:
        end = self.completed_at or time.time()
        return end - self.created_at

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "query": self.query,
            "status": self.status.value,
            "turn_count": self.turn_count,
            "agent_count": len(self.agent_states),
            "duration_seconds": round(self.duration_seconds, 2),
            "sub_queries": self.sub_queries,
            "intermediate_results_count": len(self.intermediate_results),
            "event_count": len(self.event_log),
            "has_final_result": self.final_result is not None,
        }
