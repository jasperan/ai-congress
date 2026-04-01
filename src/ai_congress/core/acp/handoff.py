"""
AgentHandoff - Structured agent-to-agent task delegation via ACP message bus.

Enables agents to delegate subtasks to specialized agents and await responses,
using the existing ACP message bus infrastructure.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from .message import ACPMessage, AgentIdentity, ChannelType
from .message_bus import ACPMessageBus
from .registry import AgentRegistry

logger = logging.getLogger(__name__)


@dataclass
class HandoffRequest:
    """A request from one agent to delegate work to another."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    from_agent: str = ""
    to_agent: str = ""
    task_type: str = ""  # e.g., "calculate", "summarize", "critique"
    payload: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class HandoffResponse:
    """Response to a handoff request."""
    request_id: str = ""
    from_agent: str = ""
    result: Any = None
    success: bool = True
    error: Optional[str] = None
    duration_ms: float = 0.0


class AgentHandoff:
    """Manages structured agent-to-agent task delegation."""

    def __init__(self, message_bus: ACPMessageBus, registry: AgentRegistry):
        self.message_bus = message_bus
        self.registry = registry
        self._pending: dict[str, asyncio.Event] = {}
        self._responses: dict[str, HandoffResponse] = {}
        self._handoff_log: list[dict] = []

    def _log_handoff(self, request: HandoffRequest, event: str, detail: str = ""):
        entry = {
            "timestamp": time.time(),
            "request_id": request.id,
            "from": request.from_agent,
            "to": request.to_agent,
            "task_type": request.task_type,
            "event": event,
            "detail": detail,
        }
        self._handoff_log.append(entry)
        logger.info(f"[Handoff] {request.from_agent} -> {request.to_agent}: {event}")

    async def delegate(
        self,
        from_agent: str,
        to_agent: str,
        task_type: str,
        payload: dict,
        timeout: float = 60.0,
    ) -> HandoffResponse:
        """Delegate a subtask from one agent to another.

        Sends a handoff message via the ACP message bus and waits for a response.
        """
        request = HandoffRequest(
            from_agent=from_agent,
            to_agent=to_agent,
            task_type=task_type,
            payload=payload,
        )
        self._log_handoff(request, "DELEGATE", f"type={task_type}")

        # Create wait event
        self._pending[request.id] = asyncio.Event()

        # Send handoff message via bus
        sender_identity = self.registry.agents.get(from_agent)
        if not sender_identity:
            sender_identity = AgentIdentity(name=from_agent, role="worker")

        msg = ACPMessage(
            sender=sender_identity,
            channel=ChannelType.DIRECT,
            msg_type="handoff_request",
            payload={
                "request_id": request.id,
                "task_type": task_type,
                **payload,
            },
            recipient=to_agent,
        )
        self.message_bus.send(msg)

        try:
            await asyncio.wait_for(self._pending[request.id].wait(), timeout=timeout)
            response = self._responses.pop(request.id, None)
            if response:
                self._log_handoff(request, "COMPLETE", f"success={response.success}")
                return response
            return HandoffResponse(
                request_id=request.id,
                from_agent=to_agent,
                success=False,
                error="No response received",
            )
        except asyncio.TimeoutError:
            self._log_handoff(request, "TIMEOUT", f"after {timeout}s")
            return HandoffResponse(
                request_id=request.id,
                from_agent=to_agent,
                success=False,
                error=f"Handoff timed out after {timeout}s",
            )
        finally:
            self._pending.pop(request.id, None)

    def complete_handoff(self, request_id: str, result: Any, success: bool = True, error: str = None):
        """Called by the receiving agent to complete a handoff."""
        response = HandoffResponse(
            request_id=request_id,
            result=result,
            success=success,
            error=error,
        )
        self._responses[request_id] = response
        event = self._pending.get(request_id)
        if event:
            event.set()

    def find_agent_for_task(self, task_type: str) -> Optional[str]:
        """Find the best agent for a given task type using the registry."""
        candidates = self.registry.get_by_capability(task_type)
        if not candidates:
            return None
        # Pick the active one with highest relevance (first match)
        active = [a for a in candidates if a.status == "active"]
        return active[0].name if active else candidates[0].name

    def get_handoff_log(self) -> list[dict]:
        return list(self._handoff_log)
