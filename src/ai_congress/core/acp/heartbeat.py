"""Heartbeat System - Autonomous agent reasoning cycles with token budgets."""

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .audit_trail import AuditTrail, AuditEvent, AuditEventType
from .registry import AgentRegistry

logger = logging.getLogger(__name__)


@dataclass
class HeartbeatConfig:
    interval_seconds: float = 30.0
    enabled_activities: list[str] = field(default_factory=lambda: [
        "health_check",
        "self_reflection",
        "mission_review",
        "peer_status",
    ])
    max_autonomous_tokens: int = 500
    session_budget: int = 10000


class HeartbeatState(str, Enum):
    SLEEPING = "sleeping"
    AWAKE = "awake"
    REFLECTING = "reflecting"
    READY = "ready"


@dataclass
class HeartbeatResult:
    agent_name: str
    timestamp: float
    state: HeartbeatState
    activities_performed: list[str]
    findings: dict[str, Any]
    tokens_used: int
    alignment_drift: float = 0.0


class HeartbeatManager:
    """Manages periodic agent wakeup cycles."""

    def __init__(
        self,
        registry: AgentRegistry,
        audit_trail: AuditTrail,
        ollama_client=None,
        config: HeartbeatConfig = None,
    ):
        self._registry = registry
        self._audit_trail = audit_trail
        self._ollama_client = ollama_client
        self._config = config or HeartbeatConfig()
        self._tokens_used: dict[str, int] = defaultdict(int)
        self._last_heartbeat: dict[str, HeartbeatResult] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._running = False

    async def start(self, agents: list[str]) -> None:
        self._running = True
        for agent in agents:
            self._tasks[agent] = asyncio.create_task(self._heartbeat_loop(agent))

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks.values():
            task.cancel()
        self._tasks.clear()

    async def _heartbeat_loop(self, agent_name: str) -> None:
        while self._running:
            try:
                await self.heartbeat_cycle(agent_name)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Heartbeat error for %s: %s", agent_name, e)
            await asyncio.sleep(self._config.interval_seconds)

    async def heartbeat_cycle(self, agent_name: str) -> HeartbeatResult:
        activities = []
        findings: dict[str, Any] = {}
        tokens = 0
        budget_remaining = self._config.session_budget - self._tokens_used[agent_name]
        budget_exhausted = budget_remaining <= 0

        # Health check (always runs, no LLM call)
        if "health_check" in self._config.enabled_activities:
            health = await self._health_check(agent_name)
            activities.append("health_check")
            findings["health"] = health

        # Self-reflection (requires budget)
        if (
            "self_reflection" in self._config.enabled_activities
            and not budget_exhausted
            and self._ollama_client is not None
        ):
            try:
                reflection = await self._self_reflect(agent_name)
                activities.append("self_reflection")
                findings["reflection"] = reflection
                tokens += min(self._config.max_autonomous_tokens, 100)
            except Exception as e:
                logger.debug("Self-reflection failed for %s: %s", agent_name, e)

        # Peer status check
        if "peer_status" in self._config.enabled_activities:
            stuck = self._registry.detect_stuck(threshold_seconds=120)
            activities.append("peer_status")
            findings["stuck_peers"] = stuck

        self._tokens_used[agent_name] += tokens

        result = HeartbeatResult(
            agent_name=agent_name,
            timestamp=time.time(),
            state=HeartbeatState.READY,
            activities_performed=activities,
            findings=findings,
            tokens_used=tokens,
        )
        self._last_heartbeat[agent_name] = result

        # Record in audit trail
        self._audit_trail.record(AuditEvent(
            event_type=AuditEventType.AGENT_HEARTBEAT,
            agent_name=agent_name,
            payload={
                "activities": activities,
                "health": findings.get("health", "unknown"),
                "tokens_used": tokens,
            },
            tokens_consumed=tokens,
        ))

        # Update registry last_active
        self._registry.touch(agent_name)

        return result

    async def _health_check(self, agent_name: str) -> str:
        if self._ollama_client is None:
            return "unknown"
        try:
            await asyncio.wait_for(
                self._ollama_client.chat(
                    model=agent_name,
                    messages=[{"role": "user", "content": "ping"}],
                    options={"num_predict": 1},
                ),
                timeout=10.0,
            )
            return "ok"
        except Exception:
            return "unresponsive"

    async def _self_reflect(self, agent_name: str) -> str:
        response = await asyncio.wait_for(
            self._ollama_client.chat(
                model=agent_name,
                messages=[{
                    "role": "user",
                    "content": "In one sentence, what could you improve about your recent responses?",
                }],
                options={"num_predict": 50},
            ),
            timeout=15.0,
        )
        return response["message"]["content"]

    def get_tokens_used(self, agent_name: str) -> int:
        return self._tokens_used.get(agent_name, 0)

    def get_last_heartbeat(self, agent_name: str) -> Optional[HeartbeatResult]:
        return self._last_heartbeat.get(agent_name)
