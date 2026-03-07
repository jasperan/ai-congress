"""Tests for ACP Heartbeat System"""
import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.ai_congress.core.acp.heartbeat import (
    HeartbeatConfig, HeartbeatState, HeartbeatResult, HeartbeatManager,
)
from src.ai_congress.core.acp.audit_trail import AuditTrail
from src.ai_congress.core.acp.registry import AgentRegistry
from src.ai_congress.core.acp.message import AgentIdentity


class TestHeartbeatConfig:
    def test_defaults(self):
        config = HeartbeatConfig()
        assert config.interval_seconds == 30.0
        assert config.max_autonomous_tokens == 500
        assert config.session_budget == 10000
        assert "health_check" in config.enabled_activities

    def test_custom_config(self):
        config = HeartbeatConfig(interval_seconds=10.0, session_budget=5000)
        assert config.interval_seconds == 10.0
        assert config.session_budget == 5000


class TestHeartbeatState:
    def test_states_exist(self):
        assert HeartbeatState.SLEEPING == "sleeping"
        assert HeartbeatState.AWAKE == "awake"
        assert HeartbeatState.REFLECTING == "reflecting"
        assert HeartbeatState.READY == "ready"


class TestHeartbeatResult:
    def test_result_creation(self):
        result = HeartbeatResult(
            agent_name="phi3:3.8b",
            timestamp=time.time(),
            state=HeartbeatState.READY,
            activities_performed=["health_check"],
            findings={"health": "ok"},
            tokens_used=0,
            alignment_drift=0.0,
        )
        assert result.agent_name == "phi3:3.8b"
        assert result.state == HeartbeatState.READY
        assert result.alignment_drift == 0.0


class TestHeartbeatManager:
    def setup_method(self):
        self.registry = AgentRegistry()
        self.audit_trail = AuditTrail()
        self.ollama_client = AsyncMock()
        self.config = HeartbeatConfig(interval_seconds=0.1, session_budget=1000)
        self.manager = HeartbeatManager(
            registry=self.registry,
            audit_trail=self.audit_trail,
            ollama_client=self.ollama_client,
            config=self.config,
        )
        self.registry.register(AgentIdentity(name="phi3:3.8b", role="worker"))
        self.registry.register(AgentIdentity(name="mistral:7b", role="critic"))

    def test_manager_creation(self):
        assert self.manager is not None
        assert self.manager._config.interval_seconds == 0.1

    @pytest.mark.asyncio
    async def test_single_heartbeat_cycle(self):
        self.ollama_client.chat.return_value = {"message": {"content": "I'm healthy"}}
        result = await self.manager.heartbeat_cycle("phi3:3.8b")
        assert result.agent_name == "phi3:3.8b"
        assert result.state == HeartbeatState.READY
        assert "health_check" in result.activities_performed

    @pytest.mark.asyncio
    async def test_heartbeat_records_audit_event(self):
        self.ollama_client.chat.return_value = {"message": {"content": "ok"}}
        await self.manager.heartbeat_cycle("phi3:3.8b")
        timeline = self.audit_trail.get_agent_timeline("phi3:3.8b")
        assert len(timeline) >= 1
        assert any(e.event_type.value == "agent.heartbeat" for e in timeline)

    @pytest.mark.asyncio
    async def test_heartbeat_health_check_failure(self):
        self.ollama_client.chat.side_effect = Exception("Model unavailable")
        result = await self.manager.heartbeat_cycle("phi3:3.8b")
        assert result.findings.get("health") == "unresponsive"

    @pytest.mark.asyncio
    async def test_budget_tracking(self):
        self.ollama_client.chat.return_value = {"message": {"content": "reflection"}}
        await self.manager.heartbeat_cycle("phi3:3.8b")
        assert self.manager.get_tokens_used("phi3:3.8b") >= 0

    @pytest.mark.asyncio
    async def test_budget_exhaustion_degrades_to_health_only(self):
        self.manager._tokens_used["phi3:3.8b"] = self.config.session_budget
        self.ollama_client.chat.return_value = {"message": {"content": "ok"}}
        result = await self.manager.heartbeat_cycle("phi3:3.8b")
        assert "self_reflection" not in result.activities_performed

    @pytest.mark.asyncio
    async def test_get_last_heartbeat(self):
        self.ollama_client.chat.return_value = {"message": {"content": "ok"}}
        await self.manager.heartbeat_cycle("phi3:3.8b")
        last = self.manager.get_last_heartbeat("phi3:3.8b")
        assert last is not None
        assert last.agent_name == "phi3:3.8b"

    @pytest.mark.asyncio
    async def test_get_last_heartbeat_none_if_never_run(self):
        assert self.manager.get_last_heartbeat("unknown") is None
