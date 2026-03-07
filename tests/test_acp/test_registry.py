"""
Tests for ACP Agent Registry
"""
import time
import pytest
from src.ai_congress.core.acp.message import AgentIdentity, PersonalityProfile
from src.ai_congress.core.acp.registry import AgentRegistry


class TestAgentRegistry:
    def setup_method(self):
        self.registry = AgentRegistry()
        self.agent1 = AgentIdentity(name="analyst", role="researcher", capabilities=["analyze", "summarize"])
        self.agent2 = AgentIdentity(name="critic", role="reviewer", capabilities=["critique", "summarize"])
        self.agent3 = AgentIdentity(name="leader", role="coordinator", capabilities=["coordinate"])

    def test_register_and_list(self):
        self.registry.register(self.agent1)
        self.registry.register(self.agent2)
        active = self.registry.get_active()
        assert len(active) == 2
        names = [a.name for a in active]
        assert "analyst" in names
        assert "critic" in names

    def test_deregister(self):
        self.registry.register(self.agent1)
        self.registry.register(self.agent2)
        self.registry.deregister("analyst")
        active = self.registry.get_active()
        assert len(active) == 1
        assert active[0].name == "critic"

    def test_get_by_role(self):
        self.registry.register(self.agent1)
        self.registry.register(self.agent2)
        self.registry.register(self.agent3)
        researchers = self.registry.get_by_role("researcher")
        assert len(researchers) == 1
        assert researchers[0].name == "analyst"

    def test_get_by_capability(self):
        self.registry.register(self.agent1)
        self.registry.register(self.agent2)
        self.registry.register(self.agent3)
        summarizers = self.registry.get_by_capability("summarize")
        assert len(summarizers) == 2
        names = [a.name for a in summarizers]
        assert "analyst" in names
        assert "critic" in names

    def test_update_status(self):
        self.registry.register(self.agent1)
        self.registry.update_status("analyst", "inactive")
        active = self.registry.get_active()
        assert len(active) == 0
        assert self.registry.agents["analyst"].status == "inactive"

    def test_detect_stuck(self):
        self.registry.register(self.agent1)
        # Manually set last_active to a time far in the past
        self.registry.agents["analyst"].last_active = time.time() - 1000
        stuck = self.registry.detect_stuck(threshold_seconds=900)
        assert "analyst" in stuck

    def test_detect_stuck_active_agent(self):
        self.registry.register(self.agent1)
        # Agent was just registered, last_active is recent
        stuck = self.registry.detect_stuck(threshold_seconds=900)
        assert "analyst" not in stuck


class TestRegistryHeartbeatFields:
    def setup_method(self):
        self.registry = AgentRegistry()
        self.agent = AgentIdentity(name="phi3", role="worker")
        self.registry.register(self.agent)

    def test_update_heartbeat_state(self):
        self.registry.update_heartbeat_state("phi3", "ready")
        assert self.registry.get_heartbeat_state("phi3") == "ready"

    def test_update_heartbeat_state_unknown_agent(self):
        self.registry.update_heartbeat_state("unknown", "ready")
        assert self.registry.get_heartbeat_state("unknown") is None

    def test_get_heartbeat_state_default(self):
        assert self.registry.get_heartbeat_state("phi3") is None

    def test_get_agents_by_heartbeat_state(self):
        self.registry.register(AgentIdentity(name="mistral", role="critic"))
        self.registry.update_heartbeat_state("phi3", "ready")
        self.registry.update_heartbeat_state("mistral", "sleeping")
        ready = self.registry.get_agents_by_heartbeat_state("ready")
        assert len(ready) == 1
        assert ready[0].name == "phi3"
