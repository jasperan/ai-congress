"""
Integration tests for Paperclip systems wired into EnhancedOrchestrator.
"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from src.ai_congress.core.enhanced_orchestrator import EnhancedOrchestrator
from src.ai_congress.core.acp.audit_trail import AuditEventType


def _make_orchestrator():
    """Create an EnhancedOrchestrator with mocked dependencies."""
    model_registry = MagicMock()
    model_registry.get_model_weight = MagicMock(return_value=0.5)
    model_registry.list_models = MagicMock(return_value=["phi3", "mistral"])

    voting_engine = MagicMock()
    ollama_client = AsyncMock()
    personality_loader = MagicMock()
    personality_loader.get_profile = MagicMock(return_value=MagicMock(
        communication_style="formal",
        extraversion=3.0,
    ))

    orch = EnhancedOrchestrator(
        model_registry=model_registry,
        voting_engine=voting_engine,
        ollama_client=ollama_client,
        personality_loader=personality_loader,
    )
    return orch


class TestPaperclipIntegration:
    def test_orchestrator_has_audit_trail(self):
        orch = _make_orchestrator()
        assert orch.audit_trail is not None
        assert len(orch.audit_trail._events) == 0

    def test_orchestrator_has_org_chart(self):
        orch = _make_orchestrator()
        assert orch.org_chart is not None
        # Default structure should be defined
        assert len(orch.org_chart._positions) > 0

    def test_orchestrator_goal_engine_initially_none(self):
        orch = _make_orchestrator()
        assert orch.goal_engine is None

    def test_load_mission(self):
        orch = _make_orchestrator()
        mission_data = {
            "mission": {
                "id": "test-mission",
                "statement": "Test mission statement",
                "principles": ["accuracy"],
                "constraints": [],
                "priority_weights": {"accuracy": 1.0},
            }
        }
        orch.load_mission(mission_data)
        assert orch.goal_engine is not None
        assert orch.goal_engine.mission.id == "test-mission"
        # Should have recorded a MISSION_LOADED audit event
        events = [e for e in orch.audit_trail._events if e.event_type == AuditEventType.MISSION_LOADED]
        assert len(events) == 1

    def test_register_agents_creates_audit_events(self):
        orch = _make_orchestrator()
        orch._register_agents(["phi3", "mistral"])
        events = [e for e in orch.audit_trail._events if e.event_type == AuditEventType.AGENT_REGISTERED]
        assert len(events) == 2
        agent_names = [e.agent_name for e in events]
        assert "phi3" in agent_names
        assert "mistral" in agent_names

    def test_message_bus_has_audit_trail(self):
        orch = _make_orchestrator()
        assert orch.message_bus._audit_trail is orch.audit_trail

    def test_message_bus_has_org_chart(self):
        orch = _make_orchestrator()
        assert orch.message_bus._org_chart is orch.org_chart

    def test_org_chart_rank_bonus_default(self):
        orch = _make_orchestrator()
        # Unassigned agent gets default bonus of 1.0
        assert orch.org_chart.get_rank_bonus("some_random_model") == 1.0

    def test_heartbeat_manager_initially_none(self):
        orch = _make_orchestrator()
        assert orch.heartbeat_manager is None
