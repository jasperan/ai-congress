"""
Tests for ACP Coordination Controller
"""
import pytest
from unittest.mock import patch
from src.ai_congress.core.acp.message import AgentIdentity, PersonalityProfile
from src.ai_congress.core.acp.coordination import CoordinationController


class TestCoordinationController:
    def test_none_level_blocks_all(self):
        controller = CoordinationController(level="none")
        agent = AgentIdentity(name="agent1", role="analyst")
        # With probability 0.0, should never communicate
        for _ in range(100):
            assert controller.should_communicate(agent, {}) is False

    def test_chatty_level_high_probability(self):
        controller = CoordinationController(level="chatty")
        agent = AgentIdentity(name="agent1", role="analyst")
        # With probability 0.9 and no personality modifier, most calls should return True
        results = [controller.should_communicate(agent, {}) for _ in range(1000)]
        true_count = sum(results)
        # Should be roughly 900 out of 1000 (allow wide margin)
        assert true_count > 700

    def test_message_limit_enforcement(self):
        controller = CoordinationController(level="minimal")
        # max_messages for minimal is 2
        assert controller.can_send_more("agent1", "round1") is True
        controller.record_message("agent1", "round1")
        assert controller.can_send_more("agent1", "round1") is True
        controller.record_message("agent1", "round1")
        assert controller.can_send_more("agent1", "round1") is False

    def test_message_limit_per_round(self):
        controller = CoordinationController(level="minimal")
        # Fill up round1
        controller.record_message("agent1", "round1")
        controller.record_message("agent1", "round1")
        assert controller.can_send_more("agent1", "round1") is False
        # round2 should still be available
        assert controller.can_send_more("agent1", "round2") is True

    def test_extraversion_modulates_probability(self):
        controller = CoordinationController(level="moderate")
        # High extraversion agent (10) should communicate more often
        high_ext_profile = PersonalityProfile(extraversion=10)
        high_ext_agent = AgentIdentity(name="extrovert", role="analyst", personality=high_ext_profile)

        # Low extraversion agent (1) should communicate less often
        low_ext_profile = PersonalityProfile(extraversion=1)
        low_ext_agent = AgentIdentity(name="introvert", role="analyst", personality=low_ext_profile)

        high_results = [controller.should_communicate(high_ext_agent, {}) for _ in range(1000)]
        low_results = [controller.should_communicate(low_ext_agent, {}) for _ in range(1000)]

        high_count = sum(high_results)
        low_count = sum(low_results)
        # High extraversion should communicate significantly more
        assert high_count > low_count

    def test_get_max_messages(self):
        assert CoordinationController(level="none").get_max_messages() == 0
        assert CoordinationController(level="minimal").get_max_messages() == 2
        assert CoordinationController(level="moderate").get_max_messages() == 5
        assert CoordinationController(level="chatty").get_max_messages() == 10

    def test_no_personality_uses_base_probability(self):
        controller = CoordinationController(level="moderate")
        agent = AgentIdentity(name="agent1", role="analyst")  # no personality
        # With base probability 0.6, should get roughly 600 out of 1000
        results = [controller.should_communicate(agent, {}) for _ in range(1000)]
        true_count = sum(results)
        assert 400 < true_count < 800

    def test_invalid_level_raises(self):
        with pytest.raises(ValueError, match="Invalid coordination level"):
            CoordinationController(level="invalid")

    def test_reset_round(self):
        controller = CoordinationController(level="minimal")
        controller.record_message("agent1", "round1")
        controller.record_message("agent1", "round1")
        assert controller.can_send_more("agent1", "round1") is False
        controller.reset_round("round1")
        assert controller.can_send_more("agent1", "round1") is True
