"""Tests for ACP Org Chart"""
import time
import pytest
from src.ai_congress.core.acp.org_chart import OrgChart, Position, OrgAssignment
from src.ai_congress.core.acp.roles import AgentRole


class TestPosition:
    def test_position_creation(self):
        pos = Position(
            title="Speaker of the House",
            role=AgentRole.JUDGE,
            rank=1,
            reports_to=None,
            responsibilities=["Final consensus approval"],
            authority=["can_veto", "can_escalate"],
        )
        assert pos.title == "Speaker of the House"
        assert pos.role == AgentRole.JUDGE
        assert pos.rank == 1
        assert pos.reports_to is None
        assert "can_veto" in pos.authority


class TestOrgAssignment:
    def test_assignment_creation(self):
        pos = Position(title="Member", role=AgentRole.WORKER, rank=3, reports_to="Committee Chair: Research")
        assignment = OrgAssignment(agent_name="phi3:3.8b", position=pos)
        assert assignment.agent_name == "phi3:3.8b"
        assert assignment.position.title == "Member"
        assert assignment.performance_score == 0.5
        assert assignment.tenure_queries == 0


class TestOrgChart:
    def setup_method(self):
        self.chart = OrgChart()
        self.chart.define_default_structure()

    def test_default_structure_has_speaker(self):
        assert "Speaker of the House" in self.chart._positions

    def test_default_structure_has_committees(self):
        titles = list(self.chart._positions.keys())
        assert any("Research" in t for t in titles)
        assert any("Oversight" in t for t in titles)

    def test_default_structure_has_floor_leader(self):
        assert "Floor Leader" in self.chart._positions

    def test_speaker_is_rank_1(self):
        speaker = self.chart._positions["Speaker of the House"]
        assert speaker.rank == 1

    def test_committee_chairs_report_to_speaker(self):
        for title, pos in self.chart._positions.items():
            if pos.rank == 2:
                assert pos.reports_to == "Speaker of the House"

    def test_appoint_agent(self):
        assignment = self.chart.appoint("mistral:7b", "Speaker of the House")
        assert assignment.agent_name == "mistral:7b"
        assert assignment.position.title == "Speaker of the House"

    def test_appoint_unknown_position_raises(self):
        with pytest.raises(KeyError):
            self.chart.appoint("phi3:3.8b", "Nonexistent Title")

    def test_get_chain_of_command(self):
        self.chart.appoint("mistral:7b", "Speaker of the House")
        self.chart.appoint("phi3:3.8b", "Committee Chair: Research")
        chain = self.chart.get_chain_of_command("phi3:3.8b")
        assert len(chain) == 2
        titles = [a.position.title for a in chain]
        assert "Committee Chair: Research" in titles
        assert "Speaker of the House" in titles

    def test_get_chain_of_command_speaker_is_root(self):
        self.chart.appoint("mistral:7b", "Speaker of the House")
        chain = self.chart.get_chain_of_command("mistral:7b")
        assert len(chain) == 1
        assert chain[0].position.title == "Speaker of the House"

    def test_get_direct_reports(self):
        self.chart.appoint("mistral:7b", "Speaker of the House")
        self.chart.appoint("phi3:3.8b", "Committee Chair: Research")
        self.chart.appoint("llama3.2:3b", "Committee Chair: Oversight")
        reports = self.chart.get_direct_reports("mistral:7b")
        report_names = [r.agent_name for r in reports]
        assert "phi3:3.8b" in report_names
        assert "llama3.2:3b" in report_names

    def test_can_perform_authority(self):
        self.chart.appoint("mistral:7b", "Speaker of the House")
        assert self.chart.can_perform("mistral:7b", "can_veto") is True
        assert self.chart.can_perform("mistral:7b", "nonexistent_authority") is False

    def test_can_perform_unassigned_agent(self):
        assert self.chart.can_perform("unknown_agent", "can_veto") is False

    def test_get_rank_bonus(self):
        self.chart.appoint("mistral:7b", "Speaker of the House")
        self.chart.appoint("phi3:3.8b", "Committee Chair: Research")
        assert self.chart.get_rank_bonus("mistral:7b") == 1.1
        assert self.chart.get_rank_bonus("phi3:3.8b") == 1.05
        assert self.chart.get_rank_bonus("unassigned") == 1.0

    def test_increment_tenure(self):
        self.chart.appoint("mistral:7b", "Speaker of the House")
        self.chart.increment_tenure("mistral:7b")
        assert self.chart._assignments["mistral:7b"].tenure_queries == 1

    def test_update_performance(self):
        self.chart.appoint("mistral:7b", "Speaker of the House")
        self.chart.update_performance("mistral:7b", 0.9)
        assert self.chart._assignments["mistral:7b"].performance_score == 0.9

    def test_rotate_underperforming(self):
        self.chart.appoint("mistral:7b", "Speaker of the House")
        self.chart.update_performance("mistral:7b", 0.3)
        self.chart.appoint("phi3:3.8b", "Committee Chair: Research")
        self.chart.update_performance("phi3:3.8b", 0.8)
        rotations = self.chart.rotate(performance_threshold=0.4)
        rotated_agents = [r[0] for r in rotations]
        assert "mistral:7b" in rotated_agents

    def test_get_all_assignments(self):
        self.chart.appoint("mistral:7b", "Speaker of the House")
        self.chart.appoint("phi3:3.8b", "Committee Chair: Research")
        assignments = self.chart.get_all_assignments()
        assert len(assignments) == 2

    def test_get_position_for_agent(self):
        self.chart.appoint("mistral:7b", "Speaker of the House")
        pos = self.chart.get_position_for_agent("mistral:7b")
        assert pos is not None
        assert pos.title == "Speaker of the House"
        assert self.chart.get_position_for_agent("unknown") is None
