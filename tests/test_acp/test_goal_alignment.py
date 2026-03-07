"""Tests for ACP Goal Alignment"""
import pytest
from src.ai_congress.core.acp.goal_alignment import (
    Mission, SessionGoal, AgentObjective, GoalAlignmentEngine,
)
from src.ai_congress.core.acp.roles import AgentRole


class TestMission:
    def test_mission_creation(self):
        mission = Mission(
            id="test-v1",
            statement="Provide balanced answers",
            principles=["accuracy", "diversity"],
            constraints=["no_hallucination"],
            priority_weights={"accuracy": 0.6, "diversity": 0.4},
        )
        assert mission.id == "test-v1"
        assert mission.statement == "Provide balanced answers"
        assert len(mission.principles) == 2
        assert len(mission.constraints) == 1
        assert mission.priority_weights["accuracy"] == 0.6


class TestSessionGoal:
    def test_session_goal_creation(self):
        goal = SessionGoal(
            session_id="sess-1",
            mission_id="test-v1",
            focus_areas=["math", "reasoning"],
            alignment_score=0.85,
        )
        assert goal.session_id == "sess-1"
        assert goal.alignment_score == 0.85


class TestAgentObjective:
    def test_agent_objective_creation(self):
        obj = AgentObjective(
            agent_name="phi3:3.8b",
            role=AgentRole.CRITIC,
            primary_goal="Find flaws in reasoning",
            alignment_context="Mission: balanced answers. Principles: accuracy, diversity.",
            constraints=["no_hallucination"],
        )
        assert obj.agent_name == "phi3:3.8b"
        assert obj.role == AgentRole.CRITIC


class TestGoalAlignmentEngine:
    def setup_method(self):
        self.mission = Mission(
            id="congress-v1",
            statement="Provide balanced, well-reasoned answers through democratic deliberation",
            principles=["accuracy", "diversity_of_thought", "evidence_based", "transparency"],
            constraints=["no_hallucination", "cite_sources", "acknowledge_uncertainty"],
            priority_weights={"accuracy": 0.4, "diversity": 0.3, "speed": 0.3},
        )
        self.engine = GoalAlignmentEngine(self.mission)

    def test_load_mission_from_dict(self):
        data = {
            "mission": {
                "id": "from-dict",
                "statement": "Test mission",
                "principles": ["a", "b"],
                "constraints": ["c"],
                "priority_weights": {"a": 1.0},
            }
        }
        mission = GoalAlignmentEngine.mission_from_dict(data)
        assert mission.id == "from-dict"
        assert mission.statement == "Test mission"

    def test_create_session_goal(self):
        goal = self.engine.create_session_goal("sess-1")
        assert goal.session_id == "sess-1"
        assert goal.mission_id == "congress-v1"
        assert goal.alignment_score == 1.0

    def test_create_agent_objective_for_critic(self):
        obj = self.engine.create_agent_objective("phi3:3.8b", AgentRole.CRITIC)
        assert obj.agent_name == "phi3:3.8b"
        assert obj.role == AgentRole.CRITIC
        assert "flaw" in obj.primary_goal.lower() or "review" in obj.primary_goal.lower()
        assert self.mission.statement in obj.alignment_context
        assert obj.constraints == self.mission.constraints

    def test_create_agent_objective_for_judge(self):
        obj = self.engine.create_agent_objective("mistral:7b", AgentRole.JUDGE)
        assert obj.role == AgentRole.JUDGE
        assert self.mission.statement in obj.alignment_context

    def test_create_agent_objective_for_worker(self):
        obj = self.engine.create_agent_objective("llama3.2:3b", AgentRole.WORKER)
        assert obj.role == AgentRole.WORKER

    def test_create_agent_objective_for_planner(self):
        obj = self.engine.create_agent_objective("qwen3:8b", AgentRole.PLANNER)
        assert obj.role == AgentRole.PLANNER

    def test_create_agent_objective_for_synthesizer(self):
        obj = self.engine.create_agent_objective("deepseek-r1:1.5b", AgentRole.SYNTHESIZER)
        assert obj.role == AgentRole.SYNTHESIZER

    def test_build_system_prompt_contains_mission(self):
        obj = self.engine.create_agent_objective("phi3:3.8b", AgentRole.WORKER)
        prompt = self.engine.build_alignment_prompt(obj)
        assert "MISSION CONTEXT" in prompt
        assert self.mission.statement in prompt
        assert "accuracy" in prompt
        assert "no_hallucination" in prompt
        assert "worker" in prompt.lower() or obj.primary_goal in prompt

    def test_score_alignment_perfect(self):
        score = self.engine.score_alignment(
            response="Paris is the capital of France, based on geographic data.",
            mission=self.mission,
        )
        assert 0.0 <= score <= 1.0

    def test_score_alignment_returns_float(self):
        score = self.engine.score_alignment(
            response="I don't know and I'm making this up entirely.",
            mission=self.mission,
        )
        assert isinstance(score, float)
