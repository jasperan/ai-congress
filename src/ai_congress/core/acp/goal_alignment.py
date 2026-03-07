"""Goal Alignment - Hierarchical mission cascading for agent objectives."""

from dataclasses import dataclass, field
from typing import Optional

from .roles import AgentRole


@dataclass
class Mission:
    id: str
    statement: str
    principles: list[str]
    constraints: list[str]
    priority_weights: dict[str, float] = field(default_factory=dict)


@dataclass
class SessionGoal:
    session_id: str
    mission_id: str
    focus_areas: list[str] = field(default_factory=list)
    alignment_score: float = 1.0


@dataclass
class AgentObjective:
    agent_name: str
    role: AgentRole
    primary_goal: str
    alignment_context: str
    constraints: list[str] = field(default_factory=list)


_ROLE_GOALS = {
    AgentRole.PLANNER: "Decompose the query into focused sub-questions that enable comprehensive coverage",
    AgentRole.WORKER: "Research and provide a thorough, evidence-based response",
    AgentRole.CRITIC: "Review responses for flaws, gaps, and unsupported claims",
    AgentRole.JUDGE: "Evaluate all responses fairly and select the most accurate consensus",
    AgentRole.SYNTHESIZER: "Merge the strongest elements from multiple responses into a coherent answer",
}


class GoalAlignmentEngine:
    """Creates and manages hierarchical goal alignment from mission to agent objectives."""

    def __init__(self, mission: Mission):
        self.mission = mission

    @staticmethod
    def mission_from_dict(data: dict) -> "Mission":
        m = data["mission"]
        return Mission(
            id=m["id"],
            statement=m["statement"],
            principles=m.get("principles", []),
            constraints=m.get("constraints", []),
            priority_weights=m.get("priority_weights", {}),
        )

    def create_session_goal(self, session_id: str) -> SessionGoal:
        return SessionGoal(
            session_id=session_id,
            mission_id=self.mission.id,
            alignment_score=1.0,
        )

    def create_agent_objective(self, agent_name: str, role: AgentRole) -> AgentObjective:
        primary_goal = _ROLE_GOALS.get(role, "Provide a helpful response")
        principles_str = ", ".join(self.mission.principles)
        alignment_context = (
            f"Mission: {self.mission.statement}. "
            f"Principles: {principles_str}."
        )
        return AgentObjective(
            agent_name=agent_name,
            role=role,
            primary_goal=primary_goal,
            alignment_context=alignment_context,
            constraints=list(self.mission.constraints),
        )

    def build_alignment_prompt(self, objective: AgentObjective) -> str:
        constraints_str = ", ".join(objective.constraints)
        return (
            f"[MISSION CONTEXT]\n"
            f"You are part of the AI Congress. Our mission: {self.mission.statement}\n"
            f"Core principles: {', '.join(self.mission.principles)}\n"
            f"Constraints: {constraints_str}\n"
            f"Your role ({objective.role.value}): {objective.primary_goal}\n"
            f"Ensure your response aligns with these principles."
        )

    def score_alignment(self, response: str, mission: Optional["Mission"] = None) -> float:
        m = mission or self.mission
        score = 1.0
        lower = response.lower()
        if any(kw in lower for kw in ["i don't know", "making this up", "no idea"]):
            score -= 0.2
        if any(kw in lower for kw in ["based on", "evidence", "according to", "data"]):
            score += 0.1
        if any(kw in lower for kw in ["however", "on the other hand", "alternatively"]):
            score += 0.05
        return max(0.0, min(1.0, score))
