"""Org Chart - Persistent organizational hierarchy for the AI Congress."""

import time
from dataclasses import dataclass, field
from typing import Optional

from .roles import AgentRole


_ROLE_MAP = {
    "planner": AgentRole.PLANNER,
    "worker": AgentRole.WORKER,
    "critic": AgentRole.CRITIC,
    "judge": AgentRole.JUDGE,
    "synthesizer": AgentRole.SYNTHESIZER,
}


@dataclass
class Position:
    title: str
    role: AgentRole
    rank: int
    reports_to: Optional[str] = None
    responsibilities: list[str] = field(default_factory=list)
    authority: list[str] = field(default_factory=list)


@dataclass
class OrgAssignment:
    agent_name: str
    position: Position
    appointed_at: float = field(default_factory=time.time)
    performance_score: float = 0.5
    tenure_queries: int = 0


class OrgChart:
    """Persistent organizational structure for the Congress."""

    RANK_BONUSES = {1: 1.1, 2: 1.05, 3: 1.0}

    def __init__(self):
        self._positions: dict[str, Position] = {}
        self._assignments: dict[str, OrgAssignment] = {}

    def define_default_structure(self) -> None:
        self._positions = {
            "Speaker of the House": Position(
                title="Speaker of the House",
                role=AgentRole.JUDGE,
                rank=1,
                reports_to=None,
                responsibilities=["Final consensus approval", "Debate moderation", "Veto power"],
                authority=["can_veto", "can_escalate", "can_delegate"],
            ),
            "Committee Chair: Research": Position(
                title="Committee Chair: Research",
                role=AgentRole.PLANNER,
                rank=2,
                reports_to="Speaker of the House",
                responsibilities=["Query decomposition", "Research direction"],
                authority=["can_delegate", "can_escalate"],
            ),
            "Committee Chair: Oversight": Position(
                title="Committee Chair: Oversight",
                role=AgentRole.CRITIC,
                rank=2,
                reports_to="Speaker of the House",
                responsibilities=["Response quality review", "Fact-checking"],
                authority=["can_escalate"],
            ),
            "Floor Leader": Position(
                title="Floor Leader",
                role=AgentRole.SYNTHESIZER,
                rank=2,
                reports_to="Speaker of the House",
                responsibilities=["Merge responses", "Bridge between factions"],
                authority=["can_escalate"],
            ),
        }

    def add_position(self, position: Position) -> None:
        self._positions[position.title] = position

    def appoint(self, agent_name: str, title: str) -> OrgAssignment:
        if title not in self._positions:
            raise KeyError(f"Position '{title}' not found in org chart")
        assignment = OrgAssignment(
            agent_name=agent_name,
            position=self._positions[title],
        )
        self._assignments[agent_name] = assignment
        return assignment

    def get_chain_of_command(self, agent_name: str) -> list[OrgAssignment]:
        if agent_name not in self._assignments:
            return []
        chain = []
        current = self._assignments[agent_name]
        chain.append(current)
        while current.position.reports_to:
            supervisor_title = current.position.reports_to
            supervisor = next(
                (a for a in self._assignments.values() if a.position.title == supervisor_title),
                None,
            )
            if supervisor is None or supervisor in chain:
                break
            chain.append(supervisor)
            current = supervisor
        return chain

    def get_direct_reports(self, agent_name: str) -> list[OrgAssignment]:
        if agent_name not in self._assignments:
            return []
        my_title = self._assignments[agent_name].position.title
        return [
            a for a in self._assignments.values()
            if a.position.reports_to == my_title
        ]

    def can_perform(self, agent_name: str, authority: str) -> bool:
        if agent_name not in self._assignments:
            return False
        return authority in self._assignments[agent_name].position.authority

    def get_rank_bonus(self, agent_name: str) -> float:
        if agent_name not in self._assignments:
            return 1.0
        rank = self._assignments[agent_name].position.rank
        return self.RANK_BONUSES.get(rank, 1.0)

    def increment_tenure(self, agent_name: str) -> None:
        if agent_name in self._assignments:
            self._assignments[agent_name].tenure_queries += 1

    def update_performance(self, agent_name: str, score: float) -> None:
        if agent_name in self._assignments:
            self._assignments[agent_name].performance_score = score

    def rotate(self, performance_threshold: float = 0.4) -> list[tuple[str, str, str]]:
        rotations = []
        for agent_name, assignment in self._assignments.items():
            if assignment.performance_score < performance_threshold:
                rotations.append((
                    agent_name,
                    assignment.position.title,
                    "underperforming",
                ))
        return rotations

    def get_all_assignments(self) -> dict[str, OrgAssignment]:
        return dict(self._assignments)

    def get_position_for_agent(self, agent_name: str) -> Optional[Position]:
        assignment = self._assignments.get(agent_name)
        return assignment.position if assignment else None
