"""
RoleDispatcher - Assigns explicit agent roles based on personality and benchmark weights.

Maps personality traits to functional roles (Planner, Worker, Critic, Judge, Synthesizer)
for structured multi-agent coordination.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .message import AgentIdentity, PersonalityProfile
from .registry import AgentRegistry

logger = logging.getLogger(__name__)


class AgentRole(str, Enum):
    PLANNER = "planner"           # Decomposes query into sub-questions
    WORKER = "worker"             # Generates responses
    CRITIC = "critic"             # Reviews and critiques responses
    JUDGE = "judge"               # Clusters and selects final answer
    SYNTHESIZER = "synthesizer"   # Merges multiple responses


@dataclass
class RoleAssignment:
    model_name: str
    role: AgentRole
    score: float  # How well the model fits this role (0-1)


class RoleDispatcher:
    """Assigns models to roles based on personality traits + benchmark weights."""

    # Role-to-trait mapping: which Big Five traits favor each role
    ROLE_TRAIT_WEIGHTS = {
        AgentRole.PLANNER: {
            "openness": 0.4,
            "conscientiousness": 0.3,
            "leadership": 0.3,
        },
        AgentRole.WORKER: {
            "conscientiousness": 0.4,
            "engagement": 0.3,
            "openness": 0.3,
        },
        AgentRole.CRITIC: {
            "extraversion": 0.3,
            "openness": 0.3,
            "neuroticism": 0.2,  # Higher neuroticism = more critical eye
            "conscientiousness": 0.2,
        },
        AgentRole.JUDGE: {
            "conscientiousness": 0.4,
            "leadership": 0.4,
            "neuroticism": -0.2,  # Low neuroticism = calm judgment
        },
        AgentRole.SYNTHESIZER: {
            "agreeableness": 0.3,
            "openness": 0.3,
            "extraversion": 0.2,
            "empathy_level": 0.2,
        },
    }

    def __init__(self, registry: AgentRegistry):
        self.registry = registry

    def _score_for_role(
        self, profile: PersonalityProfile, role: AgentRole, benchmark_weight: float = 0.5
    ) -> float:
        """Score how well a personality fits a given role."""
        trait_weights = self.ROLE_TRAIT_WEIGHTS[role]
        score = 0.0
        for trait_name, weight in trait_weights.items():
            trait_val = getattr(profile, trait_name, 5)
            if isinstance(trait_val, float) and trait_val <= 1.0:
                normalized = trait_val
            else:
                normalized = trait_val / 10.0

            if weight < 0:
                # Negative weight: invert the trait (low value = high contribution)
                score += (1.0 - normalized) * abs(weight)
            else:
                score += normalized * weight

        # Incorporate benchmark weight (higher benchmark = better at any role)
        score = score * 0.6 + benchmark_weight * 0.4
        return max(0.0, min(1.0, score))

    def assign_roles(
        self,
        models: list[str],
        profiles: dict[str, PersonalityProfile],
        weights: dict[str, float],
    ) -> dict[AgentRole, list[RoleAssignment]]:
        """Assign models to roles based on personality and benchmark weights.

        Each model is assigned to the role it scores highest for.
        Returns a mapping of role -> list of assignments.
        """
        role_assignments: dict[AgentRole, list[RoleAssignment]] = {
            role: [] for role in AgentRole
        }

        for model in models:
            profile = profiles.get(model, PersonalityProfile())
            benchmark_w = weights.get(model, 0.5)

            # Score this model for each role
            scores = {}
            for role in AgentRole:
                scores[role] = self._score_for_role(profile, role, benchmark_w)

            # Assign to highest-scoring role
            best_role = max(scores, key=scores.get)
            assignment = RoleAssignment(
                model_name=model,
                role=best_role,
                score=scores[best_role],
            )
            role_assignments[best_role].append(assignment)

        # Ensure at least one judge (pick highest benchmark weight if none)
        if not role_assignments[AgentRole.JUDGE] and models:
            best_model = max(models, key=lambda m: weights.get(m, 0.5))
            # Move from current role
            for role, assignments in role_assignments.items():
                role_assignments[role] = [a for a in assignments if a.model_name != best_model]
            role_assignments[AgentRole.JUDGE].append(
                RoleAssignment(model_name=best_model, role=AgentRole.JUDGE, score=1.0)
            )

        # Ensure at least one worker
        if not role_assignments[AgentRole.WORKER] and models:
            # Promote lowest-scoring non-worker
            all_assigned = [
                (a, role)
                for role, assignments in role_assignments.items()
                for a in assignments
                if role != AgentRole.JUDGE
            ]
            if all_assigned:
                lowest = min(all_assigned, key=lambda x: x[0].score)
                role_assignments[lowest[1]] = [
                    a for a in role_assignments[lowest[1]]
                    if a.model_name != lowest[0].model_name
                ]
                role_assignments[AgentRole.WORKER].append(
                    RoleAssignment(
                        model_name=lowest[0].model_name,
                        role=AgentRole.WORKER,
                        score=lowest[0].score,
                    )
                )

        return role_assignments

    def get_models_for_role(
        self,
        assignments: dict[AgentRole, list[RoleAssignment]],
        role: AgentRole,
    ) -> list[str]:
        """Get model names assigned to a specific role."""
        return [a.model_name for a in assignments.get(role, [])]

    def get_role_summary(
        self, assignments: dict[AgentRole, list[RoleAssignment]]
    ) -> dict:
        """Get a human-readable summary of role assignments."""
        return {
            role.value: [
                {"model": a.model_name, "score": round(a.score, 3)}
                for a in assignments_list
            ]
            for role, assignments_list in assignments.items()
        }
