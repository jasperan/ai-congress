"""
DebateManager - Multi-round debate with patience mechanic, indecisiveness
detection, and conviction bonus for ensemble LLM decision-making.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from src.ai_congress.core.semantic_voting import (
    Cluster,
    ModelResponse,
    SemanticVoteResult,
    SemanticVotingEngine,
)

logger = logging.getLogger(__name__)

PRESSURE_PROMPTS = [
    # Round 0 - collaborative
    (
        "Consider the other perspectives presented below. If you see genuine merit "
        "in another position, revise your answer. If you still believe your original "
        "answer is correct, strengthen your argument with additional reasoning."
    ),
    # Round 1 - directive
    (
        "Take a clear position. Hedging weakens the group's ability to reach "
        "consensus. State which position you support and provide your strongest "
        "argument for it. Do not equivocate."
    ),
    # Round 2 - final
    (
        "This is the final round. Commit fully to your strongest position with "
        "maximum conviction. No caveats, no hedging, no 'on the other hand'. "
        "Give your definitive answer."
    ),
]

INDECISIVE_NOTE = (
    "\n\nNOTE: Your previous responses have been inconsistent or uncommitted. "
    "You MUST choose a clear position this round."
)


@dataclass
class DebateConfig:
    max_rounds: int = 3
    consensus_threshold: float = 0.6
    temp_schedule: List[float] = field(default_factory=lambda: [0.9, 0.5, 0.2])
    conviction_bonus: float = 1.2

    def pressure_prompt(self, round_num: int, is_indecisive: bool) -> str:
        """Return escalating pressure prompt based on round."""
        idx = min(round_num, len(PRESSURE_PROMPTS) - 1)
        prompt = PRESSURE_PROMPTS[idx]
        if is_indecisive:
            prompt += INDECISIVE_NOTE
        return prompt


class DebateManager:
    """Manages multi-round debates between LLM models to reach consensus."""

    def __init__(
        self,
        ollama_client,
        voting_engine: SemanticVotingEngine,
        config: Optional[DebateConfig] = None,
    ):
        self.ollama_client = ollama_client
        self.voting_engine = voting_engine
        self.config = config or DebateConfig()

    def detect_indecisive(
        self, position_history: Dict[str, List[int]], round_num: int
    ) -> Set[str]:
        """Models that switched clusters between rounds."""
        indecisive = set()
        for model, positions in position_history.items():
            if len(positions) >= 2 and len(set(positions)) > 1:
                indecisive.add(model)
        return indecisive

    def detect_indecisive_from_clusters(self, clusters: List[Cluster]) -> Set[str]:
        """Models in singleton clusters (not aligned with anyone)."""
        indecisive = set()
        for cluster in clusters:
            if len(cluster.models) == 1:
                indecisive.add(cluster.models[0])
        return indecisive

    def apply_conviction_bonus(
        self,
        weights: Dict[str, float],
        position_history: Dict[str, List[int]],
        bonus: float = 1.2,
    ) -> Dict[str, float]:
        """Multiply weight for models that held consistent positions across all rounds."""
        adjusted = dict(weights)
        for model, positions in position_history.items():
            if len(positions) >= 2 and len(set(positions)) == 1:
                adjusted[model] = adjusted.get(model, 0.0) * bonus
        return adjusted

    def _build_debate_prompt(
        self,
        original_query: str,
        clusters: List[Cluster],
        analysis: str,
        round_num: int,
        is_indecisive: bool,
    ) -> str:
        """Build full debate prompt including cluster positions, analysis, and pressure prompt."""
        parts = [
            f"Original question: {original_query}\n",
            "Current positions from all models:\n",
        ]

        for cluster in clusters:
            parts.append(f"  Cluster '{cluster.label}' ({', '.join(cluster.models)}):")
            for model, text in cluster.responses.items():
                parts.append(f"    [{model}]: {text}")
            if cluster.key_claims:
                parts.append(f"    Key claims: {', '.join(cluster.key_claims)}")
            parts.append("")

        if analysis:
            parts.append(f"Analysis: {analysis}\n")

        pressure = self.config.pressure_prompt(round_num, is_indecisive)
        parts.append(pressure)

        return "\n".join(parts)

    def _find_model_cluster(self, model: str, clusters: List[Cluster]) -> int:
        """Find which cluster ID a model belongs to."""
        for cluster in clusters:
            if model in cluster.models:
                return cluster.id
        return -1

    async def run_debate(
        self,
        original_query: str,
        responses: List[ModelResponse],
        initial_clusters: List[Cluster],
        initial_analysis: str,
    ) -> SemanticVoteResult:
        """Run multi-round debate until consensus or patience exhausted.

        Flow per round:
        1. Detect indecisive models (switched clusters OR singleton)
        2. Build debate prompts for each model (with escalating pressure for indecisive)
        3. Query all models in parallel with adjusted temperatures
        4. Judge re-groups revised responses
        5. Update position history
        6. Check consensus -- if met, apply conviction bonus and return
        7. If patience exhausted, force-select by adjusted weight
        """
        config = self.config
        clusters = initial_clusters
        analysis = initial_analysis
        transcript: List[Dict] = []

        # Build initial weights and position history
        weights = {r.model_name: r.weight for r in responses}
        response_map = {r.model_name: r.response_text for r in responses}
        position_history: Dict[str, List[int]] = {}

        # Record initial positions
        for r in responses:
            cid = self._find_model_cluster(r.model_name, clusters)
            position_history[r.model_name] = [cid]

        debate_rounds = 0

        for round_num in range(config.max_rounds):
            debate_rounds = round_num + 1

            # 1. Detect indecisive models
            indecisive_switched = self.detect_indecisive(position_history, round_num)
            indecisive_singleton = self.detect_indecisive_from_clusters(clusters)
            all_indecisive = indecisive_switched | indecisive_singleton

            # 2 & 3. Build prompts and query all models in parallel
            temp_idx = min(round_num, len(config.temp_schedule) - 1)
            temperature = config.temp_schedule[temp_idx]

            tasks = []
            model_names = []
            for r in responses:
                is_indecisive = r.model_name in all_indecisive
                prompt = self._build_debate_prompt(
                    original_query, clusters, analysis, round_num, is_indecisive
                )
                task = self.ollama_client.chat(
                    model=r.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": temperature},
                )
                tasks.append(task)
                model_names.append(r.model_name)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Update response map with debate results
            round_responses = {}
            for model_name, result in zip(model_names, results):
                if isinstance(result, Exception):
                    logger.warning(
                        f"Model {model_name} failed in debate round {round_num}: {result}"
                    )
                    # Keep original response
                    round_responses[model_name] = response_map[model_name]
                else:
                    text = result["message"]["content"]
                    response_map[model_name] = text
                    round_responses[model_name] = text

            # Record transcript
            transcript.append(
                {"round": round_num, "responses": dict(round_responses)}
            )

            # 4. Judge re-groups revised responses
            revised_responses = [
                ModelResponse(
                    model_name=r.model_name,
                    response_text=response_map[r.model_name],
                    weight=r.weight,
                    temperature=temperature,
                )
                for r in responses
            ]
            clusters, analysis = await self.voting_engine.judge_group(revised_responses)

            # 5. Update position history
            for r in responses:
                cid = self._find_model_cluster(r.model_name, clusters)
                position_history[r.model_name].append(cid)

            # 6. Check consensus
            total_weight = sum(weights.values())
            scored = [(c, c.score(weights)) for c in clusters]
            scored.sort(key=lambda x: x[1], reverse=True)

            winning_cluster, winning_score = scored[0]
            consensus = winning_score / total_weight if total_weight > 0 else 0.0

            if consensus >= config.consensus_threshold:
                # Apply conviction bonus
                adjusted_weights = self.apply_conviction_bonus(
                    weights, position_history, config.conviction_bonus
                )
                best_model = max(
                    winning_cluster.models,
                    key=lambda m: adjusted_weights.get(m, 0.0),
                )
                return SemanticVoteResult(
                    winner=winning_cluster.responses.get(best_model, ""),
                    winning_model=best_model,
                    clusters=clusters,
                    consensus=consensus,
                    debate_triggered=True,
                    debate_rounds=debate_rounds,
                    debate_transcript=transcript,
                    dissenting_summary=analysis,
                    conviction_scores=adjusted_weights,
                )

        # 7. Patience exhausted — force-select by adjusted weight
        adjusted_weights = self.apply_conviction_bonus(
            weights, position_history, config.conviction_bonus
        )

        # Pick the best cluster by adjusted weight
        scored_adjusted = [
            (c, sum(adjusted_weights.get(m, 0.0) for m in c.models))
            for c in clusters
        ]
        scored_adjusted.sort(key=lambda x: x[1], reverse=True)
        winning_cluster = scored_adjusted[0][0]

        best_model = max(
            winning_cluster.models,
            key=lambda m: adjusted_weights.get(m, 0.0),
        )

        total_weight = sum(weights.values())
        consensus = winning_cluster.score(weights) / total_weight if total_weight > 0 else 0.0

        return SemanticVoteResult(
            winner=winning_cluster.responses.get(best_model, ""),
            winning_model=best_model,
            clusters=clusters,
            consensus=consensus,
            debate_triggered=True,
            debate_rounds=debate_rounds,
            debate_transcript=transcript,
            dissenting_summary=analysis,
            conviction_scores=adjusted_weights,
        )
