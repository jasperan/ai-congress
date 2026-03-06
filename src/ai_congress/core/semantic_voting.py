"""
Semantic Voting Engine - LLM-based semantic grouping for ensemble decisions.

Replaces simple string-matching vote grouping with a judge LLM that
clusters responses by meaning, enabling consensus detection even when
models phrase answers differently.
"""

import json
import re
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelResponse:
    model_name: str
    response_text: str
    weight: float
    temperature: float


@dataclass
class Cluster:
    id: int
    label: str
    models: List[str]
    key_claims: List[str]
    responses: Dict[str, str]  # model_name -> response_text

    def score(self, weights: Dict[str, float]) -> float:
        """Sum of model weights in this cluster."""
        return sum(weights.get(m, 0.0) for m in self.models)


@dataclass
class SemanticVoteResult:
    winner: str
    winning_model: str
    clusters: List[Cluster]
    consensus: float
    debate_triggered: bool
    debate_rounds: int
    debate_transcript: List[Dict]
    dissenting_summary: str
    conviction_scores: Dict[str, float]

    def to_dict(self) -> Dict:
        """Convert to dict including clusters as dicts."""
        d = asdict(self)
        return d


JUDGE_PROMPT_TEMPLATE = """You are a neutral judge analyzing AI model responses to a query.
Group these responses by their semantic meaning — responses that reach the same
conclusion with the same core reasoning go in the same group. Ignore phrasing
differences (e.g., "4" and "four" are the same, "great" and "excellent" are the same).

Responses:
{responses}

Output ONLY valid JSON with this structure (no markdown, no explanation):
{{"clusters": [{{"id": 1, "label": "brief description", "models": ["model_name"], "key_claims": ["claim"]}}], "analysis": "brief analysis"}}"""


class SemanticVotingEngine:
    """Semantic voting engine that uses a judge LLM to group responses."""

    def __init__(self, ollama_client, consensus_threshold: float = 0.6):
        self.ollama_client = ollama_client
        self.consensus_threshold = consensus_threshold

    def _select_judge(self, responses: List[ModelResponse]) -> str:
        """Select highest-weighted model as judge."""
        if not responses:
            return ""
        return max(responses, key=lambda r: r.weight).model_name

    async def judge_group(
        self, responses: List[ModelResponse]
    ) -> tuple[List[Cluster], str]:
        """Have judge model group responses semantically.

        Returns (clusters, analysis).
        Falls back to individual clusters if JSON parsing fails.
        """
        judge_model = self._select_judge(responses)

        # Build the responses text block
        responses_text = "\n".join(
            f'[{r.model_name}]: "{r.response_text}"' for r in responses
        )
        prompt = JUDGE_PROMPT_TEMPLATE.format(responses=responses_text)

        # Build response lookup
        response_map = {r.model_name: r.response_text for r in responses}

        try:
            result = await self.ollama_client.chat(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1},
            )
            content = result["message"]["content"]

            # Try to extract JSON from the response (may be wrapped in markdown)
            parsed = self._extract_json(content)
            if parsed is None or "clusters" not in parsed:
                logger.warning("Judge returned invalid JSON, using fallback clustering")
                return self._fallback_clusters(responses), ""

            clusters = []
            for c in parsed["clusters"]:
                cluster = Cluster(
                    id=c["id"],
                    label=c.get("label", ""),
                    models=c.get("models", []),
                    key_claims=c.get("key_claims", []),
                    responses={
                        m: response_map.get(m, "") for m in c.get("models", [])
                    },
                )
                clusters.append(cluster)

            analysis = parsed.get("analysis", "")
            return clusters, analysis

        except Exception as e:
            logger.error(f"Judge grouping failed: {e}")
            return self._fallback_clusters(responses), ""

    def _extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON from text that may contain markdown fencing."""
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code blocks
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding first { to last }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass

        return None

    def _fallback_clusters(self, responses: List[ModelResponse]) -> List[Cluster]:
        """Each response in its own cluster when judge fails."""
        return [
            Cluster(
                id=i + 1,
                label=f"Response from {r.model_name}",
                models=[r.model_name],
                key_claims=[],
                responses={r.model_name: r.response_text},
            )
            for i, r in enumerate(responses)
        ]

    async def vote(
        self, responses: List[ModelResponse]
    ) -> Optional[SemanticVoteResult]:
        """Run semantic voting.

        Returns SemanticVoteResult if consensus reached, None if debate needed.
        """
        # Empty responses
        if not responses:
            return SemanticVoteResult(
                winner="",
                winning_model="",
                clusters=[],
                consensus=0.0,
                debate_triggered=False,
                debate_rounds=0,
                debate_transcript=[],
                dissenting_summary="",
                conviction_scores={},
            )

        # Single response: auto-consensus
        if len(responses) == 1:
            r = responses[0]
            cluster = Cluster(
                id=1,
                label="Single response",
                models=[r.model_name],
                key_claims=[],
                responses={r.model_name: r.response_text},
            )
            return SemanticVoteResult(
                winner=r.response_text,
                winning_model=r.model_name,
                clusters=[cluster],
                consensus=1.0,
                debate_triggered=False,
                debate_rounds=0,
                debate_transcript=[],
                dissenting_summary="",
                conviction_scores={},
            )

        # Multiple responses: judge grouping
        clusters, analysis = await self.judge_group(responses)

        # Build weights dict
        weights = {r.model_name: r.weight for r in responses}
        total_weight = sum(weights.values())

        # Score clusters and find winner
        scored = [(c, c.score(weights)) for c in clusters]
        scored.sort(key=lambda x: x[1], reverse=True)

        winning_cluster, winning_score = scored[0]
        consensus = winning_score / total_weight if total_weight > 0 else 0.0

        if consensus >= self.consensus_threshold:
            # Pick the highest-weighted model from the winning cluster
            best_model = max(
                winning_cluster.models,
                key=lambda m: weights.get(m, 0.0),
            )
            return SemanticVoteResult(
                winner=winning_cluster.responses.get(best_model, ""),
                winning_model=best_model,
                clusters=clusters,
                consensus=consensus,
                debate_triggered=False,
                debate_rounds=0,
                debate_transcript=[],
                dissenting_summary="",
                conviction_scores={},
            )

        # Below threshold: debate needed
        return None
