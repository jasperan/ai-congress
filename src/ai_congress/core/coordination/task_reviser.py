"""Mid-debate sub-query revision based on response quality signals."""

import logging
from dataclasses import dataclass

from .coalition_formation import CoalitionFormation

logger = logging.getLogger(__name__)


@dataclass
class RevisionSignal:
    """Assessment of whether sub-queries need revision."""

    divergence_score: float
    coverage_score: float
    avg_confidence: float
    divergence_threshold: float
    confidence_threshold: float

    @property
    def should_revise(self) -> bool:
        return (
            self.divergence_score > self.divergence_threshold
            or self.coverage_score < (1.0 - self.divergence_threshold)
            or self.avg_confidence < self.confidence_threshold
        )

    @property
    def reason(self) -> str:
        reasons = []
        if self.divergence_score > self.divergence_threshold:
            reasons.append(f"high divergence ({self.divergence_score:.2f})")
        if self.coverage_score < (1.0 - self.divergence_threshold):
            reasons.append(f"low coverage ({self.coverage_score:.2f})")
        if self.avg_confidence < self.confidence_threshold:
            reasons.append(f"low confidence ({self.avg_confidence:.2f})")
        return "; ".join(reasons) if reasons else "no revision needed"


class TaskReviser:
    """Evaluates response quality and revises sub-queries at wave boundaries."""

    def __init__(
        self,
        ollama_client,
        revision_temperature: float = 0.3,
        max_revisions: int = 2,
        divergence_threshold: float = 0.4,
        confidence_threshold: float = 0.5,
    ):
        self.ollama_client = ollama_client
        self.revision_temperature = revision_temperature
        self.max_revisions = max_revisions
        self.divergence_threshold = divergence_threshold
        self.confidence_threshold = confidence_threshold
        self._similarity = CoalitionFormation()

    def _compute_divergence(self, responses: list[dict]) -> float:
        """Compute average pairwise divergence (1 - similarity) across responses."""
        texts = [r.get("response", "") for r in responses]
        if len(texts) < 2:
            return 0.0

        similarities = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                sim = self._similarity.compute_similarity(texts[i], texts[j])
                similarities.append(sim)

        avg_sim = sum(similarities) / len(similarities) if similarities else 1.0
        return 1.0 - avg_sim

    def _compute_coverage(self, sub_queries: list[dict], responses: list[dict]) -> float:
        """Check how well responses address the sub-queries (keyword overlap)."""
        if not sub_queries:
            return 1.0

        response_words = set()
        for r in responses:
            response_words.update(r.get("response", "").lower().split())

        covered = 0
        for sq in sub_queries:
            sq_words = set(sq.get("text", "").lower().split())
            if not sq_words:
                covered += 1
                continue
            overlap = len(sq_words & response_words) / len(sq_words)
            if overlap >= 0.3:
                covered += 1

        return covered / len(sub_queries)

    async def assess(self, run, responses: list[dict]) -> RevisionSignal:
        """Evaluate whether sub-queries need revision based on response signals."""
        divergence = self._compute_divergence(responses)
        coverage = self._compute_coverage(run.sub_queries, responses)

        confidences = []
        for r in responses:
            conf = r.get("confidence", 0.5)
            confidences.append(conf)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5

        return RevisionSignal(
            divergence_score=divergence,
            coverage_score=coverage,
            avg_confidence=avg_confidence,
            divergence_threshold=self.divergence_threshold,
            confidence_threshold=self.confidence_threshold,
        )

    async def revise(self, run, signal: RevisionSignal, planner_model: str) -> list[dict]:
        """Ask the planner to revise sub-queries given evidence from responses."""
        current_revision = max((sq.get("revision", 0) for sq in run.sub_queries), default=0)
        new_revision = current_revision + 1

        current_sq_text = "\n".join(f"- {sq['text']}" for sq in run.sub_queries)
        revision_prompt = (
            f"You previously decomposed a query into sub-questions, but the initial "
            f"responses suggest the decomposition needs revision.\n\n"
            f"Original query: {run.query}\n\n"
            f"Current sub-questions:\n{current_sq_text}\n\n"
            f"Revision reason: {signal.reason}\n\n"
            f"Revise the sub-questions to better decompose this query. "
            f"You may add, remove, merge, or reword sub-questions (max 3).\n"
            f"Output one sub-question per line, prefixed with '- '."
        )

        try:
            result = await self.ollama_client.chat(
                model=planner_model,
                messages=[{"role": "user", "content": revision_prompt}],
                options={"temperature": self.revision_temperature},
            )
            response_text = result.get("message", {}).get("content", "")

            lines = [
                line.strip().lstrip("- ").strip()
                for line in response_text.split("\n")
                if line.strip().startswith("- ") or line.strip().startswith("-")
            ]

            if not lines:
                logger.warning("Planner returned no revised sub-queries, keeping originals")
                return run.sub_queries

            revised = []
            old_texts = [sq["text"] for sq in run.sub_queries]
            for i, line in enumerate(lines[:3]):
                previous = old_texts[i] if i < len(old_texts) else None
                revised.append({
                    "text": line,
                    "source": planner_model,
                    "revision": new_revision,
                    "previous": previous,
                    "reason": signal.reason,
                })

            logger.info(
                "Sub-queries revised (revision %d): %d -> %d questions",
                new_revision, len(run.sub_queries), len(revised),
            )
            return revised

        except Exception as e:
            logger.warning("Sub-query revision failed: %s", e)
            return run.sub_queries

    def revisions_remaining(self, run) -> int:
        """Return how many revisions are still allowed."""
        current = max((sq.get("revision", 0) for sq in run.sub_queries), default=0)
        return max(0, self.max_revisions - current)
