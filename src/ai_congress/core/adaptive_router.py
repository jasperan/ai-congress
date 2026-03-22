"""
Adaptive Query Router - Routes queries to appropriate swarm size based on complexity.

Uses cheap heuristics (no LLM call) to estimate query complexity and determine
whether a single model, lite swarm (top 3), or full swarm is needed.
"""
import re
import logging
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)


class ComplexityEstimator:
    """Scores query complexity 0.0-1.0 using heuristics (no LLM call)."""

    COMPLEX_PATTERNS = [
        re.compile(r'\b(compare|contrast|analyze|evaluate|design|architect)\b', re.IGNORECASE),
        re.compile(r'\b(pros?\s+and\s+cons?|advantages?\s+and\s+disadvantages?)\b', re.IGNORECASE),
        re.compile(r'\b(step.by.step|in\s+detail|comprehensive|thorough)\b', re.IGNORECASE),
        re.compile(r'\b(why|how\s+does|explain\s+the\s+relationship)\b', re.IGNORECASE),
    ]

    SIMPLE_PATTERNS = [
        re.compile(r'^(what|who|when|where)\s+is\b', re.IGNORECASE),
        re.compile(r'^\d+[\s+\-*/]', re.IGNORECASE),
        re.compile(r'^(yes|no|true|false)\b', re.IGNORECASE),
        re.compile(r'^(define|translate|convert)\b', re.IGNORECASE),
    ]

    # Length thresholds for complexity scaling
    SHORT_QUERY = 30
    MEDIUM_QUERY = 100
    LONG_QUERY = 300

    def estimate(self, query: str) -> float:
        """
        Estimate query complexity from 0.0 (trivial) to 1.0 (requires full swarm).

        Scoring approach:
        - Start at 0.3 baseline
        - Adjust based on length, clause count, and pattern matches
        - Clamp result to [0.0, 1.0]
        """
        if not query or not query.strip():
            return 0.0

        query = query.strip()
        score = 0.3

        # Length adjustment
        length = len(query)
        if length < self.SHORT_QUERY:
            score -= 0.15
        elif length > self.LONG_QUERY:
            score += 0.2
        elif length > self.MEDIUM_QUERY:
            score += 0.1

        # Clause count (commas, semicolons, conjunctions as proxies)
        clause_markers = len(re.findall(r'[,;]|\band\b|\bbut\b|\bor\b|\bhowever\b', query, re.IGNORECASE))
        if clause_markers >= 4:
            score += 0.15
        elif clause_markers >= 2:
            score += 0.08

        # Question mark count (multiple questions = more complex)
        question_count = query.count('?')
        if question_count >= 3:
            score += 0.15
        elif question_count >= 2:
            score += 0.08

        # Complex pattern matches
        complex_hits = sum(1 for p in self.COMPLEX_PATTERNS if p.search(query))
        score += complex_hits * 0.12

        # Simple pattern matches (reduce score)
        simple_hits = sum(1 for p in self.SIMPLE_PATTERNS if p.search(query))
        score -= simple_hits * 0.15

        # Word count adjustment
        word_count = len(query.split())
        if word_count <= 4:
            score -= 0.1
        elif word_count >= 30:
            score += 0.1

        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, score))


@dataclass(frozen=True)
class RoutingDecision:
    """Immutable routing decision produced by AdaptiveQueryRouter."""
    mode: str  # "single", "lite", "full"
    model_count: int
    skip_debate: bool
    max_debate_rounds: int

    def __post_init__(self):
        if self.mode not in ("single", "lite", "full"):
            raise ValueError(f"Invalid routing mode: {self.mode}")
        if self.model_count < 1:
            raise ValueError(f"model_count must be >= 1, got {self.model_count}")
        if self.max_debate_rounds < 0:
            raise ValueError(f"max_debate_rounds must be >= 0, got {self.max_debate_rounds}")


class AdaptiveQueryRouter:
    """
    Routes queries to appropriate swarm size based on estimated complexity.

    Thresholds:
    - complexity < low_threshold  -> single model, no debate
    - complexity < high_threshold -> lite swarm (top 3 models), 1 debate round
    - complexity >= high_threshold -> full swarm, up to 3 debate rounds
    """

    def __init__(self, low_threshold: float = 0.3, high_threshold: float = 0.6):
        if not (0.0 <= low_threshold < high_threshold <= 1.0):
            raise ValueError(
                f"Thresholds must satisfy 0.0 <= low ({low_threshold}) < high ({high_threshold}) <= 1.0"
            )
        self.estimator = ComplexityEstimator()
        self.low = low_threshold
        self.high = high_threshold

    def route(self, query: str, available_models: List[str]) -> RoutingDecision:
        """
        Route a query to the appropriate swarm configuration.

        Args:
            query: The user query text.
            available_models: List of model names currently available.

        Returns:
            RoutingDecision with mode, model_count, skip_debate, max_debate_rounds.
        """
        if not available_models:
            raise ValueError("available_models must not be empty")

        complexity = self.estimator.estimate(query)
        logger.debug(f"Query complexity={complexity:.3f} for: {query[:80]}")

        if complexity < self.low:
            return RoutingDecision(
                mode="single",
                model_count=1,
                skip_debate=True,
                max_debate_rounds=0,
            )
        elif complexity < self.high:
            return RoutingDecision(
                mode="lite",
                model_count=min(3, len(available_models)),
                skip_debate=False,
                max_debate_rounds=1,
            )
        else:
            return RoutingDecision(
                mode="full",
                model_count=len(available_models),
                skip_debate=False,
                max_debate_rounds=3,
            )
