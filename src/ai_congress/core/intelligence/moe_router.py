"""Mixture of Experts routing at the agent level for domain-aware model selection."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .query_classifier import QueryClassifier


class MixtureOfExpertsRouter:
    """Routes queries to the best-performing models based on domain expertise.

    Maintains per-model, per-domain performance scores and uses them to
    select the top-k models for each query's classified domain.
    """

    # Exponential moving average decay factor for score updates
    EMA_ALPHA = 0.3

    def __init__(self, domain_scores: dict[str, dict[str, float]] | None = None) -> None:
        """Initialize the MoE router.

        Args:
            domain_scores: Optional initial scores as
                {model_name: {domain: score}}. Scores should be in [0, 1].
        """
        # {model_name: {domain: score}}
        self._scores: dict[str, dict[str, float]] = domain_scores or {}

    def update_score(self, model: str, domain: str, score: float) -> None:
        """Set or update the performance score for a model in a domain.

        Args:
            model: Model name.
            domain: Domain string (e.g. "math", "coding").
            score: Performance score in [0, 1].
        """
        score = max(0.0, min(1.0, score))
        if model not in self._scores:
            self._scores[model] = {}
        self._scores[model][domain] = score

    def route_query(
        self,
        query: str,
        available_models: list[str],
        top_k: int = 3,
        classifier: "QueryClassifier | None" = None,
    ) -> list[str]:
        """Route a query to the best models for its domain.

        Args:
            query: User query to classify and route.
            available_models: List of available model names.
            top_k: Number of models to return.
            classifier: Optional QueryClassifier instance. If not provided,
                a new one is created.

        Returns:
            List of up to top_k model names, sorted by domain score descending.
            Falls back to available_models if no domain scores exist.
        """
        if not available_models:
            return []

        # Classify the query domain
        if classifier is None:
            from .query_classifier import QueryClassifier
            classifier = QueryClassifier()

        domain = classifier.classify_domain(query)

        # Score each available model for this domain
        scored = []
        for model in available_models:
            model_scores = self._scores.get(model, {})
            # Use domain-specific score, fall back to "general", then 0.5
            score = model_scores.get(domain, model_scores.get("general", 0.5))
            scored.append((model, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        return [model for model, _score in scored[:top_k]]

    def record_outcome(self, model: str, domain: str, was_winner: bool) -> None:
        """Update domain scores using exponential moving average.

        Args:
            model: Model name.
            domain: Domain of the query.
            was_winner: Whether this model's response was selected as best.
        """
        if model not in self._scores:
            self._scores[model] = {}

        current = self._scores[model].get(domain, 0.5)
        outcome = 1.0 if was_winner else 0.0
        new_score = (self.EMA_ALPHA * outcome) + ((1 - self.EMA_ALPHA) * current)
        self._scores[model][domain] = round(new_score, 4)

    def get_routing_stats(self) -> dict:
        """Return per-model, per-domain performance statistics.

        Returns:
            Dict with keys:
                - model_count: int
                - domain_count: int
                - models: {model_name: {domain: score}}
        """
        all_domains: set[str] = set()
        for domains in self._scores.values():
            all_domains.update(domains.keys())

        return {
            "model_count": len(self._scores),
            "domain_count": len(all_domains),
            "domains": sorted(all_domains),
            "models": {
                model: dict(sorted(domains.items()))
                for model, domains in sorted(self._scores.items())
            },
        }
