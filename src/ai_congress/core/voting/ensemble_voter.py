"""Ensemble voting - runs multiple voting algorithms and meta-votes on results."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class EnsembleVoter:
    """Runs multiple voting algorithms and combines their results.

    Takes a VotingEngine instance and runs weighted_majority, temperature_ensemble,
    and confidence_based algorithms, then meta-votes across their outputs to select
    a final winner with an agreement ratio.
    """

    def __init__(self, voting_engine):
        self.voting_engine = voting_engine
        self.algorithm_weights: dict[str, float] = {
            "weighted_majority": 0.4,
            "temperature_ensemble": 0.3,
            "confidence_based": 0.3,
        }
        self.algorithm_history: list[dict[str, Any]] = []

    def ensemble_vote(
        self,
        responses: list[str],
        weights: list[float],
        model_names: list[str],
        temperatures: list[float] | None = None,
    ) -> dict:
        """Run multiple voting algorithms and combine results.

        Args:
            responses: List of model response texts.
            weights: Performance weights for each model (0.0-1.0).
            model_names: Names of the models that produced each response.
            temperatures: Optional temperatures used for each response.

        Returns:
            Dict with winner, confidence, agreement_ratio, algorithm_results,
            and candidates scores.
        """
        candidates: dict[str, float] = {}

        # Algorithm 1: Weighted majority
        winner1, conf1, details1 = self.voting_engine.weighted_majority_vote(
            responses, weights, model_names
        )
        candidates.setdefault(winner1, 0.0)
        candidates[winner1] += conf1 * self.algorithm_weights["weighted_majority"]

        # Algorithm 2: Temperature ensemble (if temperatures provided)
        winner2 = None
        conf2 = 0.0
        if temperatures:
            # temperature_ensemble returns (winner, confidence, details) or just str
            result = self.voting_engine.temperature_ensemble(responses, temperatures)
            if isinstance(result, tuple):
                winner2, conf2 = result[0], result[1]
            else:
                winner2 = result
                # Estimate confidence from temperature weights
                temp_weights = [1.0 / (t + 0.1) for t in temperatures]
                _, conf2, _ = self.voting_engine.weighted_majority_vote(
                    responses, temp_weights, model_names
                )
            candidates.setdefault(winner2, 0.0)
            candidates[winner2] += conf2 * self.algorithm_weights["temperature_ensemble"]

        # Algorithm 3: Confidence-based (use weights as confidence proxy)
        responses_with_conf = [
            {"text": r, "confidence": w, "model": m}
            for r, w, m in zip(responses, weights, model_names)
        ]
        winner3, conf3, details3 = self.voting_engine.confidence_based_vote(
            responses_with_conf
        )
        candidates.setdefault(winner3, 0.0)
        candidates[winner3] += conf3 * self.algorithm_weights["confidence_based"]

        # Meta-vote: highest combined score wins
        final_winner = max(candidates, key=candidates.get)
        final_confidence = candidates[final_winner]

        # Check agreement across algorithms
        algo_winners = [
            winner1,
            winner2 if temperatures else winner1,
            winner3,
        ]
        algorithms_agreed = sum(1 for w in algo_winners if w == final_winner)
        agreement_ratio = algorithms_agreed / 3

        logger.info(
            "Ensemble vote complete: agreement=%.2f, confidence=%.3f",
            agreement_ratio,
            final_confidence,
        )

        return {
            "winner": final_winner,
            "confidence": final_confidence,
            "agreement_ratio": agreement_ratio,
            "algorithm_results": {
                "weighted_majority": {"winner": winner1, "confidence": conf1},
                "temperature_ensemble": {
                    "winner": winner2 if temperatures else None,
                    "confidence": conf2 if temperatures else 0,
                },
                "confidence_based": {"winner": winner3, "confidence": conf3},
            },
            "candidates": candidates,
        }

    def update_algorithm_weights(self, algorithm: str, was_correct: bool) -> None:
        """Update algorithm weights based on accuracy feedback.

        Args:
            algorithm: Name of the algorithm to update.
            was_correct: Whether the algorithm's prediction was correct.
        """
        adjustment = 0.02 if was_correct else -0.02
        self.algorithm_weights[algorithm] = max(
            0.1, min(0.8, self.algorithm_weights.get(algorithm, 0.33) + adjustment)
        )
        # Renormalize so weights sum to 1.0
        total = sum(self.algorithm_weights.values())
        self.algorithm_weights = {
            k: v / total for k, v in self.algorithm_weights.items()
        }

        self.algorithm_history.append(
            {"algorithm": algorithm, "was_correct": was_correct}
        )
        logger.debug(
            "Updated algorithm weights: %s", self.algorithm_weights
        )
