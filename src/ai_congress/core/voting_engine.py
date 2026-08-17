"""
Voting Engine - Implements ensemble decision-making algorithms
"""
from typing import List, Dict, Tuple
import time
import logging

logger = logging.getLogger(__name__)


class VotingEngine:
    """Ensemble voting algorithms for LLM swarm decisions"""

    MAX_HISTORY = 1000

    def __init__(self, semantic_grouping: bool = True):
        """
        Args:
            semantic_grouping: When True (default), responses that mean the
                same thing but are worded differently pool their weight
                (embedding similarity with lexical fallback). When False,
                only exact normalized-string matches pool.
        """
        self.semantic_grouping = semantic_grouping
        self.voting_history = []

    def weighted_majority_vote(
        self,
        responses: List[str],
        weights: List[float],
        model_names: List[str] = None
    ) -> Tuple[str, float, Dict]:
        """
        Weighted majority voting - each response weighted by model performance

        Args:
            responses: List of model responses
            weights: List of model weights (same length as responses)
            model_names: Optional list of model names for tracking

        Returns:
            (winning_response, confidence_score, vote_breakdown)
        """
        if len(responses) != len(weights):
            raise ValueError("Responses and weights must have same length")

        # Group responses and sum weights. When semantic grouping is enabled,
        # paraphrases pool their weight instead of splitting the vote.
        response_weights = {}
        vote_details = {}

        def _add(normalized_key: str, original: str, weight: float, model: str) -> None:
            if normalized_key not in response_weights:
                response_weights[normalized_key] = 0
                vote_details[normalized_key] = {
                    'original': original,
                    'original_weight': weight,
                    'weight': 0,
                    'votes': [],
                    'models': [],
                }
            response_weights[normalized_key] += weight
            vote_details[normalized_key]['weight'] += weight
            vote_details[normalized_key]['votes'].append(weight)
            # Representative = highest-weight member (best winner text)
            if weight > vote_details[normalized_key].get('original_weight', 0.0):
                vote_details[normalized_key]['original'] = original
                vote_details[normalized_key]['original_weight'] = weight
            if model:
                vote_details[normalized_key]['models'].append(model)

        if self.semantic_grouping and len(responses) > 1:
            # Semantic keys: first index of each similarity group serves as
            # the canonical key for every member of that group. The threshold
            # is adaptive: embeddings can catch loose paraphrases; the lexical
            # fallback needs a much lower bar to pool anything at all.
            from ..utils.semantic import embedding_available, text_similarity
            similarity_threshold = 0.72 if embedding_available() else 0.35
            keys: list[str] = []
            for i, response in enumerate(responses):
                canonical = (response or "").strip().lower()
                for j in range(i):
                    if text_similarity(responses[i], responses[j]) >= similarity_threshold:
                        canonical = keys[j]
                        break
                keys.append(canonical)
                _add(
                    canonical,
                    responses[i],
                    weights[i],
                    model_names[i] if model_names and i < len(model_names) else "",
                )
        else:
            for i, response in enumerate(responses):
                normalized = (response or "").strip().lower()
                _add(
                    normalized,
                    response,
                    weights[i],
                    model_names[i] if model_names and i < len(model_names) else "",
                )

        # Find winner
        winner = max(response_weights.items(), key=lambda x: x[1])
        winning_response = vote_details[winner[0]]['original']
        total_weight = sum(weights)
        confidence = winner[1] / total_weight if total_weight > 0 else 0

        logger.info(f"Weighted vote winner: {winning_response[:50]}... (confidence: {confidence:.2f})")

        # Record in voting history (bounded)
        self.voting_history.append({
            "timestamp": time.time(),
            "winner": winning_response,
            "confidence": confidence,
            "num_responses": len(responses),
            "algorithm": "weighted_majority",
        })
        if len(self.voting_history) > self.MAX_HISTORY:
            self.voting_history = self.voting_history[-self.MAX_HISTORY:]

        return winning_response, confidence, vote_details

    def majority_vote(
        self,
        responses: List[str],
        model_names: List[str] = None
    ) -> Tuple[str, float, Dict]:
        """Simple majority voting - all models have equal weight"""
        equal_weights = [1.0] * len(responses)
        return self.weighted_majority_vote(responses, equal_weights, model_names)

    def confidence_based_vote(
        self,
        responses: List[Dict],  # [{'text': str, 'confidence': float, 'model': str}]
    ) -> Tuple[str, float, Dict]:
        """Vote based on model confidence scores"""
        texts = [r['text'] for r in responses]
        confidences = [r.get('confidence', 0.5) for r in responses]
        models = [r.get('model', f'model_{i}') for i, r in enumerate(responses)]

        return self.weighted_majority_vote(texts, confidences, models)

    def rank_responses(
        self,
        responses: List[str],
        weights: List[float],
        model_names: List[str] = None
    ) -> List[Dict]:
        """
        Rank all unique responses by their weighted votes

        Returns:
            List of dicts with response, total_weight, models, rank
        """
        _, _, vote_details = self.weighted_majority_vote(responses, weights, model_names)

        ranked = sorted(
            vote_details.values(),
            key=lambda x: x['weight'],
            reverse=True
        )

        for i, item in enumerate(ranked):
            item['rank'] = i + 1

        return ranked

    def calculate_consensus_score(
        self,
        responses: List[str],
        weights: List[float]
    ) -> float:
        """
        Calculate consensus score (0-1) based on agreement among models

        Higher score = more agreement
        """
        _, confidence, _ = self.weighted_majority_vote(responses, weights)
        return confidence

    def temperature_ensemble(
        self,
        responses: List[str],
        temperatures: List[float]
    ) -> str:
        """
        Ensemble responses from same model at different temperatures

        Lower temperatures get higher weight
        """
        # Inverse temperature as weight (lower temp = more confident)
        weights = [1.0 / (t + 0.1) for t in temperatures]

        winner, _, _ = self.weighted_majority_vote(responses, weights)
        return winner

    def deliberation_verdict(
        self,
        question: str,
        final_positions: List[Dict],
        restate: Dict = None,
        dissent_report: Dict = None,
        steelman: List[Dict] = None,
        final_answer: str = "",
    ) -> str:
        """Format a deliberation verdict (delegates to the deliberation layer).

        Moved to :mod:`core.deliberation_verdict` (report issue #11); this
        wrapper is kept for backwards compatibility.
        """
        from .deliberation_verdict import format_deliberation_verdict
        return format_deliberation_verdict(
            question=question,
            final_positions=final_positions,
            restate=restate,
            dissent_report=dissent_report,
            steelman=steelman,
            final_answer=final_answer,
        )

    @staticmethod
    def _extract_unresolved(final_positions: List[Dict]) -> List[str]:
        """Compat wrapper for the moved verdict helpers."""
        from .deliberation_verdict import extract_unresolved
        return extract_unresolved(final_positions)

    @staticmethod
    def _extract_next_steps(final_positions: List[Dict]) -> List[str]:
        """Compat wrapper for the moved verdict helpers."""
        from .deliberation_verdict import extract_next_steps
        return extract_next_steps(final_positions)
