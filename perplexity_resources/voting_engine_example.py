"""
Voting Engine - Implements ensemble decision-making algorithms
"""
from typing import List, Dict, Tuple
from collections import Counter
import numpy as np
import logging

logger = logging.getLogger(__name__)


class VotingEngine:
    """Ensemble voting algorithms for LLM swarm decisions"""

    def __init__(self):
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

        # Group responses and sum weights
        response_weights = {}
        vote_details = {}

        for i, response in enumerate(responses):
            # Normalize response for comparison
            normalized = response.strip().lower()

            if normalized not in response_weights:
                response_weights[normalized] = 0
                vote_details[normalized] = {
                    'original': response,
                    'weight': 0,
                    'votes': [],
                    'models': []
                }

            response_weights[normalized] += weights[i]
            vote_details[normalized]['weight'] += weights[i]
            vote_details[normalized]['votes'].append(weights[i])

            if model_names and i < len(model_names):
                vote_details[normalized]['models'].append(model_names[i])

        # Find winner
        winner = max(response_weights.items(), key=lambda x: x[1])
        winning_response = vote_details[winner[0]]['original']
        total_weight = sum(weights)
        confidence = winner[1] / total_weight if total_weight > 0 else 0

        logger.info(f"Weighted vote winner: {winning_response[:50]}... (confidence: {confidence:.2f})")

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
