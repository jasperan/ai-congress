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
        """Format a deliberation verdict that LEADS with what the council does
        not know, then steelmanned dissent (if any), then positions, then the
        weighted winner. Borrowed from 0xNyk/council-of-high-intelligence.
        """
        lines: List[str] = []

        # 1. Question-reframing warning (if the restate gate tripped)
        if restate and restate.get("warning"):
            lines.append("## Question-Reframing Warning")
            lines.append(restate["warning"])
            alternative_framings = [
                r for r in restate.get("restates", [])
                if r.get("alt_framing")
            ]
            if alternative_framings:
                lines.append("")
                lines.append("Alternative framings proposed:")
                for r in alternative_framings:
                    lines.append(f"- {r['agent']}: {r['alt_framing']}")
            lines.append("")

        # 2. Unresolved Questions (what the council could not resolve)
        unresolved = self._extract_unresolved(final_positions)
        lines.append("## Unresolved Questions")
        if unresolved:
            for q in unresolved:
                lines.append(f"- {q}")
        else:
            lines.append("- (none surfaced by the council)")
        lines.append("")

        # 3. Recommended Next Steps
        next_steps = self._extract_next_steps(final_positions)
        lines.append("## Recommended Next Steps")
        if next_steps:
            for s in next_steps:
                lines.append(f"- {s}")
        else:
            lines.append(
                "- run a tighter version of this question after ingesting the "
                "reframings above, or collect one concrete data point each "
                "council member disagreed on"
            )
        lines.append("")

        # 4. Steelmanned Dissent (if the dissent quota fired)
        if steelman:
            lines.append("## Steelmanned Dissent")
            lines.append(
                "Premature consensus was detected after Round 1. The following "
                "members were forced to steelman the strongest opposing view:"
            )
            lines.append("")
            for item in steelman:
                name = item.get("agent") or item.get("role") or "member"
                response = item.get("response", "").strip()
                lines.append(f"### {name}")
                lines.append(response or "(no response)")
                lines.append("")

        # 5. Final Positions from Round 3
        lines.append("## Final Positions")
        for item in final_positions:
            if not item.get("success"):
                continue
            name = item.get("agent") or item.get("role") or item.get("model") or "member"
            lines.append(f"### {name}")
            lines.append((item.get("response") or "").strip())
            lines.append("")

        # 6. Weighted winner (epistemic humility by placing last)
        lines.append("## Weighted Majority")
        lines.append(final_answer or "(no winner — all positions held)")
        if dissent_report:
            lines.append("")
            lines.append(
                f"_Round-1 agreement: {dissent_report.get('agreement_ratio', 0):.2f}, "
                f"method: {dissent_report.get('method', 'n/a')}. "
                "Consensus ranking placed last because the council's disagreements "
                "matter more than where it agrees._"
            )

        return "\n".join(lines).strip()

    @staticmethod
    def _extract_unresolved(final_positions: List[Dict]) -> List[str]:
        """Pull candidate unresolved questions from the Round-3 position statements.

        Looks for agents' declared failure assumptions ('would make me wrong'),
        explicit question marks, and uncertainty markers.
        """
        if not final_positions:
            return []
        found: List[str] = []
        seen: set = set()
        for item in final_positions:
            if not item.get("success"):
                continue
            text = item.get("response", "")
            for marker in ("wrong", "assume", "assumption", "unknown", "unclear"):
                for line in text.split("\n"):
                    line_stripped = line.strip(" -*•>\t")
                    if not line_stripped:
                        continue
                    lower = line_stripped.lower()
                    if marker in lower and line_stripped not in seen:
                        if 15 <= len(line_stripped) <= 240:
                            seen.add(line_stripped)
                            found.append(line_stripped)
                        break
            # Surface explicit questions too
            for line in text.split("\n"):
                s = line.strip(" -*•>\t")
                if s.endswith("?") and s not in seen and 10 <= len(s) <= 240:
                    seen.add(s)
                    found.append(s)
        return found[:6]

    @staticmethod
    def _extract_next_steps(final_positions: List[Dict]) -> List[str]:
        """Surface the recommended actions each agent placed first in Round 3."""
        if not final_positions:
            return []
        steps: List[str] = []
        seen: set = set()
        recommend_markers = ("recommend", "next step", "action", "should ", "propose")
        for item in final_positions:
            if not item.get("success"):
                continue
            text = item.get("response", "").strip()
            # take the first 1-2 sentences or the first line with a recommend marker
            for line in text.split("\n"):
                s = line.strip(" -*•>\t")
                if not s:
                    continue
                lower = s.lower()
                if any(m in lower for m in recommend_markers) and s not in seen:
                    if 15 <= len(s) <= 240:
                        seen.add(s)
                        steps.append(s)
                    break
            else:
                # fall back to the first sentence
                first = text.split(".")[0].strip()
                if first and first not in seen and 15 <= len(first) <= 240:
                    seen.add(first)
                    steps.append(first)
        return steps[:6]
