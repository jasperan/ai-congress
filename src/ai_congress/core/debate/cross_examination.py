"""Cross-examination protocol for multi-model debate.

Pairs models that disagree most strongly and generates targeted
examination questions to probe weaknesses in opposing positions.
"""

import logging
import re

logger = logging.getLogger(__name__)


class CrossExamination:
    """Cross-examination: pair disagreeing models for targeted questioning."""

    def build_cross_exam_prompt(
        self,
        target_response: str,
        target_model: str,
        target_anchor: str,
        examiner_response: str,
    ) -> str:
        """Build a cross-examination prompt for one examiner-target pair.

        Args:
            target_response: The response being challenged.
            target_model: Name of the model being examined.
            target_anchor: Anchor/identifier for the target (e.g. round number).
            examiner_response: The examiner's own response (basis for disagreement).

        Returns:
            Prompt instructing the examiner to ask one challenging question.
        """
        return (
            f"Regarding [{target_model}#{target_anchor}]: "
            f"They claim '{target_response[:200]}...'. "
            f"You disagree because: '{examiner_response[:200]}...'. "
            "Ask them ONE specific question that challenges their position."
        )

    def pair_examiners(
        self,
        responses: list[dict],
        anchored: list | None = None,
    ) -> list[dict]:
        """Pair each model with the model it disagrees with most.

        Disagreement is estimated by normalised character-level difference
        in response length and content word overlap (simple heuristic).

        Args:
            responses: List of dicts with at least 'text' and 'model' keys.
            anchored: Optional list of anchor identifiers (same length as
                responses). Defaults to sequential indices.

        Returns:
            List of dicts with examiner, target, and question_prompt.
        """
        if anchored is None:
            anchored = [str(i) for i in range(len(responses))]

        n = len(responses)
        if n < 2:
            return []

        # Pre-compute word sets for each response
        word_sets = [
            set(re.findall(r"\w{3,}", r.get("text", "").lower()))
            for r in responses
        ]

        pairs: list[dict] = []

        for i in range(n):
            best_target = -1
            best_distance = -1.0

            for j in range(n):
                if i == j:
                    continue

                # Compute disagreement: inverse of Jaccard similarity
                union = word_sets[i] | word_sets[j]
                if not union:
                    distance = 0.0
                else:
                    intersection = word_sets[i] & word_sets[j]
                    distance = 1.0 - (len(intersection) / len(union))

                if distance > best_distance:
                    best_distance = distance
                    best_target = j

            if best_target >= 0:
                prompt = self.build_cross_exam_prompt(
                    target_response=responses[best_target].get("text", ""),
                    target_model=responses[best_target].get("model", "unknown"),
                    target_anchor=anchored[best_target],
                    examiner_response=responses[i].get("text", ""),
                )
                pairs.append({
                    "examiner": responses[i].get("model", f"model_{i}"),
                    "target": responses[best_target].get("model", f"model_{best_target}"),
                    "examiner_index": i,
                    "target_index": best_target,
                    "disagreement": best_distance,
                    "question_prompt": prompt,
                })

        return pairs

    def format_cross_exam_round(self, pairs: list[dict]) -> str:
        """Format all cross-examination pairs for display.

        Args:
            pairs: List of pair dicts from pair_examiners().

        Returns:
            Human-readable summary of all cross-examination questions.
        """
        if not pairs:
            return "No cross-examination pairs generated."

        lines = ["=== Cross-Examination Round ===\n"]
        for i, pair in enumerate(pairs, 1):
            lines.append(f"--- Examination {i} ---")
            lines.append(f"Examiner: {pair['examiner']}")
            lines.append(f"Target:   {pair['target']}")
            lines.append(f"Disagreement score: {pair.get('disagreement', 0):.2f}")
            lines.append(f"Prompt:\n  {pair['question_prompt']}")
            lines.append("")

        return "\n".join(lines)
