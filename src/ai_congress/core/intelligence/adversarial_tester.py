"""Adversarial robustness testing via paraphrase consistency checking."""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class AdversarialTester:
    """Tests model robustness by checking consistency across paraphrased queries."""

    PARAPHRASE_PROMPT = (
        "Rephrase the following question {n} different ways while keeping "
        "exactly the same meaning. Number each paraphrase.\n\n"
        "Original question: {query}\n\n"
        "Paraphrases:"
    )

    async def generate_paraphrases(
        self,
        query: str,
        model_name: str,
        ollama_client: Any,
        n: int = 3,
    ) -> list[str]:
        """Generate paraphrased versions of a query.

        Args:
            query: Original question to paraphrase.
            model_name: Ollama model to use.
            ollama_client: Async Ollama client with generate() method.
            n: Number of paraphrases to generate.

        Returns:
            List of paraphrased query strings.
        """
        prompt = self.PARAPHRASE_PROMPT.format(n=n, query=query)

        try:
            result = await ollama_client.generate(
                model=model_name,
                prompt=prompt,
            )
            raw = result.get("response", "") if isinstance(result, dict) else str(result)
        except Exception as e:
            logger.error("Paraphrase generation failed for %s: %s", model_name, e)
            return []

        return self._parse_paraphrases(raw)

    @staticmethod
    def _parse_paraphrases(raw: str) -> list[str]:
        """Parse numbered paraphrases from LLM output."""
        paraphrases = []
        for line in raw.strip().splitlines():
            line = line.strip()
            # Match lines starting with a number followed by . or )
            match = re.match(r'^\d+[.)]\s*(.+)', line)
            if match:
                text = match.group(1).strip()
                # Remove surrounding quotes if present
                if (text.startswith('"') and text.endswith('"')) or \
                   (text.startswith("'") and text.endswith("'")):
                    text = text[1:-1].strip()
                if text:
                    paraphrases.append(text)
        return paraphrases

    @staticmethod
    def compute_consistency(answers: list[str]) -> float:
        """Compute pairwise consistency among answers.

        Compares all pairs of answers using normalized word overlap.
        A pair is considered "matching" if their overlap ratio exceeds 0.5.

        Returns:
            Float between 0.0 and 1.0. Returns 1.0 if fewer than 2 answers.
        """
        if len(answers) < 2:
            return 1.0

        matching_pairs = 0
        total_pairs = 0

        for i in range(len(answers)):
            for j in range(i + 1, len(answers)):
                total_pairs += 1
                words_a = set(answers[i].lower().split())
                words_b = set(answers[j].lower().split())
                if not words_a or not words_b:
                    continue
                overlap = len(words_a & words_b) / len(words_a | words_b)
                if overlap > 0.5:
                    matching_pairs += 1

        if total_pairs == 0:
            return 1.0
        return matching_pairs / total_pairs

    async def test_robustness(
        self,
        query: str,
        model_name: str,
        ollama_client: Any,
    ) -> dict:
        """Full robustness test: paraphrase, query each, measure consistency.

        Args:
            query: Original question to test.
            model_name: Ollama model to test.
            ollama_client: Async Ollama client.

        Returns:
            Dict with keys:
                - original_answer: str
                - paraphrase_answers: list of {paraphrase, answer}
                - consistency_score: float (0.0-1.0)
        """
        # Get original answer
        try:
            orig_result = await ollama_client.generate(
                model=model_name,
                prompt=query,
            )
            original_answer = (
                orig_result.get("response", "")
                if isinstance(orig_result, dict)
                else str(orig_result)
            )
        except Exception as e:
            logger.error("Original query failed for %s: %s", model_name, e)
            return {
                "original_answer": f"[Error: {e}]",
                "paraphrase_answers": [],
                "consistency_score": 0.0,
            }

        # Generate paraphrases
        paraphrases = await self.generate_paraphrases(
            query, model_name, ollama_client,
        )

        # Query model with each paraphrase
        paraphrase_answers = []
        all_answers = [original_answer]

        for paraphrase in paraphrases:
            try:
                result = await ollama_client.generate(
                    model=model_name,
                    prompt=paraphrase,
                )
                answer = (
                    result.get("response", "")
                    if isinstance(result, dict)
                    else str(result)
                )
            except Exception as e:
                logger.warning("Paraphrase query failed: %s", e)
                answer = f"[Error: {e}]"

            paraphrase_answers.append({
                "paraphrase": paraphrase,
                "answer": answer,
            })
            all_answers.append(answer)

        consistency_score = self.compute_consistency(all_answers)

        return {
            "original_answer": original_answer,
            "paraphrase_answers": paraphrase_answers,
            "consistency_score": consistency_score,
        }
