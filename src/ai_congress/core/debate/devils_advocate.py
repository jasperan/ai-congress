"""Devil's Advocate protocol for challenging majority positions.

Instructs a critic model to find the strongest argument against the
majority answer, then evaluates the strength of the challenge.
"""

import logging
import re

logger = logging.getLogger(__name__)


class DevilsAdvocate:
    """Devil's Advocate: challenge majority positions to stress-test consensus."""

    def build_devils_advocate_prompt(
        self, majority_answer: str, question: str
    ) -> str:
        """Build a prompt that instructs a model to argue against the majority.

        Args:
            majority_answer: The current majority/consensus answer.
            question: The original question being debated.

        Returns:
            Devil's advocate prompt string.
        """
        return (
            f"The majority believes: {majority_answer}\n\n"
            f"Original question: {question}\n\n"
            "Your job is to find the strongest argument AGAINST this position. "
            "Play devil's advocate. Find flaws, counterexamples, and alternative "
            "interpretations. If you cannot find a strong counter-argument, "
            "explain why the majority position is robust."
        )

    async def run_devils_advocate(
        self,
        question: str,
        majority_answer: str,
        critic_model: str,
        ollama_client,
    ) -> dict:
        """Query a critic model with the devil's advocate prompt.

        Args:
            question: The original question.
            majority_answer: The majority's current answer.
            critic_model: Name of the Ollama model to use as critic.
            ollama_client: An Ollama client instance with a .generate() or
                .chat() coroutine.

        Returns:
            Dict with challenge text, model name, and is_compelling flag.
        """
        prompt = self.build_devils_advocate_prompt(majority_answer, question)

        try:
            response = await ollama_client.generate(
                model=critic_model, prompt=prompt
            )
            challenge_text = (
                response.get("response", "") if isinstance(response, dict) else str(response)
            )
        except Exception:
            logger.exception("Devil's advocate query failed for model %s", critic_model)
            challenge_text = ""

        strength = self.evaluate_challenge_strength(challenge_text)

        return {
            "challenge": challenge_text,
            "model": critic_model,
            "is_compelling": strength >= 0.5,
            "strength": strength,
        }

    def evaluate_challenge_strength(self, challenge: str) -> float:
        """Heuristically score the strength of a devil's-advocate challenge.

        Scoring (0-1):
            - Length > 100 characters: +0.3
            - Contains specific examples (numbers, quotes, names): +0.3
            - Offers an alternative answer (e.g. "instead", "alternatively"): +0.4

        Args:
            challenge: The challenge text produced by the critic model.

        Returns:
            Strength score between 0.0 and 1.0.
        """
        if not challenge:
            return 0.0

        score = 0.0

        # Length heuristic
        if len(challenge) > 100:
            score += 0.3

        # Specificity: contains numbers, quoted text, or proper-noun-like words
        if re.search(r'\d+|"[^"]+"|\b[A-Z][a-z]{2,}\b', challenge):
            score += 0.3

        # Alternative answer heuristic
        alternative_markers = re.compile(
            r"\b(instead|alternatively|however|on the contrary|"
            r"rather|a better answer|counter-?example)\b",
            re.IGNORECASE,
        )
        if alternative_markers.search(challenge):
            score += 0.4

        return min(score, 1.0)
