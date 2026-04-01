"""Toulmin-model structured argumentation for debate rounds.

Asks models to structure their responses using the Toulmin argumentation
framework (Claim, Evidence, Warrant, Qualifier, Rebuttal), then parses
and scores the resulting arguments.
"""

import logging
import re

logger = logging.getLogger(__name__)

ARGUMENTATION_PROMPT = """\
Answer the question below using the Toulmin argumentation structure. \
Format your response EXACTLY as follows:

Claim: [Your main assertion — one clear sentence]
Evidence: [Supporting data, facts, or observations]
Warrant: [Why the evidence supports your claim]
Qualifier: [Your confidence level: certain / likely / possible]
Rebuttal: [What would prove your claim wrong, or the strongest counter-argument]

Question: {question}
"""

_FIELD_PATTERN = re.compile(
    r"(?:^|\n)\s*(Claim|Evidence|Warrant|Qualifier|Rebuttal)\s*:\s*(.*?)(?=\n\s*(?:Claim|Evidence|Warrant|Qualifier|Rebuttal)\s*:|$)",
    re.DOTALL | re.IGNORECASE,
)


class StructuredArgumentation:
    """Toulmin-model argumentation: build prompts, parse, and score arguments."""

    ARGUMENTATION_PROMPT = ARGUMENTATION_PROMPT

    def build_argumentation_prompt(self, question: str) -> str:
        """Return the question wrapped with Toulmin argumentation instructions.

        Args:
            question: The question to be answered.

        Returns:
            A prompt string instructing the model to use Toulmin format.
        """
        return ARGUMENTATION_PROMPT.format(question=question)

    def parse_argument(self, response: str) -> dict:
        """Extract Toulmin fields from a model response.

        Args:
            response: Raw model response text.

        Returns:
            Dict with keys claim, evidence, warrant, qualifier, rebuttal.
            Missing fields are set to None.
        """
        fields: dict[str, str | None] = {
            "claim": None,
            "evidence": None,
            "warrant": None,
            "qualifier": None,
            "rebuttal": None,
        }

        for match in _FIELD_PATTERN.finditer(response):
            key = match.group(1).lower()
            value = match.group(2).strip()
            if key in fields and value:
                fields[key] = value

        return fields

    def score_argument_quality(self, argument: dict) -> float:
        """Score the quality of a parsed argument on a 0-1 scale.

        Scoring criteria (0.2 each):
            - Claim is present and non-empty
            - Evidence is present
            - Warrant is present
            - Qualifier is present
            - Rebuttal is present

        Args:
            argument: Dict produced by parse_argument().

        Returns:
            Quality score between 0.0 and 1.0.
        """
        score = 0.0
        if argument.get("claim"):
            score += 0.2
        if argument.get("evidence"):
            score += 0.2
        if argument.get("warrant"):
            score += 0.2
        if argument.get("qualifier"):
            score += 0.2
        if argument.get("rebuttal"):
            score += 0.2
        return score

    def format_structured_debate_prompt(
        self, question: str, arguments: list[dict]
    ) -> str:
        """Format all arguments in structured form for a critique round.

        Args:
            question: The original question.
            arguments: List of parsed argument dicts.

        Returns:
            Prompt showing all arguments for cross-critique.
        """
        parts = [f"Original question: {question}\n"]
        parts.append("The following arguments have been made:\n")

        for i, arg in enumerate(arguments, 1):
            parts.append(f"--- Argument {i} ---")
            parts.append(f"Claim: {arg.get('claim', 'N/A')}")
            parts.append(f"Evidence: {arg.get('evidence', 'N/A')}")
            parts.append(f"Warrant: {arg.get('warrant', 'N/A')}")
            parts.append(f"Qualifier: {arg.get('qualifier', 'N/A')}")
            parts.append(f"Rebuttal: {arg.get('rebuttal', 'N/A')}")
            parts.append("")

        parts.append(
            "Critique each argument above. Identify the strongest and weakest. "
            "Then provide your own revised argument using the same Toulmin structure."
        )
        return "\n".join(parts)
