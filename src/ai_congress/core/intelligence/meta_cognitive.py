"""Meta-cognitive monitoring for uncertainty estimation across models."""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)


class MetaCognitiveMonitor:
    """Assesses and aggregates model uncertainty across multiple responses."""

    UNCERTAINTY_PROMPT = (
        "You just answered the following question:\n"
        "Question: {query}\n\n"
        "Your answer was:\n{response}\n\n"
        "Now rate your confidence in that answer on a scale of 1-10, "
        "and list what aspects you are uncertain about.\n\n"
        "Format your response exactly like this:\n"
        "Confidence: N\n"
        "Uncertain about:\n"
        "- first uncertainty\n"
        "- second uncertainty\n"
        "..."
    )

    async def assess_uncertainty(
        self,
        response: str,
        query: str,
        model_name: str,
        ollama_client: Any,
    ) -> dict:
        """Ask a model to self-assess its confidence and uncertainties.

        Args:
            response: The model's original response to evaluate.
            query: The original user query.
            model_name: Ollama model for self-assessment.
            ollama_client: Async Ollama client with generate() method.

        Returns:
            Dict with keys: model, confidence (int 1-10),
            uncertainties (list[str]).
        """
        prompt = self.UNCERTAINTY_PROMPT.format(query=query, response=response)

        try:
            result = await ollama_client.generate(
                model=model_name,
                prompt=prompt,
            )
            raw = result.get("response", "") if isinstance(result, dict) else str(result)
        except Exception as e:
            logger.error("Uncertainty assessment failed for %s: %s", model_name, e)
            return {
                "model": model_name,
                "confidence": 5,
                "uncertainties": [f"Assessment failed: {e}"],
            }

        return self._parse_assessment(raw, model_name)

    @staticmethod
    def _parse_assessment(raw: str, model_name: str) -> dict:
        """Parse confidence score and uncertainty list from LLM output."""
        confidence = 5  # Default
        uncertainties = []

        # Extract confidence score
        conf_match = re.search(r'[Cc]onfidence:\s*(\d+)', raw)
        if conf_match:
            confidence = min(max(int(conf_match.group(1)), 1), 10)

        # Extract uncertainties (lines starting with "- " after "Uncertain about")
        in_uncertainties = False
        for line in raw.splitlines():
            line = line.strip()
            if "uncertain about" in line.lower():
                in_uncertainties = True
                # Check if there's content on the same line after the header
                after_colon = line.split(":", 1)
                if len(after_colon) > 1 and after_colon[1].strip():
                    text = after_colon[1].strip().lstrip("- ").strip()
                    if text:
                        uncertainties.append(text)
                continue
            if in_uncertainties and (line.startswith("- ") or line.startswith("* ")):
                text = line[2:].strip()
                if text:
                    uncertainties.append(text)
            elif in_uncertainties and not line:
                # Empty line may end the list
                continue

        return {
            "model": model_name,
            "confidence": confidence,
            "uncertainties": uncertainties,
        }

    def aggregate_uncertainties(self, assessments: list[dict]) -> dict:
        """Aggregate uncertainty assessments from multiple models.

        Finds common uncertainties (mentioned by 2+ models) and computes
        average confidence.

        Args:
            assessments: List of assessment dicts from assess_uncertainty().

        Returns:
            Dict with keys:
                - avg_confidence: float
                - common_uncertainties: list[str] (mentioned by 2+ models)
                - per_model: list of per-model assessments
        """
        if not assessments:
            return {
                "avg_confidence": 0.0,
                "common_uncertainties": [],
                "per_model": [],
            }

        # Average confidence
        confidences = [a.get("confidence", 5) for a in assessments]
        avg_confidence = sum(confidences) / len(confidences)

        # Find common uncertainties by normalizing and counting
        uncertainty_counter: Counter[str] = Counter()
        for assessment in assessments:
            # Deduplicate within a single model's uncertainties
            seen = set()
            for u in assessment.get("uncertainties", []):
                normalized = u.lower().strip()
                if normalized not in seen:
                    seen.add(normalized)
                    uncertainty_counter[normalized] += 1

        # Keep uncertainties mentioned by 2+ models
        common = [
            text for text, count in uncertainty_counter.most_common()
            if count >= 2
        ]

        return {
            "avg_confidence": round(avg_confidence, 2),
            "common_uncertainties": common,
            "per_model": [
                {
                    "model": a.get("model", "unknown"),
                    "confidence": a.get("confidence", 5),
                    "uncertainties": a.get("uncertainties", []),
                }
                for a in assessments
            ],
        }
