"""Hypothetical reasoning via multi-model hypothesis generation and aggregation."""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class HypothesisExplorer:
    """Generates and aggregates hypotheses across multiple models."""

    HYPOTHESIS_PROMPT = (
        "Generate {n} possible answers to the following question, "
        "ranked by likelihood. For each answer, estimate your confidence "
        "as a percentage.\n\n"
        "Format each answer exactly like this:\n"
        "1. [answer text] (confidence: X%)\n"
        "2. [answer text] (confidence: X%)\n"
        "...\n\n"
        "Question: {query}"
    )

    async def generate_hypotheses(
        self,
        query: str,
        model_name: str,
        ollama_client: Any,
        n: int = 3,
    ) -> list[dict]:
        """Generate ranked hypotheses from a single model.

        Args:
            query: The question to generate hypotheses for.
            model_name: Ollama model to use.
            ollama_client: Async Ollama client with generate() method.
            n: Number of hypotheses to request.

        Returns:
            List of dicts: [{answer, confidence, rank}]
        """
        prompt = self.HYPOTHESIS_PROMPT.format(n=n, query=query)

        try:
            result = await ollama_client.generate(
                model=model_name,
                prompt=prompt,
            )
            raw = result.get("response", "") if isinstance(result, dict) else str(result)
        except Exception as e:
            logger.error("Hypothesis generation failed for %s: %s", model_name, e)
            return []

        return self._parse_hypotheses(raw)

    @staticmethod
    def _parse_hypotheses(raw: str) -> list[dict]:
        """Parse numbered hypotheses with confidence from LLM output.

        Expected format: "1. [answer] (confidence: X%)"
        """
        hypotheses = []
        # Match patterns like "1. answer text (confidence: 85%)"
        pattern = re.compile(
            r'(\d+)\.\s*(.+?)\s*\(confidence:\s*(\d+)%?\)',
            re.IGNORECASE,
        )

        for match in pattern.finditer(raw):
            rank = int(match.group(1))
            answer = match.group(2).strip()
            confidence = int(match.group(3))
            hypotheses.append({
                "answer": answer,
                "confidence": min(confidence, 100),
                "rank": rank,
            })

        # If regex parsing failed, try line-by-line fallback
        if not hypotheses:
            lines = raw.strip().splitlines()
            for i, line in enumerate(lines):
                line = line.strip()
                num_match = re.match(r'^(\d+)[.)]\s*(.+)', line)
                if num_match:
                    answer = num_match.group(2).strip()
                    # Try to extract confidence
                    conf_match = re.search(r'(\d+)%', answer)
                    confidence = int(conf_match.group(1)) if conf_match else 50
                    # Remove confidence marker from answer
                    answer = re.sub(r'\s*\(confidence:?\s*\d+%?\)', '', answer).strip()
                    hypotheses.append({
                        "answer": answer,
                        "confidence": min(confidence, 100),
                        "rank": i + 1,
                    })

        return hypotheses

    def aggregate_hypotheses(
        self,
        all_hypotheses: list[list[dict]],
    ) -> list[dict]:
        """Merge hypotheses across models using rank-based scoring.

        For each unique answer, compute an aggregate score as the sum of
        (1/rank) across all models that include it.

        Args:
            all_hypotheses: List of hypothesis lists, one per model.

        Returns:
            List of dicts sorted by aggregate score descending:
            [{answer, aggregate_score, model_count, avg_confidence}]
        """
        answer_data: dict[str, dict] = {}

        for model_hypotheses in all_hypotheses:
            for hyp in model_hypotheses:
                # Normalize answer for grouping
                key = hyp["answer"].lower().strip()
                rank = max(hyp.get("rank", 1), 1)
                confidence = hyp.get("confidence", 50)

                if key not in answer_data:
                    answer_data[key] = {
                        "answer": hyp["answer"],  # Keep original casing
                        "aggregate_score": 0.0,
                        "model_count": 0,
                        "confidences": [],
                    }

                answer_data[key]["aggregate_score"] += 1.0 / rank
                answer_data[key]["model_count"] += 1
                answer_data[key]["confidences"].append(confidence)

        # Build final results
        results = []
        for data in answer_data.values():
            avg_confidence = (
                sum(data["confidences"]) / len(data["confidences"])
                if data["confidences"]
                else 0.0
            )
            results.append({
                "answer": data["answer"],
                "aggregate_score": round(data["aggregate_score"], 4),
                "model_count": data["model_count"],
                "avg_confidence": round(avg_confidence, 1),
            })

        results.sort(key=lambda x: x["aggregate_score"], reverse=True)
        return results
