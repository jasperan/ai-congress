"""
Cross-Model Fact Verification

Extracts verifiable factual claims from a winning response, challenges them
against dissenting models, and produces a verification report with disputed
claims clearly marked.
"""

import re
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ClaimVerification:
    claim: str
    status: str  # "verified", "disputed", "uncertain"
    checker_verdicts: list[dict]  # [{model, verdict, reason}]
    confidence: float  # 0-1, proportion of checkers that agree


@dataclass
class FactCheckReport:
    claims: list[ClaimVerification]
    disputed_count: int
    verified_count: int
    uncertain_count: int
    corrected_response: Optional[str] = None

    def to_dict(self):
        return {
            "claims": [
                {
                    "claim": c.claim,
                    "status": c.status,
                    "confidence": c.confidence,
                    "verdicts": c.checker_verdicts,
                }
                for c in self.claims
            ],
            "summary": {
                "verified": self.verified_count,
                "disputed": self.disputed_count,
                "uncertain": self.uncertain_count,
            },
            "has_corrections": self.corrected_response is not None,
        }


class ClaimExtractor:
    """Extracts verifiable factual claims from text."""

    EXTRACT_PROMPT = """Extract 3-5 specific factual claims from this response that could be objectively verified.
Only include claims about: names, dates, numbers, cause-effect relationships, definitions, or measurable facts.
Return ONLY a JSON array of strings, nothing else.

Response to analyze:
{response}"""

    async def extract(self, response_text, ollama_client, model) -> list[str]:
        """Extract claims using LLM, with regex fallback."""
        try:
            result = await ollama_client.chat(
                model,
                [
                    {
                        "role": "user",
                        "content": self.EXTRACT_PROMPT.format(response=response_text),
                    }
                ],
            )
            content = result["message"]["content"]
            # Parse JSON array from response
            match = re.search(r"\[.*\]", content, re.DOTALL)
            if match:
                claims = json.loads(match.group())
                return [str(c) for c in claims[:5]]
        except Exception as e:
            logger.warning(f"LLM claim extraction failed: {e}")
        return []


class CrossModelFactChecker:
    """Verifies claims using dissenting models as independent checkers."""

    VERIFY_PROMPT = """Evaluate this factual claim. Is it correct?
Claim: "{claim}"

Answer with exactly one of: VERIFIED, DISPUTED, or UNCERTAIN
Then provide a brief 1-sentence explanation.
Format: STATUS: explanation"""

    def __init__(self, ollama_client, extractor=None):
        self.client = ollama_client
        self.extractor = extractor or ClaimExtractor()

    async def verify_response(
        self,
        winning_response,
        dissenting_models,
        judge_model,
        max_claims=5,
    ) -> FactCheckReport:
        """Full verification pipeline."""
        # 1. Extract claims
        claims = await self.extractor.extract(
            winning_response, self.client, judge_model
        )
        if not claims:
            return FactCheckReport([], 0, 0, 0)

        # 2. Verify each claim against dissenting models
        verifications = []
        for claim in claims[:max_claims]:
            verdicts = []
            for model in dissenting_models:
                verdict = await self._check_claim(claim, model)
                verdicts.append(verdict)

            # Tally
            status_counts = {"verified": 0, "disputed": 0, "uncertain": 0}
            for v in verdicts:
                status_counts[v["verdict"]] = status_counts.get(v["verdict"], 0) + 1

            # Majority rules
            total = len(verdicts)
            if status_counts["disputed"] > total / 2:
                status = "disputed"
            elif status_counts["verified"] > total / 2:
                status = "verified"
            else:
                status = "uncertain"

            confidence = max(status_counts.values()) / total if total > 0 else 0
            verifications.append(
                ClaimVerification(claim, status, verdicts, confidence)
            )

        # 3. Build report
        disputed = sum(1 for v in verifications if v.status == "disputed")
        verified = sum(1 for v in verifications if v.status == "verified")
        uncertain = sum(1 for v in verifications if v.status == "uncertain")

        corrected = None
        if disputed > 0:
            disputed_claims = [
                v.claim for v in verifications if v.status == "disputed"
            ]
            correction_note = (
                "\n\nFact-check warning: The following claims were disputed by other models:\n"
            )
            correction_note += "\n".join(f"- {c}" for c in disputed_claims)
            corrected = winning_response + correction_note

        return FactCheckReport(
            verifications, disputed, verified, uncertain, corrected
        )

    async def _check_claim(self, claim, model) -> dict:
        """Ask a single model to verify a single claim."""
        try:
            result = await self.client.chat(
                model,
                [
                    {
                        "role": "user",
                        "content": self.VERIFY_PROMPT.format(claim=claim),
                    }
                ],
            )
            content = result["message"]["content"].strip()
            # Parse STATUS: explanation
            for status in ["VERIFIED", "DISPUTED", "UNCERTAIN"]:
                if status in content.upper():
                    reason = (
                        content.split(":", 1)[-1].strip()
                        if ":" in content
                        else content
                    )
                    return {
                        "model": model,
                        "verdict": status.lower(),
                        "reason": reason,
                    }
            return {"model": model, "verdict": "uncertain", "reason": content}
        except Exception as e:
            return {"model": model, "verdict": "uncertain", "reason": str(e)}
