"""Self-verification loop for factual claim extraction and validation."""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class SelfVerifier:
    """Extracts factual claims from responses and verifies them."""

    EXTRACT_CLAIMS_PROMPT = (
        "Extract all factual claims from the following text. "
        "List each claim on its own line prefixed with '- '. "
        "Only include verifiable factual statements, not opinions.\n\n"
        "Text:\n{response}"
    )

    async def extract_claims(
        self,
        response: str,
        model_name: str,
        ollama_client: Any,
    ) -> list[str]:
        """Prompt an LLM to list factual claims from a response.

        Args:
            response: The text to extract claims from.
            model_name: Ollama model to use for extraction.
            ollama_client: An async Ollama client with a `generate` or `chat` method.

        Returns:
            List of factual claim strings.
        """
        prompt = self.EXTRACT_CLAIMS_PROMPT.format(response=response)

        try:
            result = await ollama_client.generate(
                model=model_name,
                prompt=prompt,
            )
            raw = result.get("response", "") if isinstance(result, dict) else str(result)
        except Exception as e:
            logger.error("Failed to extract claims via %s: %s", model_name, e)
            return []

        claims = []
        for line in raw.strip().splitlines():
            line = line.strip()
            if line.startswith("- "):
                claims.append(line[2:].strip())
            elif line.startswith("* "):
                claims.append(line[2:].strip())
        return claims

    async def verify_claims(
        self,
        claims: list[str],
        search_engine: Any,
    ) -> list[dict]:
        """Search each claim and determine whether it is supported.

        Args:
            claims: List of factual claims to verify.
            search_engine: An object with an async `search(query)` method
                that returns a list of result dicts with "snippet" or "content" keys.

        Returns:
            List of dicts: [{claim, verified, evidence}]
        """
        results = []
        for claim in claims:
            try:
                search_results = await search_engine.search(claim)
                evidence_snippets = []
                for sr in (search_results or []):
                    snippet = sr.get("snippet") or sr.get("content") or ""
                    if snippet:
                        evidence_snippets.append(snippet)

                # Simple heuristic: claim is "verified" if any search result
                # contains overlapping keywords
                claim_words = set(claim.lower().split())
                verified = False
                best_evidence = ""
                for snippet in evidence_snippets:
                    snippet_words = set(snippet.lower().split())
                    overlap = claim_words & snippet_words
                    if len(overlap) >= min(3, len(claim_words)):
                        verified = True
                        best_evidence = snippet[:300]
                        break

                results.append({
                    "claim": claim,
                    "verified": verified,
                    "evidence": best_evidence,
                })
            except Exception as e:
                logger.warning("Failed to verify claim '%s': %s", claim[:50], e)
                results.append({
                    "claim": claim,
                    "verified": False,
                    "evidence": f"Verification error: {e}",
                })
        return results

    def compute_fact_score(self, verification_results: list[dict]) -> float:
        """Compute the fraction of claims that were verified.

        Returns:
            Float between 0.0 and 1.0. Returns 1.0 if no claims exist.
        """
        if not verification_results:
            return 1.0
        verified_count = sum(1 for v in verification_results if v.get("verified"))
        return verified_count / len(verification_results)

    async def verify_response(
        self,
        response: str,
        model_name: str,
        ollama_client: Any,
        search_engine: Any,
    ) -> dict:
        """Full verification pipeline for a response.

        Args:
            response: The text to verify.
            model_name: Ollama model for claim extraction.
            ollama_client: Async Ollama client.
            search_engine: Search engine with async search() method.

        Returns:
            Dict with keys: fact_score, claims, verifications.
        """
        claims = await self.extract_claims(response, model_name, ollama_client)
        if not claims:
            return {
                "fact_score": 1.0,
                "claims": [],
                "verifications": [],
            }

        verifications = await self.verify_claims(claims, search_engine)
        fact_score = self.compute_fact_score(verifications)

        return {
            "fact_score": fact_score,
            "claims": claims,
            "verifications": verifications,
        }
