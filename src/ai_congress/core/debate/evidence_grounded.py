"""Evidence-grounded debate: enrich model responses with web-search evidence.

Extracts claims from model responses, searches for supporting or refuting
evidence, and asks models to revise their positions in light of the evidence.
"""

import logging
import re

logger = logging.getLogger(__name__)


class EvidenceGroundedDebate:
    """Ground debate responses in external evidence via web search."""

    async def search_claims(
        self,
        claims: list[str],
        search_engine,
    ) -> dict[str, list[dict]]:
        """Search for evidence supporting or refuting each claim.

        Args:
            claims: List of claim strings to search for.
            search_engine: A search engine instance with an async
                `search(query)` method returning a list of result dicts.

        Returns:
            Mapping of claim -> list of search result dicts.
        """
        evidence: dict[str, list[dict]] = {}

        for claim in claims:
            try:
                results = await search_engine.search(claim)
                evidence[claim] = results if isinstance(results, list) else []
            except Exception:
                logger.exception("Search failed for claim: %s", claim[:80])
                evidence[claim] = []

        return evidence

    def format_evidence_prompt(
        self,
        question: str,
        responses: list[dict],
        evidence: dict[str, list[dict]],
    ) -> str:
        """Build a prompt that presents search evidence alongside responses.

        Args:
            question: The original question.
            responses: List of dicts with at least 'text' and 'model' keys.
            evidence: Mapping of claim -> search results from search_claims().

        Returns:
            Prompt instructing models to revise their answers given evidence.
        """
        parts = [f"Question: {question}\n"]

        # Summarise existing responses
        parts.append("Previous responses:")
        for resp in responses:
            model = resp.get("model", "unknown")
            text = resp.get("text", "")[:300]
            parts.append(f"  [{model}]: {text}")
        parts.append("")

        # Present evidence
        parts.append("Here are search results relevant to the claims in this debate:")
        for claim, results in evidence.items():
            parts.append(f"\n  Claim: \"{claim}\"")
            if not results:
                parts.append("    No evidence found.")
                continue
            for r in results[:3]:  # limit to top 3 per claim
                title = r.get("title", "")
                snippet = r.get("snippet", r.get("body", ""))[:200]
                parts.append(f"    - {title}: {snippet}")

        parts.append(
            "\nRevise your answer considering this evidence. "
            "If the evidence supports your position, strengthen it. "
            "If it contradicts your position, update accordingly."
        )
        return "\n".join(parts)

    def compute_evidence_alignment(
        self, response: str, evidence: list[dict]
    ) -> float:
        """Compute simple word-overlap alignment between a response and evidence.

        Args:
            response: A model's response text.
            evidence: List of search result dicts with 'snippet' or 'body' keys.

        Returns:
            Alignment score between 0.0 and 1.0.
        """
        if not response or not evidence:
            return 0.0

        response_words = set(re.findall(r"\w{3,}", response.lower()))
        if not response_words:
            return 0.0

        evidence_words: set[str] = set()
        for item in evidence:
            text = item.get("snippet", item.get("body", ""))
            evidence_words.update(re.findall(r"\w{3,}", text.lower()))

        if not evidence_words:
            return 0.0

        overlap = response_words & evidence_words
        # Normalise by the smaller set to avoid penalising verbose responses
        alignment = len(overlap) / min(len(response_words), len(evidence_words))
        return min(alignment, 1.0)

    async def evidence_grounded_round(
        self,
        question: str,
        responses: list[dict],
        search_engine,
        ollama_client,
    ) -> list[dict]:
        """Run a full evidence-grounded debate round.

        1. Extract claims from each response.
        2. Search for evidence on those claims.
        3. Build an evidence prompt.
        4. Query each model with the evidence prompt.

        Args:
            question: The original question.
            responses: List of dicts with 'text', 'model' keys.
            search_engine: Search engine with async search() method.
            ollama_client: Ollama client with async generate() method.

        Returns:
            List of updated response dicts with evidence-informed answers.
        """
        # Extract claims: use the first sentence of each response as a proxy
        claims: list[str] = []
        for resp in responses:
            text = resp.get("text", "")
            # Take the first sentence as the primary claim
            first_sentence = re.split(r"[.!?]", text, maxsplit=1)[0].strip()
            if first_sentence and len(first_sentence) > 10:
                claims.append(first_sentence)

        if not claims:
            logger.warning("No extractable claims from responses")
            return responses

        # Search for evidence
        evidence = await self.search_claims(claims, search_engine)

        # Build evidence prompt
        evidence_prompt = self.format_evidence_prompt(question, responses, evidence)

        # Query each model with evidence
        updated_responses: list[dict] = []
        for resp in responses:
            model = resp.get("model", "")
            try:
                result = await ollama_client.generate(
                    model=model, prompt=evidence_prompt
                )
                new_text = (
                    result.get("response", "") if isinstance(result, dict) else str(result)
                )
            except Exception:
                logger.exception(
                    "Evidence-grounded query failed for model %s", model
                )
                new_text = resp.get("text", "")

            # Flatten all evidence results for alignment scoring
            all_evidence = [
                item for results in evidence.values() for item in results
            ]
            alignment = self.compute_evidence_alignment(new_text, all_evidence)

            updated_responses.append({
                "text": new_text,
                "model": model,
                "evidence_alignment": alignment,
                "round": "evidence_grounded",
            })

        return updated_responses
