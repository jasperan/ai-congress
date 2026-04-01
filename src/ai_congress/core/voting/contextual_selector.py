"""Contextual voting algorithm selector.

Automatically chooses the best voting algorithm based on query
characteristics and response properties.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Keyword sets for query classification
_FACTUAL_KEYWORDS = re.compile(
    r"\b(what is|who is|when did|where is|how many|define|name the|"
    r"capital of|population|year|date|number|fact)\b",
    re.IGNORECASE,
)
_SUBJECTIVE_KEYWORDS = re.compile(
    r"\b(opinion|think|feel|best|worst|favourite|favorite|recommend|"
    r"creative|write a|compose|imagine|story|poem|essay)\b",
    re.IGNORECASE,
)
_TECHNICAL_KEYWORDS = re.compile(
    r"\b(code|function|algorithm|implement|debug|error|exception|"
    r"calculate|equation|formula|proof|theorem|sql|python|java|"
    r"javascript|typescript|bash|regex)\b",
    re.IGNORECASE,
)


class ContextualVotingSelector:
    """Selects the most appropriate voting algorithm based on context.

    Analyses the query text, number of responses, and response lengths
    to pick among weighted_majority, semantic, and confidence_based
    algorithms.
    """

    def select_algorithm(
        self,
        query: str,
        num_responses: int,
        response_lengths: list[int],
    ) -> str:
        """Choose a voting algorithm for the given query and responses.

        Args:
            query: The user's original query text.
            num_responses: How many model responses were collected.
            response_lengths: List of character lengths of each response.

        Returns:
            Algorithm name string: one of "weighted_majority", "semantic",
            or "confidence_based".
        """
        avg_length = (
            sum(response_lengths) / len(response_lengths) if response_lengths else 0
        )

        # Many responses -> semantic (string matching less reliable at scale)
        if num_responses > 5:
            logger.debug("Selected 'semantic': many responses (%d)", num_responses)
            return "semantic"

        # Long responses -> semantic (rich text better compared semantically)
        if avg_length > 500:
            logger.debug("Selected 'semantic': long responses (avg %.0f chars)", avg_length)
            return "semantic"

        # Short responses -> weighted_majority (likely factual / concise)
        if avg_length < 50:
            logger.debug(
                "Selected 'weighted_majority': short responses (avg %.0f chars)",
                avg_length,
            )
            return "weighted_majority"

        # Query-type heuristics
        if _TECHNICAL_KEYWORDS.search(query):
            logger.debug("Selected 'confidence_based': technical query")
            return "confidence_based"

        if _SUBJECTIVE_KEYWORDS.search(query):
            logger.debug("Selected 'semantic': subjective query")
            return "semantic"

        if _FACTUAL_KEYWORDS.search(query):
            logger.debug("Selected 'weighted_majority': factual query")
            return "weighted_majority"

        # Default fallback
        logger.debug("Selected 'weighted_majority': default fallback")
        return "weighted_majority"
