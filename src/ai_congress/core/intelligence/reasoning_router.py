"""Adaptive reasoning mode selection based on query classification."""

import re


class ReasoningRouter:
    """Classifies queries and selects the best reasoning mode."""

    # Keywords that suggest specific query types
    FACTUAL_KEYWORDS = [
        "what is", "who is", "who was", "define", "when was", "where is",
        "name the", "list the", "what are",
    ]
    ANALYTICAL_KEYWORDS = [
        "explain why", "analyze", "compare", "contrast", "evaluate",
        "step by step", "what causes", "how does", "implications",
        "pros and cons", "advantages", "disadvantages",
    ]
    CREATIVE_KEYWORDS = [
        "write", "create", "imagine", "design", "compose", "invent",
        "poem", "story", "song", "brainstorm", "generate ideas",
    ]
    CALCULATION_KEYWORDS = [
        "calculate", "compute", "how many", "how much", "solve",
        "what is the sum", "total", "percentage", "convert",
    ]
    RESEARCH_KEYWORDS = [
        "search", "find", "look up", "research", "latest", "recent",
        "current", "news about", "what happened",
    ]

    # Keywords mapping to reasoning modes
    COT_KEYWORDS = [
        "step by step", "explain why", "analyze", "reason through",
        "think through", "break down", "detailed explanation",
        "walk me through", "how does", "why does",
    ]
    REACT_KEYWORDS = [
        "calculate", "compute", "how many", "search", "find",
        "look up", "what is the current", "latest",
    ]

    def classify_query(self, query: str) -> str:
        """Classify query into a domain type.

        Returns one of: "factual", "analytical", "creative",
        "calculation", "research".
        """
        query_lower = query.lower().strip()

        # Check each category by keyword presence
        scores = {
            "calculation": self._keyword_score(query_lower, self.CALCULATION_KEYWORDS),
            "research": self._keyword_score(query_lower, self.RESEARCH_KEYWORDS),
            "analytical": self._keyword_score(query_lower, self.ANALYTICAL_KEYWORDS),
            "creative": self._keyword_score(query_lower, self.CREATIVE_KEYWORDS),
            "factual": self._keyword_score(query_lower, self.FACTUAL_KEYWORDS),
        }

        # Return highest scoring category, defaulting to factual
        best = max(scores, key=scores.get)
        if scores[best] > 0:
            return best
        return "factual"

    def select_reasoning_mode(self, query: str) -> str:
        """Select the best reasoning mode for a query.

        Returns one of: "direct", "cot", "react".
        """
        query_lower = query.lower().strip()

        # Very short queries -> direct
        word_count = len(query_lower.split())
        if word_count <= 5:
            return "direct"

        # Check for react-triggering keywords
        react_score = self._keyword_score(query_lower, self.REACT_KEYWORDS)
        if react_score > 0:
            return "react"

        # Check for cot-triggering keywords
        cot_score = self._keyword_score(query_lower, self.COT_KEYWORDS)
        if cot_score > 0:
            return "cot"

        # Long complex queries -> cot
        if word_count >= 20:
            return "cot"

        # Check if query contains multiple sentences or questions
        sentences = re.split(r'[.?!]', query.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) >= 2:
            return "cot"

        return "direct"

    @staticmethod
    def _keyword_score(text: str, keywords: list[str]) -> int:
        """Count how many keyword patterns appear in text."""
        score = 0
        for kw in keywords:
            if kw in text:
                score += 1
        return score
