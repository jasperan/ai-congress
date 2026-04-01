"""Query domain classification for intelligent model routing."""

from __future__ import annotations

import re


class QueryClassifier:
    """Classifies queries into knowledge domains for optimal model selection."""

    MATH_KEYWORDS = [
        "calculate", "solve", "equation", "integral", "derivative",
        "algebra", "geometry", "trigonometry", "matrix", "polynomial",
        "probability", "statistics", "theorem", "proof", "formula",
        "sum", "product", "factor", "logarithm", "exponent",
    ]
    MATH_PATTERNS = [
        r'\d+\s*[\+\-\*/\^%]\s*\d+',  # arithmetic expressions
        r'[xyz]\s*[=<>]\s*\d+',         # variable equations
        r'\d+\s*[!]',                    # factorials
    ]

    CODING_KEYWORDS = [
        "code", "function", "bug", "implement", "algorithm",
        "debug", "compile", "syntax", "variable", "class",
        "method", "api", "library", "framework", "database",
        "python", "javascript", "java", "rust", "golang", "go",
        "typescript", "c++", "ruby", "swift", "kotlin",
        "html", "css", "sql", "bash", "shell",
        "git", "docker", "kubernetes", "deploy",
        "refactor", "optimize", "test case", "unit test",
    ]

    SCIENCE_KEYWORDS = [
        "experiment", "hypothesis", "research", "molecule",
        "atom", "cell", "gene", "dna", "protein", "evolution",
        "physics", "chemistry", "biology", "quantum", "relativity",
        "electron", "photon", "wavelength", "frequency",
        "ecosystem", "climate", "entropy", "thermodynamics",
        "neuroscience", "astronomy", "geological",
    ]

    HISTORY_KEYWORDS = [
        "when did", "who was", "who were", "historical", "century",
        "dynasty", "empire", "revolution", "war", "battle",
        "civilization", "ancient", "medieval", "renaissance",
        "colonial", "independence", "president", "king", "queen",
        "treaty", "constitution",
    ]
    HISTORY_PATTERNS = [
        r'\b1[0-9]{3}\b',  # years 1000-1999
        r'\b20[0-2][0-9]\b',  # years 2000-2029
        r'\b\d+(st|nd|rd|th)\s+century\b',
    ]

    CREATIVE_KEYWORDS = [
        "write", "poem", "story", "imagine", "design",
        "creative", "fiction", "narrative", "compose", "lyrics",
        "screenplay", "dialogue", "character", "plot", "novel",
        "haiku", "sonnet", "limerick", "essay", "blog post",
        "brainstorm", "invent",
    ]

    # Default domain-model performance scores
    DEFAULT_DOMAIN_SCORES: dict[str, dict[str, float]] = {
        "math": {},
        "coding": {},
        "science": {},
        "history": {},
        "creative": {},
        "general": {},
    }

    def classify_domain(self, query: str) -> str:
        """Classify query into a knowledge domain.

        Returns one of: "math", "coding", "science", "history",
        "creative", "general".
        """
        query_lower = query.lower().strip()

        scores = {
            "math": self._score_domain(query_lower, self.MATH_KEYWORDS, self.MATH_PATTERNS),
            "coding": self._score_domain(query_lower, self.CODING_KEYWORDS),
            "science": self._score_domain(query_lower, self.SCIENCE_KEYWORDS),
            "history": self._score_domain(query_lower, self.HISTORY_KEYWORDS, self.HISTORY_PATTERNS),
            "creative": self._score_domain(query_lower, self.CREATIVE_KEYWORDS),
        }

        best_domain = max(scores, key=scores.get)
        if scores[best_domain] > 0:
            return best_domain
        return "general"

    def get_domain_models(
        self,
        domain: str,
        model_scores: dict[str, dict[str, float]],
    ) -> list[str]:
        """Return top models for a given domain based on performance scores.

        Args:
            domain: The classified domain (e.g. "math", "coding").
            model_scores: Dict of {model_name: {domain: score}} with per-domain
                performance scores.

        Returns:
            List of model names sorted by score descending for the domain.
            Falls back to all models if no domain-specific scores exist.
        """
        if not model_scores:
            return []

        domain_model_scores = []
        for model_name, domains in model_scores.items():
            score = domains.get(domain, domains.get("general", 0.5))
            domain_model_scores.append((model_name, score))

        domain_model_scores.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _score in domain_model_scores]

    @staticmethod
    def _score_domain(
        text: str,
        keywords: list[str],
        patterns: list[str] | None = None,
    ) -> int:
        """Score how well text matches a domain via keywords and patterns."""
        score = 0
        for kw in keywords:
            if kw in text:
                score += 1
        if patterns:
            for pattern in patterns:
                if re.search(pattern, text):
                    score += 2  # Pattern matches are stronger signals
        return score
