"""
PrecedentInjector — classifies precedent action and augments system prompts.

Three possible actions:
  FAST_FOLLOW  — similarity > 0.92 AND consensus > 0.85: return cached ruling
  SOFT_CITE    — similarity > 0.75: inject precedent, models choose FOLLOW/DISTINGUISH
  NO_PRECEDENT — below threshold: normal flow
"""

import logging
import re
from enum import Enum
from typing import List

from .precedent_store import Precedent

logger = logging.getLogger(__name__)


class PrecedentAction(Enum):
    NO_PRECEDENT = "no_precedent"
    SOFT_CITE = "soft_cite"
    FAST_FOLLOW = "fast_follow"


class PrecedentInjector:
    """Injects retrieved precedents into agent system prompts."""

    FOLLOW_SIMILARITY = 0.92
    FOLLOW_CONSENSUS = 0.85
    SOFT_SIMILARITY = 0.75

    def classify_action(self, precedents: List[Precedent]) -> PrecedentAction:
        """Determine what action to take based on best matching precedent."""
        if not precedents:
            return PrecedentAction.NO_PRECEDENT

        best = precedents[0]

        if (
            best.similarity >= self.FOLLOW_SIMILARITY
            and best.consensus >= self.FOLLOW_CONSENSUS
        ):
            return PrecedentAction.FAST_FOLLOW

        if best.similarity >= self.SOFT_SIMILARITY:
            return PrecedentAction.SOFT_CITE

        return PrecedentAction.NO_PRECEDENT

    def augment_system_prompt(
        self,
        base_prompt: str,
        precedents: List[Precedent],
        action: PrecedentAction,
    ) -> str:
        """Add precedent context to system prompt for SOFT_CITE action."""
        if action != PrecedentAction.SOFT_CITE or not precedents:
            return base_prompt

        best = precedents[0]

        precedent_block = (
            f"\n\n--- PRIOR CONGRESS RULING ---\n"
            f"A prior Congress ruling exists on a similar topic "
            f"(consensus: {best.consensus:.2f}, similarity: {best.similarity:.2f}).\n\n"
            f"Prior question: {best.query_text}\n"
            f"Prior ruling: {best.ruling_text}\n\n"
            f"You must either:\n"
            f"  - FOLLOW: State 'I FOLLOW the prior ruling' and build on it.\n"
            f"  - DISTINGUISH: State 'I DISTINGUISH from the prior ruling' "
            f"and explain why this case differs.\n"
            f"--- END PRIOR RULING ---\n"
        )

        return base_prompt + precedent_block

    def detect_distinguish(self, response_text: str) -> bool:
        """Check if a model's response explicitly distinguishes from precedent."""
        text_upper = response_text.upper()
        distinguish_patterns = [
            r"\bI\s+DISTINGUISH\b",
            r"\bDISTINGUISH\s+FROM\s+THE\s+PRIOR\b",
            r"\bOVERRULE\b",
            r"\bDISAGREE\s+WITH\s+THE\s+PRIOR\s+RULING\b",
        ]
        for pattern in distinguish_patterns:
            if re.search(pattern, text_upper):
                return True
        return False

    def build_fast_follow_response(self, precedent: Precedent) -> dict:
        """Build a response dict for FAST_FOLLOW short-circuit."""
        return {
            "final_answer": precedent.ruling_text,
            "confidence": precedent.consensus,
            "precedent": {
                "action": PrecedentAction.FAST_FOLLOW.value,
                "cited": precedent.to_dict(),
                "disposition": "followed",
                "superseded": False,
            },
        }
