"""Debate modules for AI Congress structured argumentation and deliberation."""

from .structured_argumentation import StructuredArgumentation
from .devils_advocate import DevilsAdvocate
from .evidence_grounded import EvidenceGroundedDebate
from .dynamic_depth import DynamicDebateDepth
from .cross_examination import CrossExamination

__all__ = [
    "StructuredArgumentation",
    "DevilsAdvocate",
    "EvidenceGroundedDebate",
    "DynamicDebateDepth",
    "CrossExamination",
]
