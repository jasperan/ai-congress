"""
Response Anchoring - Hash-anchored response tracking for debate rounds.

Gives each model's response a deterministic hash ID so models can reference
specific responses precisely during debates, eliminating ambiguity from
paraphrasing or positional references.
"""

import hashlib
from dataclasses import dataclass


@dataclass
class AnchoredResponse:
    model_name: str
    anchor: str  # short hash
    text: str


def compute_anchor(model_name: str, text: str) -> str:
    """Compute a short deterministic hash anchor for a response."""
    content = f"{model_name}:{text[:200]}"
    return hashlib.sha256(content.encode()).hexdigest()[:8]


def anchor_responses(
    model_names: list[str], responses: list[str]
) -> list[AnchoredResponse]:
    """Create hash-anchored response references."""
    anchored = []
    for name, text in zip(model_names, responses):
        anchor = compute_anchor(name, text)
        anchored.append(AnchoredResponse(model_name=name, anchor=anchor, text=text))
    return anchored


def format_anchored_debate_prompt(
    question: str,
    anchored: list[AnchoredResponse],
    instruction: str,
) -> str:
    """Build a debate prompt with hash-anchored response references.

    Models can reference e.g. 'mistral#a3f2b1c0' precisely when critiquing.
    """
    entries = []
    for ar in anchored:
        entries.append(f"[{ar.model_name}#{ar.anchor}]: {ar.text}")

    responses_block = "\n---\n".join(entries)
    return (
        f"Original question: {question}\n\n"
        f"Responses (reference by model#hash when critiquing):\n"
        f"{responses_block}\n\n"
        f"{instruction}"
    )
