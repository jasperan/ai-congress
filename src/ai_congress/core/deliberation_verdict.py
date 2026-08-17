"""Deliberation verdict formatting (report issue #11).

The council verdict formatter previously lived inside ``VotingEngine``; it
belongs next to the ``DeliberationOrchestrator`` because it is about the
deliberation protocol's output shape, not vote aggregation. This module is
the new home. ``VotingEngine.deliberation_verdict`` remains as a thin
delegate for backwards compatibility.

The verdict LEADS with what the council does not know (unresolved questions),
then steelmanned dissent, then positions, then the weighted winner — borrowed
from 0xNyk/council-of-high-intelligence. Consensus ranking is placed last
because the council's disagreements matter more than where it agrees.
"""

from typing import Dict, List


def extract_unresolved(final_positions: List[Dict]) -> List[str]:
    """Pull candidate unresolved questions from the Round-3 position statements.

    Looks for agents' declared failure assumptions ('would make me wrong'),
    explicit question marks, and uncertainty markers.
    """
    if not final_positions:
        return []
    found: List[str] = []
    seen: set = set()
    for item in final_positions:
        if not item.get("success"):
            continue
        text = item.get("response", "")
        for marker in ("wrong", "assume", "assumption", "unknown", "unclear"):
            for line in text.split("\n"):
                line_stripped = line.strip(" -*•>\t")
                if not line_stripped:
                    continue
                lower = line_stripped.lower()
                if marker in lower and line_stripped not in seen:
                    if 15 <= len(line_stripped) <= 240:
                        seen.add(line_stripped)
                        found.append(line_stripped)
                    break
        # Surface explicit questions too
        for line in text.split("\n"):
            s = line.strip(" -*•>\t")
            if s.endswith("?") and s not in seen and 10 <= len(s) <= 240:
                seen.add(s)
                found.append(s)
    return found[:6]


def extract_next_steps(final_positions: List[Dict]) -> List[str]:
    """Surface the recommended actions each agent placed first in Round 3."""
    if not final_positions:
        return []
    steps: List[str] = []
    seen: set = set()
    recommend_markers = ("recommend", "next step", "action", "should ", "propose")
    for item in final_positions:
        if not item.get("success"):
            continue
        text = item.get("response", "").strip()
        # take the first 1-2 sentences or the first line with a recommend marker
        for line in text.split("\n"):
            s = line.strip(" -*•>\t")
            if not s:
                continue
            lower = s.lower()
            if any(m in lower for m in recommend_markers) and s not in seen:
                if 15 <= len(s) <= 240:
                    seen.add(s)
                    steps.append(s)
                break
        else:
            # fall back to the first sentence
            first = text.split(".")[0].strip()
            if first and first not in seen and 15 <= len(first) <= 240:
                seen.add(first)
                steps.append(first)
    return steps[:6]


def format_deliberation_verdict(
    question: str,
    final_positions: List[Dict],
    restate: Dict = None,
    dissent_report: Dict = None,
    steelman: List[Dict] = None,
    final_answer: str = "",
) -> str:
    """Format a deliberation verdict that LEADS with what the council does
    not know, then steelmanned dissent (if any), then positions, then the
    weighted winner.
    """
    lines: List[str] = []

    # 1. Question-reframing warning (if the restate gate tripped)
    if restate and restate.get("warning"):
        lines.append("## Question-Reframing Warning")
        lines.append(restate["warning"])
        alternative_framings = [
            r for r in restate.get("restates", [])
            if r.get("alt_framing")
        ]
        if alternative_framings:
            lines.append("")
            lines.append("Alternative framings proposed:")
            for r in alternative_framings:
                lines.append(f"- {r['agent']}: {r['alt_framing']}")
        lines.append("")

    # 2. Unresolved Questions (what the council could not resolve)
    unresolved = extract_unresolved(final_positions)
    lines.append("## Unresolved Questions")
    if unresolved:
        for q in unresolved:
            lines.append(f"- {q}")
    else:
        lines.append("- (none surfaced by the council)")
    lines.append("")

    # 3. Recommended Next Steps
    next_steps = extract_next_steps(final_positions)
    lines.append("## Recommended Next Steps")
    if next_steps:
        for s in next_steps:
            lines.append(f"- {s}")
    else:
        lines.append(
            "- run a tighter version of this question after ingesting the "
            "reframings above, or collect one concrete data point each "
            "council member disagreed on"
        )
    lines.append("")

    # 4. Steelmanned Dissent (if the dissent quota fired)
    if steelman:
        lines.append("## Steelmanned Dissent")
        lines.append(
            "Premature consensus was detected after Round 1. The following "
            "members were forced to steelman the strongest opposing view:"
        )
        lines.append("")
        for item in steelman:
            name = item.get("agent") or item.get("role") or "member"
            response = item.get("response", "").strip()
            lines.append(f"### {name}")
            lines.append(response or "(no response)")
            lines.append("")

    # 5. Final Positions from Round 3
    lines.append("## Final Positions")
    for item in final_positions:
        if not item.get("success"):
            continue
        name = item.get("agent") or item.get("role") or item.get("model") or "member"
        lines.append(f"### {name}")
        lines.append((item.get("response") or "").strip())
        lines.append("")

    # 6. Weighted winner (epistemic humility by placing last)
    lines.append("## Weighted Majority")
    lines.append(final_answer or "(no winner — all positions held)")
    if dissent_report:
        lines.append("")
        lines.append(
            f"_Round-1 agreement: {dissent_report.get('agreement_ratio', 0):.2f}, "
            f"method: {dissent_report.get('method', 'n/a')}. "
            "Consensus ranking placed last because the council's disagreements "
            "matter more than where it agrees._"
        )

    return "\n".join(lines).strip()
