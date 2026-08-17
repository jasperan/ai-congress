"""
Bargaining-based consensus for high-stakes questions (3.2.7).

ECON-style negotiation: each model proposes a position with a demand, a
mediator proposes compromises/allocations, and members iterate for up to
``max_rounds`` guided by a utility function derived from priority weights.
Backed by research #7 (LLMs deviate from rational equilibrium as games get
complex; negotiation must be mediator-guided): the mediator is a fixed,
deterministic heuristic over stated utilities — not another model left to
freewheel — so drift is bounded.

Opt-in per request via ``strategy: "bargaining"``.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ...utils.semantic import text_similarity

logger = logging.getLogger(__name__)


@dataclass
class Proposal:
    """A member's stated position in a negotiation round."""
    agent: str
    model: str
    position: str
    utility: float = 0.5      # member's claimed satisfaction with the position
    demand_level: float = 1.0  # how strongly they insist on it (0..1)


@dataclass
class RoundRecord:
    round_no: int
    proposals: List[Proposal] = field(default_factory=list)
    compromise: Dict[str, Any] = field(default_factory=dict)
    convergence: float = 0.0


def _utility_from_weights(priority_weights: Dict[str, float]) -> Callable[[str], float]:
    """Build a utility scorer from mission-style priority weights.

    The scorer is a keyword-overlap heuristic against each priority label:
    a proposal mentioning high-priority concerns (e.g. 'cost', 'safety')
    scores higher. Returns values in [0, 1].
    """
    if not priority_weights:
        return lambda text: 0.5
    items = sorted(priority_weights.items(), key=lambda kv: kv[1], reverse=True)

    def score(text: str) -> float:
        lower = (text or "").lower()
        total, weight_sum = 0.0, 0.0
        for label, w in items:
            weight_sum += w
            if label.lower() in lower or any(
                tok in lower for tok in label.lower().split()
            ):
                total += w
        return total / weight_sum if weight_sum else 0.5

    return score


class BargainingSession:
    """Mediator-guided bargaining loop (≤3 rounds by default)."""

    def __init__(
        self,
        client,
        priority_weights: Optional[Dict[str, float]] = None,
        max_rounds: int = 3,
        tolerance: float = 0.75,      # convergence when best proposal utility ≥ this
        demand_decay: float = 0.4,    # how fast insistence softens each round
        timeout_s: float = 45.0,
    ):
        self.client = client
        self.priority_weights = priority_weights or {}
        self._utility = _utility_from_weights(self.priority_weights)
        self.max_rounds = max_rounds
        self.tolerance = tolerance
        self.demand_decay = demand_decay
        self.timeout_s = timeout_s

    def _member_prompt(
        self, question: str, round_no: int, context: str, demands: Dict[str, float]
    ) -> str:
        lines = [
            f"You are negotiating a consensus on the following high-stakes question:",
            f"QUESTION: {question}",
            "",
        ]
        if context:
            lines.append(f"PROPOSALS SO FAR:\n{context}\n")
        lines.append("Respond with:")
        lines.append("POSITION: <your proposed position, 1-3 sentences>")
        lines.append("INSISTENCE: <0.0 to 1.0 — how strongly you insist on this, given the compromises on the table>")
        if round_no == 1:
            lines.append("You are proposing from first principles; be substantive.")
        elif round_no >= self.max_rounds:
            lines.append("This is the FINAL round. Concede where you can while preserving your core demand.")
        return "\n".join(lines)

    def _parse_proposal(self, raw: str, agent: str, model: str, demand: float) -> Proposal:
        position, insistence = raw, demand
        if "POSITION:" in raw:
            position = raw.split("POSITION:", 1)[1].split("INSISTENCE:", 1)[0].strip()
        if "INSISTENCE:" in raw:
            try:
                inval = raw.split("INSISTENCE:", 1)[1].strip().split()[0]
                insistence = max(0.0, min(1.0, float(inval)))
            except (ValueError, IndexError):
                insistence = demand
        return Proposal(agent=agent, model=model, position=position[:500], utility=self._utility(position), demand_level=insistence)

    async def _ask_member(
        self, spec: Dict[str, Any], question: str, round_no: int, context: str, demands: Dict[str, float]
    ) -> Proposal:
        model = spec.get("model") or spec.get("name", "model")
        agent = spec.get("name") or spec.get("role") or model
        prompt = self._member_prompt(question, round_no, context, demands)
        try:
            result = await self.client.generate(prompt=prompt, model=model, stream=False, temperature=0.6)
            raw = (result.get("response") or result.get("content") or "") if isinstance(result, dict) else str(result)
        except asyncio.TimeoutError:
            raw = "POSITION: (abstain) I defer to the council.\nINSISTENCE: 0.0"
        except Exception as exc:
            logger.warning("Bargaining member %s failed: %s", model, exc)
            raw = "POSITION: (abstain)\nINSISTENCE: 0.0"
        return self._parse_proposal(raw, agent, model, demands.get(model, 0.5))

    def _mediate(self, proposals: List[Proposal], round_no: int) -> Dict[str, Any]:
        """Deterministic mediator: blend proposals by utility, weigh demands.

        The compromise text is the utility-weighted union of member positions
        (top claims per member, weighted). Convergence = best-utility share.
        """
        if not proposals:
            return {"text": "", "utility": 0.0, "concessions": []}

        total_demand = sum(p.demand_level for p in proposals) or 1.0
        parts: List[str] = []
        for p in sorted(proposals, key=lambda x: x.utility, reverse=True):
            weight = p.demand_level / total_demand
            brief = p.position.strip()
            if brief and not brief.startswith("(abstain)"):
                parts.append(f"{p.agent} ({weight:.0%} weight): {brief}")
        text = "\n".join(parts[:3])  # top 3 weighted positions bound the prompt size

        best = max(p.utility for p in proposals)
        compromise_utility = best * 0.9  # mediator pulls toward the strongest position
        return {
            "text": text or "(no substantive proposals)",
            "utility": round(compromise_utility, 4),
            "convergence": round(compromise_utility, 4),
            "concessions": [
                {"agent": p.agent, "insisted": p.demand_level, "utility": p.utility}
                for p in proposals
            ],
        }

    def _build_context(self, rounds: List[RoundRecord]) -> str:
        ctx: List[str] = []
        for r in rounds:
            for p in r.proposals:
                ctx.append(f"[R{r.round_no}] {p.agent}: {p.position[:200]} (insistence {p.demand_level:.2f})")
        return "\n".join(ctx)

    async def negotiate(
        self, agents: List[Dict[str, Any]], question: str, round_no=None
    ) -> Dict[str, Any]:
        """Run the full bargaining loop and return the consensus record."""
        start = time.monotonic()
        rounds: List[RoundRecord] = []
        demands: Dict[str, float] = {
            (a.get("model") or a.get("name", "")): 0.8 for a in agents
        }
        final_compromise: Optional[Dict[str, Any]] = None

        for rn in range(1, self.max_rounds + 1):
            context = self._build_context(rounds)
            proposals = await asyncio.gather(
                *[self._ask_member(a, question, rn, context, demands) for a in agents]
            )
            compromise = self._mediate(proposals, rn)
            record = RoundRecord(round_no=rn, proposals=proposals, compromise=compromise)
            rounds.append(record)
            logger.info(
                "Bargaining round %d: %d proposals, convergence %.2f",
                rn, len(proposals), compromise["convergence"],
            )
            final_compromise = compromise

            # Soften insistence for the next round (guided convergence)
            demands = {k: max(0.1, v * self.demand_decay) for k, v in demands.items()}

            if compromise["convergence"] >= self.tolerance:
                break

        return {
            "strategy": "bargaining",
            "consensus": final_compromise,
            "rounds": [
                {
                    "round_no": r.round_no,
                    "proposals": [
                        {"agent": p.agent, "position": p.position[:300],
                         "utility": p.utility, "insistence": p.demand_level}
                        for p in r.proposals
                    ],
                    "compromise": r.compromise,
                }
                for r in rounds
            ],
            "duration_s": round(time.monotonic() - start, 2),
            "settled": bool(final_compromise and final_compromise["convergence"] >= self.tolerance),
        }