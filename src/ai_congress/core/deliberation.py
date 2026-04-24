"""
Deliberation Orchestrator - 3-round structured debate with a Problem Restate
Gate and a Dissent Quota, borrowing from 0xNyk/council-of-high-intelligence.

Round 1: blind independent analysis (400 words max).
Round 2: cross-examination, each agent must engage >= 2 peers (300 words max).
Round 3: final crystallization (100 words max).

An optional Problem Restate Gate runs before Round 1, asking every agent to
restate the question and offer one alternative framing. When >= min_divergent
agents reframe divergently, a 'question-reframing' warning is attached to the
output so the user can notice the question itself may be the problem.

An optional Dissent Quota fires after Round 1. A consensus detector scores
pairwise agreement across outputs; when it exceeds a configurable threshold,
two agents are forced to steelman the strongest opposing view. The steelman
pass is appended to the final output so hidden disagreement surfaces instead
of being hidden behind a confident-looking vote.
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

from .consensus_detector import (
    ConsensusReport,
    detect_consensus,
    pick_steelman_targets,
)

logger = logging.getLogger(__name__)


AgentSpec = Dict[str, str]
QueryFn = Callable[[AgentSpec, List[Dict[str, str]], float], Awaitable[Dict]]


RESTATE_PROMPT = (
    "Before analyzing, do two things in fewer than 120 words each:\n"
    "1. Restate the user's question in your own words (one sentence).\n"
    "2. Offer ONE alternative framing that might be closer to the real problem.\n\n"
    "Use this exact format:\n"
    "RESTATE: <your restatement>\n"
    "ALT_FRAMING: <your alternative framing>\n\n"
    "User's question:\n{question}"
)

ROUND1_PROMPT = (
    "Provide your independent analysis of the user's question. "
    "You cannot see other council members yet. Keep it under 400 words. "
    "Be concrete, name the trade-offs you see, and flag anything you are unsure about.\n\n"
    "User's question:\n{question}"
)

ROUND2_TEMPLATE = (
    "Round 2: cross-examination. Here are the Round 1 responses from the other "
    "council members:\n\n{others}\n\n"
    "Write your Round 2 response in under 300 words. You MUST engage at least two "
    "other members by name. Quote the specific claim you are responding to, then "
    "agree, disagree, or refine it. Do not repeat your Round 1 points verbatim.\n\n"
    "User's question:\n{question}"
)

ROUND3_TEMPLATE = (
    "Round 3: final position. Given the debate so far:\n\n{others}\n\n"
    "State your final position in under 100 words. Lead with a concrete "
    "recommendation. End with the single assumption most likely to make "
    "you wrong.\n\n"
    "User's question:\n{question}"
)

STEELMAN_TEMPLATE = (
    "The council is converging too fast. Your assignment is to steelman the "
    "strongest opposing view to the emerging consensus, even if you disagree. "
    "Here is the current majority position:\n\n{majority}\n\n"
    "Here is the response that diverged most from the majority:\n\n{outlier}\n\n"
    "In under 200 words, make the best possible case against the majority. "
    "Stay technical, specific, and in good faith. Do not hedge.\n\n"
    "User's question:\n{question}"
)

STEELMAN_NO_DISSENT_TEMPLATE = (
    "The council is converging too fast and nobody dissented in Round 1. "
    "Your assignment is to invent the strongest counter-position an expert "
    "outside this room would hold. Here is the current majority position:\n\n"
    "{majority}\n\n"
    "In under 200 words, make the best possible case AGAINST the majority. "
    "Invent the missing dissenter. Stay technical, specific, and in good faith. "
    "Do not hedge.\n\n"
    "User's question:\n{question}"
)


@dataclass
class DeliberationConfig:
    consensus_threshold: float = 0.7
    pairwise_threshold: float = 0.7
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    restate_gate_enabled: bool = True
    restate_divergence_min: int = 3
    dissent_quota_enabled: bool = True
    dissent_steelman_count: int = 2
    round1_word_limit: int = 400
    round2_word_limit: int = 300
    round3_word_limit: int = 100


@dataclass
class RoundResult:
    name: str
    outputs: List[Dict]   # [{agent, role, model, response, success, error?}]

    def texts(self) -> List[str]:
        return [o.get("response", "") for o in self.outputs]


@dataclass
class RestateGateResult:
    restates: List[Dict]            # [{agent, restate, alt_framing, raw}]
    divergent_count: int
    warning: Optional[str]


@dataclass
class DeliberationResult:
    agents: List[AgentSpec]
    restate: Optional[RestateGateResult]
    rounds: List[RoundResult]
    dissent_report: Optional[ConsensusReport]
    steelman: Optional[RoundResult]
    final_positions: List[Dict]
    metadata: Dict = field(default_factory=dict)


def _strip_response(text: str) -> str:
    return (text or "").strip()


def _enforce_word_limit(text: str, limit: int) -> str:
    words = text.split()
    if len(words) <= limit:
        return text
    return " ".join(words[:limit]) + " [...]"


def _parse_restate(raw: str) -> Tuple[str, str]:
    restate = ""
    alt = ""
    m = re.search(r"RESTATE\s*:\s*(.+?)(?:\n\s*ALT_FRAMING|$)", raw, re.IGNORECASE | re.DOTALL)
    if m:
        restate = m.group(1).strip()
    m2 = re.search(r"ALT_FRAMING\s*:\s*(.+)", raw, re.IGNORECASE | re.DOTALL)
    if m2:
        alt = m2.group(1).strip()
    if not restate and not alt:
        restate = raw.strip().split("\n", 1)[0][:280]
    return restate, alt


def _count_divergent_restates(restates: List[Dict], threshold: float = 0.55) -> int:
    """Count restatements that diverge lexically from the modal phrasing.

    Failed restates (success=False) are ignored. For short restatements
    (<12 tokens combined) we tighten the threshold so a single synonym
    swap doesn't falsely trip the divergence warning.
    """
    from .consensus_detector import _jaccard, _tokens
    usable = [r for r in restates if r.get("success", True)]
    texts = [
        (r.get("restate") or "") + " " + (r.get("alt_framing") or "")
        for r in usable
    ]
    if len(texts) < 2:
        return 0
    divergent = 0
    for i, t in enumerate(texts):
        t_tokens = _tokens(t)
        short = len(t_tokens) < 12
        effective_threshold = threshold * 0.6 if short else threshold
        sims = [_jaccard(t, texts[j]) for j in range(len(texts)) if j != i]
        if sims and (sum(sims) / len(sims)) < effective_threshold:
            divergent += 1
    return divergent


class DeliberationOrchestrator:
    """Runs the 3-round deliberation protocol against an injected query fn."""

    def __init__(
        self,
        query_fn: QueryFn,
        config: Optional[DeliberationConfig] = None,
    ):
        self.query_fn = query_fn
        self.config = config or DeliberationConfig()

    async def _ask(
        self,
        agent: AgentSpec,
        user_prompt: str,
        temperature: float,
    ) -> Dict:
        messages = []
        system = agent.get("system_prompt") or agent.get("role_prompt")
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_prompt})
        result = await self.query_fn(agent, messages, temperature)
        text = _strip_response(result.get("response", ""))
        return {
            "agent": agent.get("name") or agent.get("role") or agent.get("model"),
            "role": agent.get("role"),
            "model": agent.get("model"),
            "response": text,
            "success": bool(result.get("success", True)) and bool(text),
            "error": result.get("error"),
        }

    async def _ask_all(
        self,
        agents: List[AgentSpec],
        user_prompt_for: Callable[[AgentSpec], str],
        temperature: float,
    ) -> List[Dict]:
        tasks = [
            self._ask(a, user_prompt_for(a), temperature)
            for a in agents
        ]
        return await asyncio.gather(*tasks)

    async def run_restate_gate(
        self,
        agents: List[AgentSpec],
        question: str,
        temperature: float,
    ) -> RestateGateResult:
        raw_outputs = await self._ask_all(
            agents,
            lambda _a: RESTATE_PROMPT.format(question=question),
            temperature=min(temperature, 0.5),
        )
        restates: List[Dict] = []
        for o in raw_outputs:
            restate, alt = _parse_restate(o.get("response", ""))
            restates.append({
                "agent": o["agent"],
                "restate": restate,
                "alt_framing": alt,
                "raw": o.get("response", ""),
                "success": o["success"],
            })
        divergent = _count_divergent_restates(restates)
        warning = None
        if divergent >= self.config.restate_divergence_min:
            warning = (
                f"question-reframing warning: {divergent} of {len(restates)} "
                "council members restated or reframed your question differently. "
                "The question itself may be ambiguous; consider tightening it before "
                "acting on the verdict."
            )
        return RestateGateResult(
            restates=restates,
            divergent_count=divergent,
            warning=warning,
        )

    async def run_round1(
        self, agents: List[AgentSpec], question: str, temperature: float
    ) -> RoundResult:
        prompt = ROUND1_PROMPT.format(question=question)
        outputs = await self._ask_all(agents, lambda _a: prompt, temperature)
        for o in outputs:
            o["response"] = _enforce_word_limit(o.get("response", ""), self.config.round1_word_limit)
        return RoundResult(name="round1", outputs=outputs)

    async def run_round2(
        self,
        agents: List[AgentSpec],
        question: str,
        round1: RoundResult,
        temperature: float,
    ) -> RoundResult:
        def build(a: AgentSpec) -> str:
            agent_key = a.get("name") or a.get("role") or a.get("model")
            others_entries = [
                f"{o['agent']} ({o.get('role') or ''}): {o['response']}"
                for o in round1.outputs
                if (o.get("agent") != agent_key) and o.get("success")
            ]
            others = "\n\n".join(others_entries) or "(no peers responded)"
            return ROUND2_TEMPLATE.format(others=others, question=question)
        outputs = await self._ask_all(agents, build, temperature)
        for o in outputs:
            o["response"] = _enforce_word_limit(o.get("response", ""), self.config.round2_word_limit)
        return RoundResult(name="round2", outputs=outputs)

    async def run_round3(
        self,
        agents: List[AgentSpec],
        question: str,
        round2: RoundResult,
        temperature: float,
    ) -> RoundResult:
        def build(a: AgentSpec) -> str:
            agent_key = a.get("name") or a.get("role") or a.get("model")
            others_entries = [
                f"{o['agent']} ({o.get('role') or ''}): {o['response']}"
                for o in round2.outputs
                if (o.get("agent") != agent_key) and o.get("success")
            ]
            others = "\n\n".join(others_entries) or "(no peers responded)"
            return ROUND3_TEMPLATE.format(others=others, question=question)
        outputs = await self._ask_all(agents, build, temperature)
        for o in outputs:
            o["response"] = _enforce_word_limit(o.get("response", ""), self.config.round3_word_limit)
        return RoundResult(name="round3", outputs=outputs)

    async def run_dissent_pass(
        self,
        agents: List[AgentSpec],
        question: str,
        round1: RoundResult,
        report: ConsensusReport,
        temperature: float,
    ) -> Optional[RoundResult]:
        texts = round1.texts()
        if not texts:
            return None
        picks = pick_steelman_targets(report, count=self.config.dissent_steelman_count)
        picks = [p for p in picks if 0 <= p < len(agents)]
        if not picks:
            return None
        majority_idx = int(max(range(len(texts)), key=lambda k: len(texts[k])))
        outlier_idx = report.outlier_index if report.outlier_index is not None else majority_idx
        majority_txt = texts[majority_idx]
        outlier_txt = texts[outlier_idx]

        # When all outputs collapse to the same text the "outlier" is just the
        # majority in disguise. Swap to a template that asks the picks to
        # invent an outside dissenter instead of restating the majority.
        use_no_dissent = (
            majority_txt.strip() == outlier_txt.strip()
            or report.agreement_ratio >= 0.98
        )

        async def steelman(i: int) -> Dict:
            if use_no_dissent:
                prompt = STEELMAN_NO_DISSENT_TEMPLATE.format(
                    majority=majority_txt,
                    question=question,
                )
            else:
                prompt = STEELMAN_TEMPLATE.format(
                    majority=majority_txt,
                    outlier=outlier_txt,
                    question=question,
                )
            return await self._ask(agents[i], prompt, temperature=min(1.0, temperature + 0.1))

        outputs = await asyncio.gather(*[steelman(i) for i in picks])
        for o in outputs:
            o["response"] = _enforce_word_limit(o.get("response", ""), 200)
        return RoundResult(name="steelman", outputs=outputs)

    async def run(
        self,
        agents: List[AgentSpec],
        question: str,
        temperature: float = 0.7,
    ) -> DeliberationResult:
        if len(agents) < 2:
            raise ValueError("deliberation needs at least 2 agents")

        restate: Optional[RestateGateResult] = None
        if self.config.restate_gate_enabled:
            logger.info("deliberation: running Problem Restate Gate")
            restate = await self.run_restate_gate(agents, question, temperature)
            if restate.warning:
                logger.info(f"deliberation: {restate.warning}")

        logger.info("deliberation: Round 1 (independent analysis)")
        round1 = await self.run_round1(agents, question, temperature)

        dissent_report: Optional[ConsensusReport] = None
        steelman: Optional[RoundResult] = None
        if self.config.dissent_quota_enabled:
            texts_for_consensus = [o.get("response", "") for o in round1.outputs if o.get("success")]
            if len(texts_for_consensus) >= 2:
                dissent_report = detect_consensus(
                    texts_for_consensus,
                    consensus_threshold=self.config.consensus_threshold,
                    pairwise_threshold=self.config.pairwise_threshold,
                    embedding_model=self.config.embedding_model,
                )
                if dissent_report.premature:
                    logger.info(
                        "deliberation: premature consensus detected "
                        f"(agreement={dissent_report.agreement_ratio:.2f}), forcing steelman pass"
                    )
                    steelman = await self.run_dissent_pass(
                        agents, question, round1, dissent_report, temperature
                    )

        logger.info("deliberation: Round 2 (cross-examination)")
        round2 = await self.run_round2(agents, question, round1, temperature)

        logger.info("deliberation: Round 3 (final crystallization)")
        round3 = await self.run_round3(
            agents, question, round2, temperature=max(0.3, temperature - 0.2)
        )

        return DeliberationResult(
            agents=list(agents),
            restate=restate,
            rounds=[round1, round2, round3],
            dissent_report=dissent_report,
            steelman=steelman,
            final_positions=round3.outputs,
            metadata={
                "round1_word_limit": self.config.round1_word_limit,
                "round2_word_limit": self.config.round2_word_limit,
                "round3_word_limit": self.config.round3_word_limit,
                "consensus_threshold": self.config.consensus_threshold,
            },
        )
