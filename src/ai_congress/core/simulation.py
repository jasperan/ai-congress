"""
Tick-based Congressional Simulation Engine

Runs a multi-phase simulation where LLM-powered congressional agents
debate, discuss, and vote on bills/topics. Each tick produces events
streamed in real-time via an async generator.
"""

import json
import logging
import os
import random
import re
import time
from enum import Enum
from typing import AsyncGenerator, Dict, List, Optional

import ollama

logger = logging.getLogger(__name__)


class Phase(Enum):
    INTRODUCTION = "introduction"
    COMMITTEE = "committee"
    FLOOR_DEBATE = "floor_debate"
    AMENDMENTS = "amendments"
    FINAL_ARGUMENTS = "final_arguments"
    VOTING = "voting"


# Phase definitions: (start_tick, end_tick, agents_per_tick_min, agents_per_tick_max)
PHASE_CONFIG = {
    Phase.INTRODUCTION:    (1,  10, 1, 2),
    Phase.COMMITTEE:       (11, 30, 2, 3),
    Phase.FLOOR_DEBATE:    (31, 60, 2, 4),
    Phase.AMENDMENTS:      (61, 80, 2, 3),
    Phase.FINAL_ARGUMENTS: (81, 90, 1, 2),
    Phase.VOTING:          (91, 100, 1, 1),
}

PHASE_PROMPTS = {
    Phase.INTRODUCTION: (
        "A new bill has been introduced: \"{topic}\"\n\n"
        "As {agent_name} ({party}, {state}), give your initial reaction to this bill. "
        "State whether you're leaning for or against it and why. Keep it brief."
    ),
    Phase.COMMITTEE: (
        "The bill \"{topic}\" is now in committee review.\n\n"
        "Recent discussion:\n{context}\n\n"
        "As {agent_name} ({party}, {state}), engage with the committee discussion. "
        "Ask pointed questions, raise concerns, or highlight benefits. Reference other members' points if relevant. Keep it brief."
    ),
    Phase.FLOOR_DEBATE: (
        "The bill \"{topic}\" has moved to floor debate.\n\n"
        "Recent discussion:\n{context}\n\n"
        "As {agent_name} ({party}, {state}), make your case on the floor. "
        "You may challenge opponents' arguments, support allies, or introduce new points. Keep it brief."
    ),
    Phase.AMENDMENTS: (
        "The bill \"{topic}\" is in the amendment phase.\n\n"
        "Recent discussion:\n{context}\n\n"
        "As {agent_name} ({party}, {state}), propose an amendment, support or oppose a proposed amendment, "
        "or cross-examine another member's position. Keep it brief."
    ),
    Phase.FINAL_ARGUMENTS: (
        "Final arguments on the bill \"{topic}\".\n\n"
        "Recent discussion:\n{context}\n\n"
        "As {agent_name} ({party}, {state}), deliver your closing statement. "
        "Summarize your position and make your final case. Keep it brief."
    ),
    Phase.VOTING: (
        "It is time to vote on the bill: \"{topic}\"\n\n"
        "Key arguments from the debate:\n{context}\n\n"
        "As {agent_name} ({party}, {state}), cast your vote. "
        "You MUST start your response with exactly one of: YEA, NAY, or ABSTAIN. "
        "Then briefly explain your rationale."
    ),
}


def _get_phase(tick: int) -> Phase:
    """Determine the current phase based on tick number."""
    for phase, (start, end, _, _) in PHASE_CONFIG.items():
        if start <= tick <= end:
            return phase
    return Phase.VOTING


def _load_agents(num_agents: int) -> List[dict]:
    """Load congress agents from config, mixing parties."""
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "config", "us_congress.json"
    )
    config_path = os.path.normpath(config_path)

    with open(config_path, "r") as f:
        all_agents = json.load(f)

    republicans = [a for a in all_agents if a.get("party") == "Republican"]
    democrats = [a for a in all_agents if a.get("party") == "Democratic"]

    # Split evenly, favoring balance
    num_r = num_agents // 2
    num_d = num_agents - num_r

    # Cap to available
    num_r = min(num_r, len(republicans))
    num_d = min(num_d, len(democrats))

    # If one side is short, fill from the other
    if num_r + num_d < num_agents:
        shortfall = num_agents - (num_r + num_d)
        if len(republicans) > num_r:
            extra = min(shortfall, len(republicans) - num_r)
            num_r += extra
            shortfall -= extra
        if shortfall > 0 and len(democrats) > num_d:
            extra = min(shortfall, len(democrats) - num_d)
            num_d += extra

    selected = republicans[:num_r] + democrats[:num_d]
    random.shuffle(selected)
    return selected


def _parse_vote(response: str) -> str:
    """Extract vote from agent response."""
    text = response.strip().upper()
    # Check first word
    first_word = text.split()[0] if text.split() else ""
    # Strip punctuation from first word
    first_word = re.sub(r"[^A-Z]", "", first_word)

    if first_word == "YEA":
        return "yea"
    elif first_word == "NAY":
        return "nay"
    elif first_word == "ABSTAIN":
        return "abstain"

    # Fallback: search anywhere in response
    if "YEA" in text and "NAY" not in text:
        return "yea"
    elif "NAY" in text and "YEA" not in text:
        return "nay"
    elif "ABSTAIN" in text:
        return "abstain"

    # Default to abstain if we can't parse
    return "abstain"


class CongressSimulation:
    """Tick-based congressional simulation engine."""

    def __init__(
        self,
        topic: str,
        num_agents: int = 10,
        num_ticks: int = 100,
        model: str = "qwen3.5:9b",
        context_window: int = 10,
        max_response_tokens: int = 256,
        temperature: float = 0.8,
    ):
        self.topic = topic
        self.num_ticks = num_ticks
        self.model = model
        self.context_window = context_window
        self.max_response_tokens = max_response_tokens
        self.temperature = temperature

        self.agents = _load_agents(min(num_agents, 20))
        self.transcript: List[Dict] = []
        self.votes: Dict[str, str] = {}
        self.agent_positions: Dict[str, str] = {}  # name -> for/against/undecided

        # Build agent lookup
        self._agent_map = {a["name"]: a for a in self.agents}

        logger.info(
            "Simulation initialized: topic=%r, agents=%d, ticks=%d, model=%s",
            topic[:60], len(self.agents), self.num_ticks, self.model,
        )

    def _build_context(self) -> str:
        """Build context string from recent transcript entries."""
        recent = self.transcript[-self.context_window:]
        if not recent:
            return "(No prior discussion.)"

        lines = []
        for entry in recent:
            name = entry["agent_name"]
            agent = self._agent_map.get(name, {})
            party = agent.get("party", "")
            # Truncate long responses for context
            text = entry["response"][:300]
            lines.append(f"{name} ({party}): {text}")
        return "\n".join(lines)

    def _select_speakers(self, tick: int, phase: Phase) -> List[dict]:
        """Select which agents speak on this tick."""
        _, _, min_speakers, max_speakers = PHASE_CONFIG[phase]

        if phase == Phase.VOTING:
            # During voting, each agent votes once in order
            vote_tick_offset = tick - PHASE_CONFIG[Phase.VOTING][0]
            if vote_tick_offset < len(self.agents):
                return [self.agents[vote_tick_offset]]
            return []

        num_speakers = random.randint(min_speakers, max_speakers)
        num_speakers = min(num_speakers, len(self.agents))

        # Weighted selection: agents who spoke less get slightly higher chance
        speak_counts = {}
        for entry in self.transcript:
            name = entry["agent_name"]
            speak_counts[name] = speak_counts.get(name, 0) + 1

        # Build weights (inverse of speak count)
        weights = []
        for agent in self.agents:
            count = speak_counts.get(agent["name"], 0)
            weights.append(1.0 / (1.0 + count))

        # Normalize
        total = sum(weights)
        weights = [w / total for w in weights]

        selected = []
        available = list(range(len(self.agents)))
        available_weights = list(weights)

        for _ in range(num_speakers):
            if not available:
                break
            # Normalize available weights
            w_total = sum(available_weights)
            if w_total <= 0:
                break
            norm_w = [w / w_total for w in available_weights]

            idx = random.choices(range(len(available)), weights=norm_w, k=1)[0]
            selected.append(self.agents[available[idx]])
            available.pop(idx)
            available_weights.pop(idx)

        return selected

    def _build_tick_prompt(self, agent: dict, phase: Phase) -> str:
        """Build the prompt for a specific agent on a specific tick."""
        template = PHASE_PROMPTS[phase]
        context = self._build_context()

        return template.format(
            topic=self.topic,
            agent_name=agent["name"],
            party=agent.get("party", "Unknown"),
            state=agent.get("state", "Unknown"),
            context=context,
        )

    def _update_position(self, agent_name: str, response: str):
        """Track agent's leaning based on keywords in response."""
        text = response.lower()
        support_words = ["support", "favor", "agree", "endorse", "back this", "yea", "beneficial"]
        oppose_words = ["oppose", "against", "reject", "concerned", "nay", "harmful", "dangerous"]

        support_score = sum(1 for w in support_words if w in text)
        oppose_score = sum(1 for w in oppose_words if w in text)

        if support_score > oppose_score:
            self.agent_positions[agent_name] = "for"
        elif oppose_score > support_score:
            self.agent_positions[agent_name] = "against"
        else:
            self.agent_positions.setdefault(agent_name, "undecided")

    async def run(self) -> AsyncGenerator[Dict, None]:
        """Run the simulation, yielding events as they occur."""
        # Build phase summary for the start event
        phases_summary = {}
        for phase, (start, end, min_s, max_s) in PHASE_CONFIG.items():
            phases_summary[phase.value] = {
                "start_tick": start,
                "end_tick": end,
                "speakers_per_tick": f"{min_s}-{max_s}",
            }

        agent_summaries = [
            {
                "name": a["name"],
                "party": a.get("party", "Unknown"),
                "state": a.get("state", "Unknown"),
                "chamber": a.get("chamber", "Unknown"),
            }
            for a in self.agents
        ]

        yield {
            "type": "simulation_start",
            "topic": self.topic,
            "agents": agent_summaries,
            "max_ticks": self.num_ticks,
            "phases": phases_summary,
        }

        for tick in range(1, self.num_ticks + 1):
            phase = _get_phase(tick)
            phase_name = phase.value.replace("_", " ").title()

            yield {
                "type": "tick_start",
                "tick": tick,
                "phase": phase.value,
                "phase_name": phase_name,
            }

            speakers = self._select_speakers(tick, phase)
            messages_this_tick = 0

            if len(speakers) > 1 and phase != Phase.VOTING:
                yield {
                    "type": "group_discussion",
                    "tick": tick,
                    "participants": [s["name"] for s in speakers],
                    "topic_focus": phase_name,
                }

            for agent in speakers:
                yield {
                    "type": "agent_speaking",
                    "tick": tick,
                    "agent_name": agent["name"],
                    "party": agent.get("party", "Unknown"),
                    "state": agent.get("state", "Unknown"),
                    "chamber": agent.get("chamber", "Unknown"),
                }

                # Inline streaming: yield tokens as they arrive from Ollama
                prompt = self._build_tick_prompt(agent, phase)
                messages = [
                    {"role": "system", "content": agent["system_prompt"]},
                    {"role": "user", "content": prompt},
                ]
                full_response = ""
                start_ms = time.monotonic()

                try:
                    client = ollama.AsyncClient(host="http://127.0.0.1:11434")
                    response_stream = await client.chat(
                        model=self.model,
                        messages=messages,
                        stream=True,
                        think=False,
                        options={
                            "num_predict": self.max_response_tokens,
                            "temperature": self.temperature,
                        },
                    )

                    async for chunk in response_stream:
                        # Support both dict-style and attribute-style access
                        msg = chunk.get("message", {}) if hasattr(chunk, "get") else getattr(chunk, "message", {})
                        token = msg.get("content", "") if hasattr(msg, "get") else getattr(msg, "content", "")
                        if token:
                            full_response += token
                            yield {
                                "type": "token_stream",
                                "tick": tick,
                                "agent_name": agent["name"],
                                "tokens": token,
                            }

                except Exception as e:
                    logger.error("Ollama query failed for %s: %s", agent["name"], e)
                    full_response = f"[Connection error: {e}]"

                latency_ms = int((time.monotonic() - start_ms) * 1000)

                # Strip residual <think> blocks
                full_response = re.sub(
                    r"<think>.*?</think>", "", full_response, flags=re.DOTALL
                ).strip()

                # Record transcript
                self.transcript.append({
                    "tick": tick,
                    "phase": phase.value,
                    "agent_name": agent["name"],
                    "response": full_response,
                })

                # Update position tracking
                self._update_position(agent["name"], full_response)

                # Handle voting phase
                if phase == Phase.VOTING:
                    vote = _parse_vote(full_response)
                    self.votes[agent["name"]] = vote

                    yield {
                        "type": "vote_cast",
                        "tick": tick,
                        "agent_name": agent["name"],
                        "vote": vote,
                        "rationale": full_response,
                    }

                yield {
                    "type": "agent_done",
                    "tick": tick,
                    "agent_name": agent["name"],
                    "full_response": full_response,
                    "latency_ms": latency_ms,
                }

                messages_this_tick += 1

            yield {
                "type": "tick_end",
                "tick": tick,
                "messages_this_tick": messages_this_tick,
            }

        # Tally votes
        yea = sum(1 for v in self.votes.values() if v == "yea")
        nay = sum(1 for v in self.votes.values() if v == "nay")
        abstain = sum(1 for v in self.votes.values() if v == "abstain")

        if yea > nay:
            result = "PASSED"
        elif nay > yea:
            result = "FAILED"
        else:
            result = "TIED"

        yield {
            "type": "simulation_complete",
            "yea": yea,
            "nay": nay,
            "abstain": abstain,
            "result": result,
        }

        logger.info(
            "Simulation complete: %s (yea=%d, nay=%d, abstain=%d)",
            result, yea, nay, abstain,
        )
