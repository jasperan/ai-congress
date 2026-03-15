"""
Tick-based Congressional Simulation Engine

Runs a multi-phase simulation where LLM-powered congressional agents
debate, discuss, and vote on bills/topics. Each tick produces events
streamed in real-time via an async generator.

Features:
- Opinion drift with continuous sentiment scoring (-1.0 to +1.0)
- Direct address / cross-examination between agents
- Mutable bill text through amendment proposals and votes
- Lobby/witness agents (non-voting, Committee phase only)
- Filibuster + cloture mechanics in Floor Debate
- Persuasion tracking between agents
- Historical calibration scoring
"""

import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncGenerator, Dict, List, Optional, Tuple

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

# ── Sentiment Scoring ────────────────────────────────────────────────────────

SUPPORT_SIGNALS = {
    "strongly support": 0.9, "wholeheartedly support": 0.9,
    "fully endorse": 0.8, "endorse": 0.7, "enthusiastically": 0.7,
    "support": 0.6, "favor": 0.6, "agree": 0.5, "back this": 0.6,
    "beneficial": 0.5, "necessary": 0.4, "promising": 0.4,
    "important step": 0.4, "common ground": 0.3, "merit": 0.3,
    "yea": 0.8, "vote yes": 0.8, "in favor": 0.6,
}

OPPOSE_SIGNALS = {
    "strongly oppose": 0.9, "vehemently oppose": 0.9,
    "categorically reject": 0.8, "reject": 0.7, "dangerous": 0.8,
    "oppose": 0.6, "against": 0.6, "concerned": 0.4,
    "harmful": 0.7, "unconstitutional": 0.8, "overreach": 0.6,
    "unacceptable": 0.7, "reckless": 0.7, "misguided": 0.5,
    "nay": 0.8, "vote no": 0.8, "cannot support": 0.7,
}

# ── Historical Stances ───────────────────────────────────────────────────────
# Known positions for accuracy scoring (topic_keyword -> {senator: expected_vote})

HISTORICAL_STANCES = {
    "ai": {
        "Chuck Schumer": "yea", "Dick Durbin": "yea", "Patty Murray": "yea",
        "Ron Wyden": "yea", "Chuck Grassley": "nay", "JD Vance": "nay",
        "Mitch McConnell": "nay", "John Thune": "nay", "John Cornyn": "nay",
    },
    "gun": {
        "Chuck Schumer": "yea", "Dick Durbin": "yea", "Patty Murray": "yea",
        "Ron Wyden": "yea", "Chuck Grassley": "nay", "JD Vance": "nay",
        "Mitch McConnell": "nay", "John Thune": "nay", "John Cornyn": "nay",
    },
    "healthcare": {
        "Chuck Schumer": "yea", "Dick Durbin": "yea", "Patty Murray": "yea",
        "Ron Wyden": "yea", "Chuck Grassley": "nay", "JD Vance": "nay",
        "Mitch McConnell": "nay", "John Thune": "nay", "John Cornyn": "nay",
    },
    "immigration": {
        "Chuck Schumer": "yea", "Dick Durbin": "yea", "Patty Murray": "yea",
        "Ron Wyden": "yea", "Chuck Grassley": "nay", "JD Vance": "nay",
        "Mitch McConnell": "nay", "John Thune": "nay", "John Cornyn": "nay",
    },
    "climate": {
        "Chuck Schumer": "yea", "Dick Durbin": "yea", "Patty Murray": "yea",
        "Ron Wyden": "yea", "Chuck Grassley": "nay", "JD Vance": "nay",
        "Mitch McConnell": "nay", "John Thune": "nay", "John Cornyn": "nay",
    },
}

# ── Data Classes ─────────────────────────────────────────────────────────────


@dataclass
class Amendment:
    id: int
    proposer: str
    text: str
    tick_proposed: int
    votes_yea: int = 0
    votes_nay: int = 0
    status: str = "pending"  # pending, passed, failed


@dataclass
class FilibusterState:
    agent_name: str
    start_tick: int
    duration: int  # how many ticks total
    ticks_elapsed: int = 0
    cloture_attempted: bool = False
    ended: bool = False


# ── Phase Prompts ────────────────────────────────────────────────────────────

PHASE_PROMPTS = {
    Phase.INTRODUCTION: (
        "A new bill has been introduced: \"{topic}\"\n\n"
        "As {agent_name} ({party}, {state}), give your initial reaction to this bill. "
        "State whether you're leaning for or against it and why. Keep it brief."
    ),
    Phase.COMMITTEE: (
        "The bill \"{topic}\" is now in committee review.\n\n"
        "Recent discussion:\n{context}\n\n"
        "{lobby_context}"
        "As {agent_name} ({party}, {state}), engage with the committee discussion. "
        "Ask pointed questions, raise concerns, or highlight benefits. "
        "{address_instruction}"
        "Keep it brief."
    ),
    Phase.FLOOR_DEBATE: (
        "The bill \"{topic}\" has moved to floor debate.\n\n"
        "Recent discussion:\n{context}\n\n"
        "As {agent_name} ({party}, {state}), make your case on the floor. "
        "{address_instruction}"
        "Keep it brief."
    ),
    Phase.AMENDMENTS: (
        "The bill \"{topic}\" is in the amendment phase.\n\n"
        "{amendment_context}"
        "Recent discussion:\n{context}\n\n"
        "As {agent_name} ({party}, {state}), you may propose an amendment "
        "(start with 'AMENDMENT:'), support or oppose a proposed amendment, "
        "or cross-examine another member's position. "
        "{address_instruction}"
        "Keep it brief."
    ),
    Phase.FINAL_ARGUMENTS: (
        "Final arguments on the bill: \"{topic}\"\n\n"
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

FILIBUSTER_PROMPT = (
    "You are filibustering the bill: \"{topic}\"\n\n"
    "As {agent_name} ({party}, {state}), continue your extended speech against this bill. "
    "Be verbose, cite precedents, tell stories, read passages, do whatever it takes to "
    "hold the floor and delay the vote. Keep going."
)

LOBBY_PROMPT = (
    "You are testifying before the Senate committee on the bill: \"{topic}\"\n\n"
    "As {lobby_name} ({affiliation}), present your {bias} perspective. "
    "Be persuasive and specific. Cite data, case studies, or consequences. Keep it brief."
)

AMENDMENT_VOTE_PROMPT = (
    "Amendment #{amendment_id} has been proposed: \"{amendment_text}\"\n\n"
    "As {agent_name} ({party}, {state}), vote on this amendment. "
    "Reply with exactly YEA or NAY followed by a one-sentence reason."
)


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

    num_r = num_agents // 2
    num_d = num_agents - num_r
    num_r = min(num_r, len(republicans))
    num_d = min(num_d, len(democrats))

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


def _load_lobby_agents() -> List[dict]:
    """Load lobby/witness agents from config."""
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "config", "lobby_agents.json"
    )
    config_path = os.path.normpath(config_path)

    if not os.path.exists(config_path):
        logger.warning("No lobby_agents.json found, skipping lobby agents")
        return []

    with open(config_path, "r") as f:
        return json.load(f)


def _parse_vote(response: str) -> str:
    """Extract vote from agent response."""
    text = response.strip().upper()
    first_word = text.split()[0] if text.split() else ""
    first_word = re.sub(r"[^A-Z]", "", first_word)

    if first_word == "YEA":
        return "yea"
    elif first_word == "NAY":
        return "nay"
    elif first_word == "ABSTAIN":
        return "abstain"

    if "YEA" in text and "NAY" not in text:
        return "yea"
    elif "NAY" in text and "YEA" not in text:
        return "nay"
    elif "ABSTAIN" in text:
        return "abstain"

    return "abstain"


def _compute_sentiment(text: str) -> float:
    """Compute sentiment score from text. Returns -1.0 to +1.0."""
    text_lower = text.lower()

    support_score = 0.0
    oppose_score = 0.0
    matches = 0

    for phrase, weight in SUPPORT_SIGNALS.items():
        count = text_lower.count(phrase)
        if count:
            support_score += weight * count
            matches += count

    for phrase, weight in OPPOSE_SIGNALS.items():
        count = text_lower.count(phrase)
        if count:
            oppose_score += weight * count
            matches += count

    if matches == 0:
        return 0.0

    raw = (support_score - oppose_score) / max(support_score + oppose_score, 0.01)
    return max(-1.0, min(1.0, raw))


def _extract_amendment(response: str) -> Optional[str]:
    """Try to extract an amendment proposal from a response."""
    patterns = [
        r"AMENDMENT:\s*(.+?)(?:\n\n|\Z)",
        r"I (?:hereby )?propose(?:\s+an amendment)?:\s*(.+?)(?:\n\n|\Z)",
        r"I move to amend.*?:\s*(.+?)(?:\n\n|\Z)",
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        if match:
            text = match.group(1).strip()
            if len(text) > 10:
                return text[:200]
    return None


class CongressSimulation:
    """Tick-based congressional simulation engine with advanced mechanics."""

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
        self.current_bill_text = topic  # mutable; amendments append to this
        self.num_ticks = num_ticks
        self.model = model
        self.context_window = context_window
        self.max_response_tokens = max_response_tokens
        self.temperature = temperature

        self.agents = _load_agents(min(num_agents, 20))
        self.lobby_agents = _load_lobby_agents()
        self.transcript: List[Dict] = []
        self.votes: Dict[str, str] = {}

        # Opinion drift: continuous sentiment per agent
        self.agent_sentiment: Dict[str, float] = {a["name"]: 0.0 for a in self.agents}
        self.sentiment_history: Dict[str, List[Tuple[int, float]]] = {
            a["name"]: [] for a in self.agents
        }

        # Amendment tracking
        self.amendments: List[Amendment] = []
        self._next_amendment_id = 1

        # Filibuster state
        self.filibuster: Optional[FilibusterState] = None
        self._filibuster_used = False  # only one per simulation

        # Persuasion tracking: (influencer, influenced) -> cumulative strength
        self.persuasion_edges: Dict[Tuple[str, str], float] = {}

        # Agent lookup
        self._agent_map = {a["name"]: a for a in self.agents}

        logger.info(
            "Simulation initialized: topic=%r, agents=%d, lobby=%d, ticks=%d, model=%s",
            topic[:60], len(self.agents), len(self.lobby_agents),
            self.num_ticks, self.model,
        )

    # ── Context Building ─────────────────────────────────────────────────

    def _build_context(self) -> str:
        recent = self.transcript[-self.context_window:]
        if not recent:
            return "(No prior discussion.)"

        lines = []
        for entry in recent:
            name = entry["agent_name"]
            agent = self._agent_map.get(name, {})
            party = agent.get("party", entry.get("affiliation", ""))
            text = entry["response"][:300]
            lines.append(f"{name} ({party}): {text}")
        return "\n".join(lines)

    def _build_lobby_context(self) -> str:
        """Build context from lobby agent testimony (for Committee phase)."""
        lobby_entries = [
            e for e in self.transcript if e.get("is_lobby")
        ]
        if not lobby_entries:
            return ""

        lines = ["Expert/lobby testimony so far:"]
        for entry in lobby_entries[-3:]:
            lines.append(f"  {entry['agent_name']} ({entry.get('affiliation', '')}): {entry['response'][:200]}")
        return "\n".join(lines) + "\n\n"

    def _build_amendment_context(self) -> str:
        """Build context about pending/passed amendments."""
        if not self.amendments:
            return ""

        lines = ["Current amendments:"]
        for a in self.amendments:
            status_str = a.status.upper()
            lines.append(f"  #{a.id} [{status_str}] by {a.proposer}: {a.text[:100]}")
        return "\n".join(lines) + "\n\n"

    # ── Direct Address ───────────────────────────────────────────────────

    def _find_address_target(self, speaker_name: str) -> Optional[Tuple[str, str]]:
        """Find an opposing agent to directly address. Returns (name, snippet)."""
        speaker_sentiment = self.agent_sentiment.get(speaker_name, 0.0)

        candidates = []
        for entry in reversed(self.transcript[-20:]):
            name = entry["agent_name"]
            if name == speaker_name or name not in self._agent_map:
                continue
            their_sentiment = self.agent_sentiment.get(name, 0.0)
            # Opposing sentiment = good target
            if (speaker_sentiment >= 0 and their_sentiment < -0.2) or \
               (speaker_sentiment < 0 and their_sentiment > 0.2):
                snippet = entry["response"][:120]
                candidates.append((name, snippet))

        if not candidates:
            return None
        return random.choice(candidates[:3])

    def _build_address_instruction(self, speaker: dict, phase: Phase) -> str:
        """Build direct address instruction for the prompt."""
        if phase in (Phase.INTRODUCTION, Phase.VOTING, Phase.FINAL_ARGUMENTS):
            return ""

        target = self._find_address_target(speaker["name"])
        if not target:
            return "Reference other members' points if relevant. "

        name, snippet = target
        return (
            f"You MUST directly respond to {name}'s argument: \"{snippet}...\" "
            f"Challenge, rebut, or build on their point. "
        )

    # ── Sentiment & Persuasion ───────────────────────────────────────────

    def _update_sentiment(self, agent_name: str, response: str, tick: int) -> Dict:
        """Update agent sentiment and return drift event data."""
        old_score = self.agent_sentiment.get(agent_name, 0.0)
        new_raw = _compute_sentiment(response)

        # Blend with momentum (70% new, 30% old) for smooth drift
        new_score = round(0.7 * new_raw + 0.3 * old_score, 3)
        new_score = max(-1.0, min(1.0, new_score))

        self.agent_sentiment[agent_name] = new_score
        self.sentiment_history[agent_name].append((tick, new_score))

        drift = round(new_score - old_score, 3)
        direction = "toward" if drift > 0 else "against" if drift < 0 else "steady"

        return {
            "agent_name": agent_name,
            "old_score": old_score,
            "new_score": new_score,
            "drift": drift,
            "direction": direction,
        }

    def _update_persuasion(self, speaker_name: str, tick: int):
        """Track persuasion: did this speaker's prior argument shift anyone?"""
        speaker_entries = [
            e for e in self.transcript
            if e["agent_name"] == speaker_name and e["tick"] < tick
        ]
        if not speaker_entries:
            return

        last_speech_tick = speaker_entries[-1]["tick"]

        # Check if any agent's sentiment shifted toward speaker's position
        # between last_speech_tick and now
        speaker_pos = self.agent_sentiment.get(speaker_name, 0.0)

        for name, history in self.sentiment_history.items():
            if name == speaker_name:
                continue
            # Find sentiment before and after the speech
            before = [s for t, s in history if t <= last_speech_tick]
            after = [s for t, s in history if t > last_speech_tick]
            if not before or not after:
                continue

            old_s = before[-1]
            new_s = after[-1]
            shift = new_s - old_s

            # Did they move toward the speaker's position?
            if (speaker_pos > 0 and shift > 0.05) or (speaker_pos < 0 and shift < -0.05):
                key = (speaker_name, name)
                self.persuasion_edges[key] = self.persuasion_edges.get(key, 0.0) + abs(shift)

    # ── Speaker Selection ────────────────────────────────────────────────

    def _select_speakers(self, tick: int, phase: Phase) -> List[dict]:
        _, _, min_speakers, max_speakers = PHASE_CONFIG[phase]

        if phase == Phase.VOTING:
            vote_tick_offset = tick - PHASE_CONFIG[Phase.VOTING][0]
            if vote_tick_offset < len(self.agents):
                return [self.agents[vote_tick_offset]]
            return []

        num_speakers = random.randint(min_speakers, max_speakers)
        num_speakers = min(num_speakers, len(self.agents))

        speak_counts = {}
        for entry in self.transcript:
            if not entry.get("is_lobby"):
                name = entry["agent_name"]
                speak_counts[name] = speak_counts.get(name, 0) + 1

        weights = []
        for agent in self.agents:
            count = speak_counts.get(agent["name"], 0)
            weights.append(1.0 / (1.0 + count))

        total = sum(weights)
        weights = [w / total for w in weights]

        selected = []
        available = list(range(len(self.agents)))
        available_weights = list(weights)

        for _ in range(num_speakers):
            if not available:
                break
            w_total = sum(available_weights)
            if w_total <= 0:
                break
            norm_w = [w / w_total for w in available_weights]

            idx = random.choices(range(len(available)), weights=norm_w, k=1)[0]
            selected.append(self.agents[available[idx]])
            available.pop(idx)
            available_weights.pop(idx)

        return selected

    # ── Filibuster ───────────────────────────────────────────────────────

    def _check_filibuster_start(self, tick: int, phase: Phase) -> bool:
        """Check if a filibuster should start this tick."""
        if phase != Phase.FLOOR_DEBATE or self._filibuster_used:
            return False
        if self.filibuster is not None:
            return False
        if tick < 35 or tick > 52:  # only mid-debate
            return False

        # Find strongly opposing agents
        strong_opposers = [
            name for name, score in self.agent_sentiment.items()
            if score < -0.4
        ]
        if not strong_opposers:
            return False

        # 12% chance per tick
        if random.random() > 0.12:
            return False

        filibusterer = random.choice(strong_opposers)
        duration = random.randint(3, 5)
        self.filibuster = FilibusterState(
            agent_name=filibusterer,
            start_tick=tick,
            duration=duration,
        )
        self._filibuster_used = True
        return True

    def _attempt_cloture(self) -> Tuple[bool, int, int]:
        """Attempt cloture vote. Returns (passed, yea, nay)."""
        yea = 0
        nay = 0
        for name, score in self.agent_sentiment.items():
            if name == self.filibuster.agent_name:
                nay += 1  # filibusterer always votes nay on cloture
                continue
            # Agents wanting to proceed (positive sentiment) vote yea
            if score > 0:
                yea += 1
            elif score < -0.3:
                nay += 1  # strong opposers support filibuster
            else:
                # Moderate agents lean toward ending filibuster
                yea += 1
        threshold = int(len(self.agents) * 0.6)
        return yea >= threshold, yea, nay

    # ── Amendment Mechanics ──────────────────────────────────────────────

    def _process_amendment_proposal(self, agent_name: str, response: str, tick: int):
        """Check if response contains an amendment proposal."""
        if len(self.amendments) >= 3:  # max 3 amendments per simulation
            return None

        text = _extract_amendment(response)
        if not text:
            return None

        amendment = Amendment(
            id=self._next_amendment_id,
            proposer=agent_name,
            text=text,
            tick_proposed=tick,
        )
        self._next_amendment_id += 1
        self.amendments.append(amendment)
        return amendment

    async def _vote_on_amendments(self, tick: int) -> AsyncGenerator[Dict, None]:
        """Hold votes on pending amendments."""
        pending = [a for a in self.amendments if a.status == "pending"]
        if not pending:
            return

        for amendment in pending:
            yield {
                "type": "amendment_voting",
                "tick": tick,
                "amendment_id": amendment.id,
                "amendment_text": amendment.text,
            }

            # Quick vote: use sentiment as proxy (avoids 10 extra LLM calls)
            for agent in self.agents:
                name = agent["name"]
                sentiment = self.agent_sentiment.get(name, 0.0)

                # Agents with positive sentiment tend to support amendments
                # (they want the bill to pass, amendments help refine it)
                # Agents with negative sentiment oppose
                if name == amendment.proposer:
                    amendment.votes_yea += 1
                elif sentiment > 0.1:
                    amendment.votes_yea += 1
                elif sentiment < -0.1:
                    amendment.votes_nay += 1
                else:
                    # Undecided: coin flip
                    if random.random() > 0.5:
                        amendment.votes_yea += 1
                    else:
                        amendment.votes_nay += 1

            if amendment.votes_yea > amendment.votes_nay:
                amendment.status = "passed"
                self.current_bill_text += f" [Amended: {amendment.text[:100]}]"

                yield {
                    "type": "amendment_vote_result",
                    "tick": tick,
                    "amendment_id": amendment.id,
                    "result": "passed",
                    "yea": amendment.votes_yea,
                    "nay": amendment.votes_nay,
                    "new_bill_text": self.current_bill_text,
                }
            else:
                amendment.status = "failed"
                yield {
                    "type": "amendment_vote_result",
                    "tick": tick,
                    "amendment_id": amendment.id,
                    "result": "failed",
                    "yea": amendment.votes_yea,
                    "nay": amendment.votes_nay,
                }

    # ── Historical Calibration ───────────────────────────────────────────

    def _compute_historical_accuracy(self) -> Optional[Dict]:
        """Compare simulation votes against known historical stances."""
        topic_lower = self.topic.lower()

        # Find matching topic
        matched_stances = None
        for keyword, stances in HISTORICAL_STANCES.items():
            if keyword in topic_lower:
                matched_stances = stances
                break

        if not matched_stances:
            return None

        correct = 0
        total = 0
        details = []
        for name, expected in matched_stances.items():
            actual = self.votes.get(name)
            if actual:
                total += 1
                match = actual == expected
                if match:
                    correct += 1
                details.append({
                    "agent": name,
                    "expected": expected,
                    "actual": actual,
                    "match": match,
                })

        if total == 0:
            return None

        return {
            "accuracy": round(correct / total * 100, 1),
            "correct": correct,
            "total": total,
            "details": details,
        }

    # ── LLM Query ────────────────────────────────────────────────────────

    async def _stream_llm(
        self, messages: List[Dict], tick: int, agent_name: str,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[Dict, None]:
        """Stream LLM response, yielding token events. Returns full response via .result attr."""
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
                    "num_predict": max_tokens or self.max_response_tokens,
                    "temperature": self.temperature,
                },
            )

            async for chunk in response_stream:
                msg = chunk.get("message", {}) if hasattr(chunk, "get") else getattr(chunk, "message", {})
                token = msg.get("content", "") if hasattr(msg, "get") else getattr(msg, "content", "")
                if token:
                    full_response += token
                    yield {
                        "type": "token_stream",
                        "tick": tick,
                        "agent_name": agent_name,
                        "tokens": token,
                    }

        except Exception as e:
            logger.error("Ollama query failed for %s: %s", agent_name, e)
            full_response = f"[Connection error: {e}]"

        latency_ms = int((time.monotonic() - start_ms) * 1000)

        # Strip residual <think> blocks
        full_response = re.sub(
            r"<think>.*?</think>", "", full_response, flags=re.DOTALL
        ).strip()

        # Store result on the generator for caller to access
        self._last_response = full_response
        self._last_latency = latency_ms

    # ── Prompt Building ──────────────────────────────────────────────────

    def _build_tick_prompt(self, agent: dict, phase: Phase) -> str:
        template = PHASE_PROMPTS[phase]
        context = self._build_context()
        address_instruction = self._build_address_instruction(agent, phase)
        lobby_context = self._build_lobby_context() if phase == Phase.COMMITTEE else ""
        amendment_context = self._build_amendment_context() if phase == Phase.AMENDMENTS else ""

        return template.format(
            topic=self.current_bill_text,
            agent_name=agent["name"],
            party=agent.get("party", "Unknown"),
            state=agent.get("state", "Unknown"),
            context=context,
            address_instruction=address_instruction,
            lobby_context=lobby_context,
            amendment_context=amendment_context,
        )

    # ── Main Simulation Loop ─────────────────────────────────────────────

    async def run(self) -> AsyncGenerator[Dict, None]:
        """Run the simulation, yielding events as they occur."""
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

        lobby_summaries = [
            {
                "name": la["name"],
                "affiliation": la.get("affiliation", ""),
                "bias": la.get("bias", "neutral"),
            }
            for la in self.lobby_agents
        ]

        yield {
            "type": "simulation_start",
            "topic": self.topic,
            "agents": agent_summaries,
            "lobby_agents": lobby_summaries,
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

            # ── Filibuster handling ──────────────────────────────────
            if self.filibuster and not self.filibuster.ended:
                fb = self.filibuster
                fb.ticks_elapsed += 1

                agent = self._agent_map[fb.agent_name]
                prompt = FILIBUSTER_PROMPT.format(
                    topic=self.current_bill_text,
                    agent_name=fb.agent_name,
                    party=agent.get("party", "Unknown"),
                    state=agent.get("state", "Unknown"),
                )

                yield {
                    "type": "agent_speaking",
                    "tick": tick,
                    "agent_name": fb.agent_name,
                    "party": agent.get("party", "Unknown"),
                    "state": agent.get("state", "Unknown"),
                    "is_filibuster": True,
                }

                messages = [
                    {"role": "system", "content": agent["system_prompt"]},
                    {"role": "user", "content": prompt},
                ]
                async for event in self._stream_llm(messages, tick, fb.agent_name,
                                                     max_tokens=512):
                    yield event

                full_response = self._last_response
                self.transcript.append({
                    "tick": tick,
                    "phase": phase.value,
                    "agent_name": fb.agent_name,
                    "response": full_response,
                    "is_filibuster": True,
                })

                yield {
                    "type": "agent_done",
                    "tick": tick,
                    "agent_name": fb.agent_name,
                    "full_response": full_response,
                    "latency_ms": self._last_latency,
                    "is_filibuster": True,
                }

                # Attempt cloture after 2 ticks
                if fb.ticks_elapsed >= 2 and not fb.cloture_attempted:
                    fb.cloture_attempted = True
                    passed, yea, nay = self._attempt_cloture()

                    yield {
                        "type": "cloture_vote",
                        "tick": tick,
                        "agent_name": fb.agent_name,
                        "result": "passed" if passed else "failed",
                        "yea": yea,
                        "nay": nay,
                    }

                    if passed:
                        fb.ended = True
                        yield {
                            "type": "filibuster_end",
                            "tick": tick,
                            "agent_name": fb.agent_name,
                            "reason": "cloture",
                        }

                # End by exhaustion
                if fb.ticks_elapsed >= fb.duration:
                    fb.ended = True
                    if not fb.cloture_attempted:
                        yield {
                            "type": "filibuster_end",
                            "tick": tick,
                            "agent_name": fb.agent_name,
                            "reason": "exhaustion",
                        }

                yield {"type": "tick_end", "tick": tick, "messages_this_tick": 1}
                continue  # skip normal speakers during filibuster

            # ── Check for filibuster start ───────────────────────────
            if self._check_filibuster_start(tick, phase):
                yield {
                    "type": "filibuster_start",
                    "tick": tick,
                    "agent_name": self.filibuster.agent_name,
                    "duration": self.filibuster.duration,
                }
                # Filibuster will be handled next iteration
                yield {"type": "tick_end", "tick": tick, "messages_this_tick": 0}
                continue

            # ── Lobby agents (Committee phase only) ──────────────────
            if phase == Phase.COMMITTEE and self.lobby_agents:
                # One lobby agent testifies every 5 ticks
                lobby_idx = (tick - PHASE_CONFIG[Phase.COMMITTEE][0]) // 5
                if lobby_idx < len(self.lobby_agents) and \
                   (tick - PHASE_CONFIG[Phase.COMMITTEE][0]) % 5 == 0:
                    la = self.lobby_agents[lobby_idx]

                    yield {
                        "type": "lobby_speaking",
                        "tick": tick,
                        "agent_name": la["name"],
                        "affiliation": la.get("affiliation", ""),
                        "bias": la.get("bias", "neutral"),
                    }

                    prompt = LOBBY_PROMPT.format(
                        topic=self.current_bill_text,
                        lobby_name=la["name"],
                        affiliation=la.get("affiliation", ""),
                        bias=la.get("bias", "neutral"),
                    )
                    messages = [
                        {"role": "system", "content": la.get("system_prompt",
                            f"You are {la['name']}, representing {la.get('affiliation', '')}.")},
                        {"role": "user", "content": prompt},
                    ]
                    async for event in self._stream_llm(messages, tick, la["name"]):
                        yield event

                    full_response = self._last_response
                    self.transcript.append({
                        "tick": tick,
                        "phase": phase.value,
                        "agent_name": la["name"],
                        "response": full_response,
                        "is_lobby": True,
                        "affiliation": la.get("affiliation", ""),
                    })

                    yield {
                        "type": "agent_done",
                        "tick": tick,
                        "agent_name": la["name"],
                        "full_response": full_response,
                        "latency_ms": self._last_latency,
                        "is_lobby": True,
                    }

            # ── Amendment votes at tick 75 ───────────────────────────
            if tick == 75 and phase == Phase.AMENDMENTS:
                async for event in self._vote_on_amendments(tick):
                    yield event

            # ── Regular speakers ─────────────────────────────────────
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
                # Direct address detection
                address_target = self._find_address_target(agent["name"])
                if address_target and phase not in (Phase.INTRODUCTION, Phase.VOTING):
                    yield {
                        "type": "direct_address",
                        "tick": tick,
                        "speaker": agent["name"],
                        "target": address_target[0],
                        "snippet": address_target[1][:80],
                    }

                yield {
                    "type": "agent_speaking",
                    "tick": tick,
                    "agent_name": agent["name"],
                    "party": agent.get("party", "Unknown"),
                    "state": agent.get("state", "Unknown"),
                    "chamber": agent.get("chamber", "Unknown"),
                    "sentiment": self.agent_sentiment.get(agent["name"], 0.0),
                }

                prompt = self._build_tick_prompt(agent, phase)
                messages = [
                    {"role": "system", "content": agent["system_prompt"]},
                    {"role": "user", "content": prompt},
                ]
                async for event in self._stream_llm(messages, tick, agent["name"]):
                    yield event

                full_response = self._last_response
                latency_ms = self._last_latency

                # Record transcript
                self.transcript.append({
                    "tick": tick,
                    "phase": phase.value,
                    "agent_name": agent["name"],
                    "response": full_response,
                })

                # Update sentiment and emit drift event
                drift_data = self._update_sentiment(agent["name"], full_response, tick)
                yield {
                    "type": "opinion_drift",
                    "tick": tick,
                    **drift_data,
                }

                # Update persuasion tracking
                self._update_persuasion(agent["name"], tick)

                # Check for amendment proposals
                if phase == Phase.AMENDMENTS:
                    amendment = self._process_amendment_proposal(
                        agent["name"], full_response, tick
                    )
                    if amendment:
                        yield {
                            "type": "amendment_proposed",
                            "tick": tick,
                            "agent_name": agent["name"],
                            "amendment_id": amendment.id,
                            "amendment_text": amendment.text,
                        }

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
                        "final_sentiment": self.agent_sentiment.get(agent["name"], 0.0),
                    }

                yield {
                    "type": "agent_done",
                    "tick": tick,
                    "agent_name": agent["name"],
                    "full_response": full_response,
                    "latency_ms": latency_ms,
                    "sentiment": self.agent_sentiment.get(agent["name"], 0.0),
                }

                messages_this_tick += 1

            # ── Emit persuasion updates every 10 ticks ───────────────
            if tick % 10 == 0 and self.persuasion_edges:
                top_edges = sorted(
                    self.persuasion_edges.items(), key=lambda x: x[1], reverse=True
                )[:5]
                for (influencer, influenced), strength in top_edges:
                    yield {
                        "type": "persuasion_update",
                        "tick": tick,
                        "influencer": influencer,
                        "influenced": influenced,
                        "strength": round(strength, 3),
                    }

            yield {
                "type": "tick_end",
                "tick": tick,
                "messages_this_tick": messages_this_tick,
            }

        # ── Final Tally ──────────────────────────────────────────────────
        yea = sum(1 for v in self.votes.values() if v == "yea")
        nay = sum(1 for v in self.votes.values() if v == "nay")
        abstain = sum(1 for v in self.votes.values() if v == "abstain")

        if yea > nay:
            result = "PASSED"
        elif nay > yea:
            result = "FAILED"
        else:
            result = "TIED"

        # Historical accuracy
        historical = self._compute_historical_accuracy()

        # Final persuasion summary
        top_persuaders = sorted(
            self.persuasion_edges.items(), key=lambda x: x[1], reverse=True
        )[:3]

        # Amendments summary
        amendments_summary = [
            {"id": a.id, "text": a.text[:100], "status": a.status,
             "proposer": a.proposer, "yea": a.votes_yea, "nay": a.votes_nay}
            for a in self.amendments
        ]

        yield {
            "type": "simulation_complete",
            "yea": yea,
            "nay": nay,
            "abstain": abstain,
            "result": result,
            "amendments": amendments_summary,
            "historical_accuracy": historical,
            "top_persuaders": [
                {"influencer": inf, "influenced": infd, "strength": round(s, 3)}
                for (inf, infd), s in top_persuaders
            ],
            "final_sentiments": {
                name: round(score, 3)
                for name, score in self.agent_sentiment.items()
            },
        }

        logger.info(
            "Simulation complete: %s (yea=%d, nay=%d, abstain=%d)",
            result, yea, nay, abstain,
        )
