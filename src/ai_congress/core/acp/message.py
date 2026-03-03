from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import uuid
import time


class AgentStatus(str, Enum):
    ACTIVE = "active"
    IDLE = "idle"
    STUCK = "stuck"
    AWAY = "away"


class ChannelType(str, Enum):
    DIRECT = "direct"
    BROADCAST = "broadcast"

    @staticmethod
    def room(name: str) -> str:
        return f"room:{name}"

    @staticmethod
    def parse_room(channel: str) -> str | None:
        return channel.split(":", 1)[1] if channel.startswith("room:") else None


@dataclass
class PersonalityProfile:
    """Big Five personality traits (1-10) + emotional state."""
    openness: int = 5
    conscientiousness: int = 5
    extraversion: int = 5
    agreeableness: int = 5
    neuroticism: int = 5
    risk_tolerance: int = 5
    empathy_level: int = 5
    leadership: int = 5
    stress: int = 0
    confidence: float = 0.5
    engagement: int = 5
    communication_style: str = "balanced"


@dataclass
class AgentIdentity:
    """Uniquely identifies an agent in the system."""
    name: str
    role: str
    status: str = "active"
    personality: PersonalityProfile | None = None
    capabilities: list[str] = field(default_factory=list)
    last_active: float = field(default_factory=time.time)


@dataclass
class ACPMessage:
    """Agent Communication Protocol message."""
    sender: AgentIdentity
    channel: str
    msg_type: str
    payload: dict[str, Any]
    recipient: str | None = None
    coordination_level: str = "moderate"
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
