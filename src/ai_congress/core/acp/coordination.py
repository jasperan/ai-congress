import asyncio
import random
from collections import defaultdict
from enum import Enum

from .message import AgentIdentity


class MessageProtocol(str, Enum):
    FIRE_AND_FORGET = "fire_and_forget"
    ACK_REQUIRED = "ack_required"
    CONFIRMED = "confirmed"


class CoordinationController:
    LEVELS = {
        "none": {"probability": 0.0, "max_messages": 0},
        "minimal": {"probability": 0.3, "max_messages": 2},
        "moderate": {"probability": 0.6, "max_messages": 5},
        "chatty": {"probability": 0.9, "max_messages": 10},
    }

    def __init__(self, level: str = "moderate"):
        if level not in self.LEVELS:
            raise ValueError(f"Invalid coordination level: {level}. Must be one of {list(self.LEVELS)}")
        self.level = level
        self._message_counts: dict[tuple[str, str], int] = defaultdict(int)

    def should_communicate(self, agent: AgentIdentity, context: dict) -> bool:
        base_prob = self.LEVELS[self.level]["probability"]
        if base_prob == 0.0:
            return False
        if agent.personality:
            modifier = agent.personality.extraversion / 5.0
            base_prob = min(1.0, base_prob * modifier)
        return random.random() < base_prob

    def can_send_more(self, agent_name: str, round_id: str) -> bool:
        sent = self._message_counts[(agent_name, round_id)]
        return sent < self.LEVELS[self.level]["max_messages"]

    def record_message(self, agent_name: str, round_id: str) -> None:
        self._message_counts[(agent_name, round_id)] += 1

    def get_max_messages(self) -> int:
        return self.LEVELS[self.level]["max_messages"]

    def reset_round(self, round_id: str) -> None:
        keys_to_remove = [k for k in self._message_counts if k[1] == round_id]
        for k in keys_to_remove:
            del self._message_counts[k]

    async def send_with_protocol(
        self,
        bus,
        message,
        protocol: MessageProtocol = MessageProtocol.FIRE_AND_FORGET,
        timeout: float = 30.0,
    ):
        bus.send(message)

        if protocol == MessageProtocol.FIRE_AND_FORGET:
            return None

        if protocol in (MessageProtocol.ACK_REQUIRED, MessageProtocol.CONFIRMED):
            expected_type = "ack" if protocol == MessageProtocol.ACK_REQUIRED else "confirm"
            return await self._wait_for_response(
                bus, message.sender.name, message.id, expected_type, timeout,
            )

        return None

    async def _wait_for_response(self, bus, sender_name, message_id, expected_type, timeout):
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            messages = bus.get_messages(sender_name, clear=True)
            for msg in messages:
                if msg.msg_type == expected_type and msg.payload.get("ack_for") == message_id:
                    return msg
            await asyncio.sleep(0.02)
        return None
