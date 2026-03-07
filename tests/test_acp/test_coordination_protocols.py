"""Tests for ACP Communication Protocols"""
import asyncio
import pytest
from src.ai_congress.core.acp.message import AgentIdentity, ACPMessage
from src.ai_congress.core.acp.message_bus import ACPMessageBus
from src.ai_congress.core.acp.coordination import CoordinationController, MessageProtocol


class TestMessageProtocol:
    def test_protocols_exist(self):
        assert MessageProtocol.FIRE_AND_FORGET == "fire_and_forget"
        assert MessageProtocol.ACK_REQUIRED == "ack_required"
        assert MessageProtocol.CONFIRMED == "confirmed"


class TestSendWithProtocol:
    def setup_method(self):
        self.bus = ACPMessageBus()
        self.controller = CoordinationController(level="chatty")
        self.sender = AgentIdentity(name="sender", role="worker")
        self.receiver = AgentIdentity(name="receiver", role="critic")
        self.bus.register_agent("sender")
        self.bus.register_agent("receiver")

    @pytest.mark.asyncio
    async def test_fire_and_forget(self):
        msg = ACPMessage(
            sender=self.sender, channel="direct",
            msg_type="opinion", payload={"text": "test"},
            recipient="receiver",
        )
        result = await self.controller.send_with_protocol(
            self.bus, msg, MessageProtocol.FIRE_AND_FORGET,
        )
        assert result is None
        received = self.bus.get_messages("receiver")
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_ack_required_timeout(self):
        msg = ACPMessage(
            sender=self.sender, channel="direct",
            msg_type="opinion", payload={"text": "test"},
            recipient="receiver",
        )
        result = await self.controller.send_with_protocol(
            self.bus, msg, MessageProtocol.ACK_REQUIRED, timeout=0.1,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_ack_required_success(self):
        msg = ACPMessage(
            sender=self.sender, channel="direct",
            msg_type="request", payload={"text": "do this"},
            recipient="receiver",
        )

        async def send_ack():
            await asyncio.sleep(0.05)
            ack = ACPMessage(
                sender=self.receiver, channel="direct",
                msg_type="ack", payload={"ack_for": msg.id},
                recipient="sender",
            )
            self.bus.send(ack)

        asyncio.create_task(send_ack())
        result = await self.controller.send_with_protocol(
            self.bus, msg, MessageProtocol.ACK_REQUIRED, timeout=1.0,
        )
        assert result is not None
        assert result.msg_type == "ack"
