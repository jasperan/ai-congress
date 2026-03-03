"""
Tests for ACP Message Bus
"""
import pytest
from src.ai_congress.core.acp.message import AgentIdentity, ACPMessage
from src.ai_congress.core.acp.message_bus import ACPMessageBus


class TestACPMessageBus:
    def setup_method(self):
        self.bus = ACPMessageBus()
        self.agent1 = AgentIdentity(name="agent1", role="analyst")
        self.agent2 = AgentIdentity(name="agent2", role="reviewer")
        self.agent3 = AgentIdentity(name="agent3", role="coordinator")
        self.bus.register_agent("agent1")
        self.bus.register_agent("agent2")
        self.bus.register_agent("agent3")

    def test_direct_message(self):
        msg = ACPMessage(
            sender=self.agent1,
            channel="direct",
            msg_type="opinion",
            payload={"text": "hello"},
            recipient="agent2",
        )
        self.bus.send(msg)
        messages = self.bus.get_messages("agent2")
        assert len(messages) == 1
        assert messages[0].payload["text"] == "hello"
        assert messages[0].sender.name == "agent1"

    def test_direct_message_not_received_by_others(self):
        msg = ACPMessage(
            sender=self.agent1,
            channel="direct",
            msg_type="opinion",
            payload={"text": "private"},
            recipient="agent2",
        )
        self.bus.send(msg)
        # agent3 should not receive it
        messages = self.bus.get_messages("agent3")
        assert len(messages) == 0
        # agent1 (sender) should not receive it either
        messages = self.bus.get_messages("agent1")
        assert len(messages) == 0

    def test_broadcast(self):
        msg = ACPMessage(
            sender=self.agent1,
            channel="broadcast",
            msg_type="announcement",
            payload={"text": "attention everyone"},
        )
        self.bus.send(msg)
        # agent2 and agent3 should receive it
        msg2 = self.bus.get_messages("agent2")
        msg3 = self.bus.get_messages("agent3")
        assert len(msg2) == 1
        assert len(msg3) == 1
        # sender should NOT receive their own broadcast
        msg1 = self.bus.get_messages("agent1")
        assert len(msg1) == 0

    def test_room_message(self):
        self.bus.join_room("agent1", "debate")
        self.bus.join_room("agent2", "debate")
        # agent3 is NOT in the room
        msg = ACPMessage(
            sender=self.agent1,
            channel="room:debate",
            msg_type="discussion",
            payload={"text": "room talk"},
        )
        self.bus.send(msg)
        # agent2 is in the room and should receive it
        messages = self.bus.get_messages("agent2")
        assert len(messages) == 1
        assert messages[0].payload["text"] == "room talk"
        # agent3 is not in the room
        messages = self.bus.get_messages("agent3")
        assert len(messages) == 0
        # sender should not receive their own room message
        messages = self.bus.get_messages("agent1")
        assert len(messages) == 0

    def test_get_messages_clears_queue(self):
        msg = ACPMessage(
            sender=self.agent1,
            channel="direct",
            msg_type="opinion",
            payload={"text": "hello"},
            recipient="agent2",
        )
        self.bus.send(msg)
        first_read = self.bus.get_messages("agent2")
        assert len(first_read) == 1
        # Second read should return empty (queue was cleared)
        second_read = self.bus.get_messages("agent2")
        assert len(second_read) == 0

    def test_message_history(self):
        for i in range(5):
            msg = ACPMessage(
                sender=self.agent1,
                channel="broadcast",
                msg_type="update",
                payload={"index": i},
            )
            self.bus.send(msg)
        history = self.bus.get_history(limit=3)
        assert len(history) == 3
        # Should be the last 3 messages
        assert history[0].payload["index"] == 2
        assert history[1].payload["index"] == 3
        assert history[2].payload["index"] == 4

    def test_leave_room(self):
        self.bus.join_room("agent1", "debate")
        self.bus.join_room("agent2", "debate")
        self.bus.leave_room("agent2", "debate")

        msg = ACPMessage(
            sender=self.agent1,
            channel="room:debate",
            msg_type="discussion",
            payload={"text": "after leave"},
        )
        self.bus.send(msg)
        # agent2 left the room, should not receive messages
        messages = self.bus.get_messages("agent2")
        assert len(messages) == 0

    def test_deregister_agent(self):
        self.bus.join_room("agent2", "debate")
        self.bus.deregister_agent("agent2")
        # Broadcast should not reach deregistered agent
        msg = ACPMessage(
            sender=self.agent1,
            channel="broadcast",
            msg_type="announcement",
            payload={"text": "test"},
        )
        self.bus.send(msg)
        messages = self.bus.get_messages("agent2")
        assert len(messages) == 0
