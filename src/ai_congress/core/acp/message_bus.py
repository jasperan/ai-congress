from collections import defaultdict
from copy import copy

from .message import ACPMessage, ChannelType


class ACPMessageBus:
    MAX_HISTORY = 1000

    def __init__(self, audit_trail=None, org_chart=None):
        self._queues: dict[str, list[ACPMessage]] = defaultdict(list)
        self._rooms: dict[str, set[str]] = defaultdict(set)
        self._agents: set[str] = set()
        self._history: list[ACPMessage] = []
        self._audit_trail = audit_trail
        self._org_chart = org_chart
        self._conversation_state: dict[str, list[ACPMessage]] = defaultdict(list)

    def register_agent(self, name: str) -> None:
        self._agents.add(name)

    def deregister_agent(self, name: str) -> None:
        self._agents.discard(name)
        self._queues.pop(name, None)
        self._conversation_state.pop(name, None)
        for room_members in self._rooms.values():
            room_members.discard(name)

    def join_room(self, agent_name: str, room: str) -> None:
        self._rooms[room].add(agent_name)

    def leave_room(self, agent_name: str, room: str) -> None:
        if room in self._rooms:
            self._rooms[room].discard(agent_name)

    def send(self, message: ACPMessage) -> None:
        self._history.append(message)
        if len(self._history) > self.MAX_HISTORY:
            self._history = self._history[-self.MAX_HISTORY:]

        sender = message.sender.name
        self._conversation_state[sender].append(message)

        if message.channel == ChannelType.DIRECT and message.recipient:
            if message.recipient in self._agents:
                self._queues[message.recipient].append(message)
        elif message.channel == ChannelType.BROADCAST:
            for agent in self._agents:
                if agent != sender:
                    self._queues[agent].append(message)
        elif (room_name := ChannelType.parse_room(message.channel)):
            for agent in self._rooms.get(room_name, set()):
                if agent != sender:
                    self._queues[agent].append(message)

        # Persist to audit trail
        if self._audit_trail:
            from .audit_trail import AuditEvent, AuditEventType
            self._audit_trail.record(AuditEvent(
                event_type=AuditEventType.MESSAGE_SENT,
                agent_name=sender,
                payload={
                    "channel": message.channel,
                    "msg_type": message.msg_type,
                    "recipient": message.recipient,
                },
            ))

    def get_messages(self, agent_name: str, clear: bool = True) -> list[ACPMessage]:
        messages = list(self._queues.get(agent_name, []))
        if clear:
            self._queues[agent_name] = []
        return messages

    def get_history(self, limit: int = 100) -> list[ACPMessage]:
        return self._history[-limit:]

    def get_context(self, agent_name: str, limit: int = 20) -> list[ACPMessage]:
        msgs = self._conversation_state.get(agent_name, [])
        return msgs[-limit:]

    def send_to_supervisor(self, message: ACPMessage) -> None:
        if not self._org_chart:
            return
        chain = self._org_chart.get_chain_of_command(message.sender.name)
        if len(chain) >= 2:
            supervisor = chain[1]
            msg_copy = copy(message)
            msg_copy.channel = ChannelType.DIRECT
            msg_copy.recipient = supervisor.agent_name
            self.send(msg_copy)

    def send_to_reports(self, message: ACPMessage) -> None:
        if not self._org_chart:
            return
        reports = self._org_chart.get_direct_reports(message.sender.name)
        for report in reports:
            msg_copy = copy(message)
            msg_copy.channel = ChannelType.DIRECT
            msg_copy.recipient = report.agent_name
            self.send(msg_copy)

    def escalate(self, message: ACPMessage, required_authority: str = "") -> bool:
        if not self._org_chart:
            return False
        chain = self._org_chart.get_chain_of_command(message.sender.name)
        for assignment in chain:
            if assignment.agent_name == message.sender.name:
                continue
            if self._org_chart.can_perform(assignment.agent_name, required_authority):
                msg_copy = copy(message)
                msg_copy.channel = ChannelType.DIRECT
                msg_copy.recipient = assignment.agent_name
                self.send(msg_copy)
                return True
        return False
