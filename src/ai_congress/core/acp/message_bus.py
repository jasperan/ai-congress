from collections import defaultdict
from .message import ACPMessage, ChannelType


class ACPMessageBus:
    MAX_HISTORY = 1000

    def __init__(self):
        self._queues: dict[str, list[ACPMessage]] = defaultdict(list)
        self._rooms: dict[str, set[str]] = defaultdict(set)
        self._agents: set[str] = set()
        self._history: list[ACPMessage] = []

    def register_agent(self, name: str) -> None:
        self._agents.add(name)

    def deregister_agent(self, name: str) -> None:
        self._agents.discard(name)
        self._queues.pop(name, None)
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

    def get_messages(self, agent_name: str, clear: bool = True) -> list[ACPMessage]:
        messages = list(self._queues.get(agent_name, []))
        if clear:
            self._queues[agent_name] = []
        return messages

    def get_history(self, limit: int = 100) -> list[ACPMessage]:
        return self._history[-limit:]
