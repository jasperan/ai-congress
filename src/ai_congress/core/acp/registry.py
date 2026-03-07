import time
from .message import AgentIdentity, AgentStatus


class AgentRegistry:
    def __init__(self):
        self.agents: dict[str, AgentIdentity] = {}
        self._heartbeat_states: dict[str, str] = {}

    def register(self, agent: AgentIdentity) -> None:
        self.agents[agent.name] = agent

    def deregister(self, name: str) -> None:
        self.agents.pop(name, None)

    def get_active(self) -> list[AgentIdentity]:
        return [a for a in self.agents.values() if a.status == AgentStatus.ACTIVE]

    def get_by_role(self, role: str) -> list[AgentIdentity]:
        return [a for a in self.agents.values() if a.role == role]

    def get_by_capability(self, capability: str) -> list[AgentIdentity]:
        return [a for a in self.agents.values() if capability in a.capabilities]

    def update_status(self, name: str, status: str) -> None:
        if name in self.agents:
            self.agents[name].status = status

    def touch(self, name: str) -> None:
        if name in self.agents:
            self.agents[name].last_active = time.time()

    def detect_stuck(self, threshold_seconds: int = 900) -> list[str]:
        now = time.time()
        return [
            name for name, agent in self.agents.items()
            if (now - agent.last_active) > threshold_seconds
        ]

    def update_heartbeat_state(self, name: str, state: str) -> None:
        if name in self.agents:
            self._heartbeat_states[name] = state

    def get_heartbeat_state(self, name: str) -> str | None:
        return self._heartbeat_states.get(name)

    def get_agents_by_heartbeat_state(self, state: str) -> list[AgentIdentity]:
        return [
            self.agents[name]
            for name, s in self._heartbeat_states.items()
            if s == state and name in self.agents
        ]
