"""Persistent debate artifacts for enhanced swarm runs."""

import time


class DebateArtifact:
    """Persistent, inspectable debate state for observability."""

    def __init__(self):
        self.rounds: list[dict] = []
        self.position_history: dict[str, list] = {}

    def save_round(self, round_num: int, responses: dict, clusters: list = None):
        self.rounds.append({
            "round": round_num,
            "timestamp": time.time(),
            "responses": dict(responses),
            "cluster_count": len(clusters) if clusters else 0,
        })

    def record_position(self, model_name: str, cluster_id: int):
        if model_name not in self.position_history:
            self.position_history[model_name] = []
        self.position_history[model_name].append(cluster_id)

    def get_position_evolution(self, model_name: str) -> list:
        return self.position_history.get(model_name, [])

    def to_dict(self) -> dict:
        return {
            "total_rounds": len(self.rounds),
            "rounds": self.rounds,
            "position_history": dict(self.position_history),
        }
