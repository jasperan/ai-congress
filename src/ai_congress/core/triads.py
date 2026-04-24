"""
Triad loader - reads config/triads.json and resolves archetype roles to
full agent specs for the DeliberationOrchestrator.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_TRIADS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "config", "triads.json"
)


class TriadError(ValueError):
    pass


def _load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def load_triads(path: Optional[str] = None) -> dict:
    """Load and return the triads config."""
    target = path or DEFAULT_TRIADS_PATH
    if not os.path.exists(target):
        raise TriadError(f"triads config not found: {target}")
    return _load_json(target)


def list_triads(path: Optional[str] = None) -> List[str]:
    """Return the sorted list of triad names."""
    data = load_triads(path)
    return sorted(list(data.get("triads", {}).keys()))


def resolve_triad(
    name: str,
    path: Optional[str] = None,
    fallback_model: Optional[str] = None,
    available_models: Optional[List[str]] = None,
) -> List[Dict]:
    """Resolve a triad name into a list of 3 agent specs.

    Each spec has keys: role, name, model, system_prompt. If the hinted model
    is not in available_models (when provided), fall back to fallback_model.
    """
    data = load_triads(path)
    triads = data.get("triads", {})
    if name not in triads:
        raise TriadError(
            f"unknown triad '{name}'. available: {', '.join(sorted(triads.keys())) or '(none)'}"
        )
    archetypes = data.get("archetypes", {})
    agents_cfg = triads[name].get("agents", [])
    resolved: List[Dict] = []
    for idx, agent in enumerate(agents_cfg):
        role = agent.get("role", f"member_{idx}")
        system_prompt = agent.get("system_prompt") or archetypes.get(role)
        if not system_prompt:
            raise TriadError(
                f"triad '{name}' agent {idx} role='{role}' has no system_prompt "
                "and no matching archetype"
            )
        model = agent.get("model") or fallback_model
        if available_models and model and model not in available_models:
            logger.info(
                f"triad '{name}': model '{model}' not available, "
                f"falling back to '{fallback_model}'"
            )
            model = fallback_model
        if not model:
            raise TriadError(
                f"triad '{name}' agent {idx}: no model resolvable and no fallback"
            )
        resolved.append({
            "role": role,
            "name": f"{role}@{model}",
            "model": model,
            "system_prompt": system_prompt,
            "description": triads[name].get("description"),
        })
    return resolved


def describe_triad(name: str, path: Optional[str] = None) -> Dict:
    """Return the triad metadata without resolving models."""
    data = load_triads(path)
    triads = data.get("triads", {})
    if name not in triads:
        raise TriadError(f"unknown triad '{name}'")
    entry = triads[name]
    return {
        "name": name,
        "description": entry.get("description"),
        "roles": [a.get("role") for a in entry.get("agents", [])],
        "models": [a.get("model") for a in entry.get("agents", [])],
    }
