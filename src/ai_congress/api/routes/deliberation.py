"""Deliberation routes: the flagship council mode over HTTP (4.2.3),
triad discovery, and debate replay exposure (4.2.2)."""
import logging
from dataclasses import fields, replace
from typing import Optional

from fastapi import APIRouter, HTTPException

from ..schemas import DeliberationRequest
from ..state import config, model_registry, swarm
from ...core.deliberation import DeliberationConfig
from ...core.triads import (
    TriadError,
    describe_triad,
    list_triads,
    resolve_triad,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["deliberation"])


async def _build_agents(triad: Optional[str], models: Optional[list]) -> list:
    """Resolve a council of agent specs from a named triad or a raw model list.

    Mirrors the CLI: a triad wins; otherwise the models are turned into
    neutral council members. Requires at least 2 agents.
    """
    if triad:
        available = None
        try:
            available = [
                m.get("name") for m in await model_registry.list_available_models()
            ] or None
        except Exception as e:
            logger.warning("Could not list models for triad fallback: %s", e)
        try:
            agents = resolve_triad(
                triad,
                fallback_model=config.agents.base_model,
                available_models=available,
            )
        except TriadError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if len(agents) < 2:
            raise HTTPException(
                status_code=400,
                detail=f"triad '{triad}' resolves to fewer than 2 agents",
            )
        return agents

    if not models or len(models) < 2:
        raise HTTPException(
            status_code=400,
            detail="deliberation needs a 'triad' or at least 2 models",
        )
    return [
        {
            "role": f"member_{i+1}",
            "name": f"member_{i+1}@{m}",
            "model": m,
            "system_prompt": (
                f"You are council member {i+1}. Offer an independent analysis "
                "of the user's question. Be concrete and flag what you're unsure about."
            ),
        }
        for i, m in enumerate(models)
    ]


def _config_from_overrides(overrides: Optional[dict]) -> Optional[DeliberationConfig]:
    """Apply per-request DeliberationConfig overrides onto the config defaults."""
    if not overrides:
        return None
    cfg_yaml = getattr(config, "deliberation", None)
    base = DeliberationConfig()
    if cfg_yaml is not None:
        try:
            base = DeliberationConfig(
                **{f.name: getattr(cfg_yaml, f.name) for f in fields(DeliberationConfig)}
            )
        except Exception:
            base = DeliberationConfig()
    allowed = {f.name for f in fields(DeliberationConfig)}
    safe = {k: v for k, v in overrides.items() if k in allowed}
    if not safe:
        return None
    try:
        return replace(base, **safe)
    except (TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"invalid deliberation config: {e}")


@router.post("/deliberation")
async def run_deliberation(request: DeliberationRequest):
    """Run the 3-round council protocol and return the full verdict (4.2.3)."""
    from ...utils.logger import info_message

    agents = await _build_agents(request.triad, request.models)
    swarm.inference_backend = request.inference_backend

    info_message(
        "DELIBERATION_REQUEST",
        f"{len(agents)} council members",
        f"triad={request.triad or '-'} evidence={request.evidence or 'default'}",
        config.logging.verbosity,
    )

    try:
        result = await swarm.deliberation_swarm(
            agents=agents,
            prompt=request.question,
            temperature=request.temperature,
            deliberation_config=_config_from_overrides(request.config),
            evidence=request.evidence,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Deliberation error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    result["agents"] = agents
    result["triad"] = request.triad
    return result


@router.get("/triads")
async def get_triads():
    """List available deliberation triads with descriptions."""
    try:
        names = list_triads()
    except TriadError as e:
        raise HTTPException(status_code=400, detail=str(e))
    out = []
    for name in names:
        try:
            info = describe_triad(name)
            out.append({"name": name, "description": info.get("description", "")})
        except TriadError:
            out.append({"name": name, "description": ""})
    return {"triads": out}


@router.get("/triads/{name}")
async def get_triad(name: str):
    """Describe one triad: members (role @ model) and description."""
    try:
        info = describe_triad(name)
    except TriadError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return info


@router.get("/replays")
async def list_replays(limit: int = 50):
    """List recent saved debate/deliberation replays (4.2.2)."""
    from ...core.observability.debate_replay import DebateReplayManager
    manager = DebateReplayManager()
    try:
        return {"replays": manager.list_debates(limit=limit)}
    except Exception as e:
        logger.warning("Replay list failed: %s", e)
        return {"replays": []}


@router.get("/replays/{session_id}")
async def get_replay(session_id: str):
    """Load one saved debate/deliberation replay (4.2.2)."""
    from ...core.observability.debate_replay import DebateReplayManager
    manager = DebateReplayManager()
    artifact = manager.load_debate(session_id)
    if not artifact:
        raise HTTPException(status_code=404, detail=f"replay not found: {session_id}")
    artifact["timeline"] = manager.format_replay_timeline(artifact)
    return artifact
