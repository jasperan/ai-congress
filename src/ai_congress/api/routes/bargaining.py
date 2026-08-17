"""Bargaining-consensus route (3.2.7): mediator-guided negotiation for
high-stakes questions. Opt-in per request — more deliberative than a vote."""
import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional

from ..state import swarm, security_ctx
from ...core.negotiation import BargainingSession
from ...utils.evals import _AdaptedClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["bargaining"])


class BargainingRequest(BaseModel):
    question: str
    models: List[str] = Field(default_factory=list)
    max_rounds: int = 3
    priority_weights: Optional[Dict[str, float]] = None
    inference_backend: str = "ollama"


@router.post("/bargaining", dependencies=[Depends(security_ctx.require_api_key)])
async def bargain(request: BargainingRequest):
    """Run a mediated bargaining negotiation over the question."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question is required")
    models = request.models or None
    try:
        available = await swarm.model_registry.list_available_models()
        live = [m["name"] for m in available]
        requested = models or [m["name"] for m in available][:3]
        models = [m for m in requested if m in live] or requested
        agents = [{"name": f"member_{i}", "model": m} for i, m in enumerate(models, 1)]

        # Pick the client matching the requested backend (pi/openai when asked)
        client = swarm.ollama_client
        if request.inference_backend == "pi" and swarm.pi_client is not None:
            client = swarm.pi_client
        elif request.inference_backend == "openai" and swarm.openai_client is not None:
            client = swarm.openai_client

        session = BargainingSession(
            client=_AdaptedClient(client),
            priority_weights=request.priority_weights or {},
            max_rounds=max(1, min(5, request.max_rounds)),
        )
        result = await session.negotiate(agents, request.question)
        result["models"] = models
        result["inference_backend"] = request.inference_backend
        return result
    except Exception as e:
        logger.error("Bargaining error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))