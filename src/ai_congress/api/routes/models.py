"""Model registry routes."""
import logging
import time
from typing import List

from fastapi import APIRouter, HTTPException

from ..schemas import ModelInfo
from ..state import config, model_registry, swarm

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["models"])

# Short TTL cache: /api/models is hit on every frontend mount and can be
# slow when Ollama has many models. Invalidate after MODELS_CACHE_TTL s.
_MODELS_CACHE_TTL = 30.0
_models_cache = None
_models_cache_at = 0.0


@router.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List all available models (Ollama + OpenAI if configured)"""
    global _models_cache, _models_cache_at

    now = time.monotonic()
    if _models_cache is not None and (now - _models_cache_at) < _MODELS_CACHE_TTL:
        return _models_cache

    models = await model_registry.list_available_models()

    result = [
        ModelInfo(
            name=m['name'],
            size=m.get('size', 0),
            weight=model_registry.get_model_weight(m['name']),
            backend="ollama",
        )
        for m in models
    ]

    # Include OpenAI model if configured
    if swarm.openai_client is not None:
        result.append(ModelInfo(
            name=config.openai.model or "openai",
            size=0,
            weight=1.0,
            backend="openai",
        ))

    _models_cache = result
    _models_cache_at = now
    return result


@router.post("/models/pull/{model_name}")
async def pull_model(model_name: str):
    """Pull a new model from Ollama"""
    success = await model_registry.pull_model(model_name)
    if success:
        return {"message": f"Model {model_name} pulled successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to pull model")
