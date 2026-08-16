"""Precedent (stare decisis) routes."""
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from ..schemas import PrecedentSearchRequest
from ..state import get_enhanced_orchestrator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["precedents"])


@router.get("/precedents")
async def list_precedents(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    domain: Optional[str] = None,
):
    """List stored precedent rulings."""
    try:
        orch = get_enhanced_orchestrator()
        if orch.precedent_store is None:
            return {"precedents": [], "message": "Precedent store not available"}
        precedents = await orch.precedent_store.list_precedents(
            limit=limit, offset=offset, domain=domain,
        )
        return {"precedents": [p.to_dict() for p in precedents]}
    except Exception as e:
        logger.error(f"List precedents error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/precedents/search")
async def search_precedents(request: PrecedentSearchRequest):
    """Search for similar precedent rulings via vector similarity."""
    try:
        orch = get_enhanced_orchestrator()
        if orch.precedent_store is None:
            return {"precedents": [], "message": "Precedent store not available"}
        precedents = await orch.precedent_store.search_precedents(
            query_text=request.query,
            domain=request.domain,
            top_k=request.top_k,
            min_similarity=request.min_similarity,
        )
        return {"precedents": [p.to_dict() for p in precedents], "query": request.query}
    except Exception as e:
        logger.error(f"Search precedents error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/precedents/{precedent_id}")
async def get_precedent(precedent_id: str):
    """Get a specific precedent by ID, including supersession chain."""
    try:
        orch = get_enhanced_orchestrator()
        if orch.precedent_store is None:
            raise HTTPException(status_code=503, detail="Precedent store not available")
        precedent = await orch.precedent_store.get_precedent(precedent_id)
        if precedent is None:
            raise HTTPException(status_code=404, detail="Precedent not found")
        return precedent.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get precedent error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
