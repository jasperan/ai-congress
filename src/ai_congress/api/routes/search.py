"""Web search / browsing routes."""
import logging

from fastapi import APIRouter, HTTPException

from ..schemas import BrowseRequest, WebSearchRequest
from ..state import config, event_logger, web_browser, web_search_engine
from ...integrations.web_search import get_web_search_engine
from ...integrations.web_browser import get_web_browser

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["search"])


@router.post("/search/web")
async def search_web(request: WebSearchRequest):
    """Search the web"""
    global web_search_engine

    try:
        if web_search_engine is None:
            web_search_engine = get_web_search_engine(
                max_results=config.web_search.max_results,
                timeout=config.web_search.timeout,
                default_engine=config.web_search.default_engine,
                searxng_url=config.web_search.searxng_url if config.web_search.searxng_url else None,
                yacy_url=config.web_search.yacy_url if config.web_search.yacy_url else None
            )

        if request.search_type == "news":
            results = await web_search_engine.search_news(
                request.query,
                max_results=request.max_results
            )
        else:
            results = await web_search_engine.search(
                request.query,
                max_results=request.max_results
            )

        return {"success": True, "results": results, "query": request.query}

    except Exception as e:
        logger.error(f"Web search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/browse")
async def browse_url(request: BrowseRequest):
    """Fetch and parse URL content"""
    global web_browser

    try:
        if web_browser is None:
            web_browser = get_web_browser(timeout=config.web_search.timeout)

        result = await web_browser.fetch_url(
            request.url,
            extract_clean_text=request.extract_clean_text
        )

        return result

    except Exception as e:
        logger.error(f"Browse error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
