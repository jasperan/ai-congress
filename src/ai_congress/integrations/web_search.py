"""
Web Search Module
Provides web search capabilities using multiple search engines
Supports: DuckDuckGo, SearXNG (self-hosted), Yacy (self-hosted)
"""
import logging
from typing import List, Dict, Any, Optional
from duckduckgo_search import DDGS
import asyncio
import aiohttp
from urllib.parse import urlencode, quote_plus

logger = logging.getLogger(__name__)


class WebSearchEngine:
    """Multi-engine web search supporting DuckDuckGo, SearXNG, and Yacy"""
    
    def __init__(
        self,
        max_results: int = 5,
        timeout: int = 10,
        default_engine: str = "duckduckgo",
        searxng_url: Optional[str] = None,
        yacy_url: Optional[str] = None
    ):
        """
        Initialize web search engine
        
        Args:
            max_results: Maximum number of results to return
            timeout: Request timeout in seconds
            default_engine: Default search engine (duckduckgo, searxng, yacy)
            searxng_url: URL for SearXNG instance (e.g., http://localhost:8080)
            yacy_url: URL for Yacy instance (e.g., http://localhost:8090)
        """
        self.max_results = max_results
        self.timeout = timeout
        self.default_engine = default_engine
        self.searxng_url = searxng_url
        self.yacy_url = yacy_url
        
        available_engines = ["duckduckgo"]
        if searxng_url:
            available_engines.append("searxng")
        if yacy_url:
            available_engines.append("yacy")
        
        logger.info(f"Web search engine initialized (Available: {', '.join(available_engines)})")
    
    async def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        region: str = "wt-wt",
        safesearch: str = "moderate",
        engine: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search the web using specified engine
        
        Args:
            query: Search query
            max_results: Override default max results
            region: Region code (wt-wt for worldwide)
            safesearch: Safe search level (on, moderate, off)
            engine: Override default search engine (duckduckgo, searxng, yacy)
            
        Returns:
            List of search results
        """
        try:
            if max_results is None:
                max_results = self.max_results
            
            if engine is None:
                engine = self.default_engine
            
            logger.info(f"Searching web for: {query} (engine: {engine})")
            
            # Route to appropriate search engine with fallback
            results = []
            if engine == "searxng" and self.searxng_url:
                results = await self._search_searxng(query, max_results, safesearch)
                if not results:
                    logger.warning("SearXNG search failed, falling back to DuckDuckGo")
                    results = await self._search_duckduckgo(query, max_results, region, safesearch)
            elif engine == "yacy" and self.yacy_url:
                results = await self._search_yacy(query, max_results)
                if not results:
                    logger.warning("Yacy search failed, falling back to DuckDuckGo")
                    results = await self._search_duckduckgo(query, max_results, region, safesearch)
            else:
                results = await self._search_duckduckgo(query, max_results, region, safesearch)
            
            logger.info(f"Found {len(results)} search results")
            return results
            
        except Exception as e:
            logger.error(f"Error performing web search: {e}")
            return []
    
    async def _search_duckduckgo(
        self,
        query: str,
        max_results: int,
        region: str = "wt-wt",
        safesearch: str = "moderate"
    ) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo"""
        def _search():
            with DDGS() as ddgs:
                results = list(ddgs.text(
                    query,
                    region=region,
                    safesearch=safesearch,
                    max_results=max_results
                ))
            return results
        
        results = await asyncio.to_thread(_search)
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                'title': result.get('title', ''),
                'url': result.get('href', ''),
                'snippet': result.get('body', ''),
                'source': 'duckduckgo'
            })
        
        return formatted_results
    
    async def _search_searxng(
        self,
        query: str,
        max_results: int,
        safesearch: str = "moderate"
    ) -> List[Dict[str, Any]]:
        """Search using SearXNG instance"""
        try:
            # Map safesearch levels
            safesearch_map = {"off": 0, "moderate": 1, "on": 2}
            safesearch_value = safesearch_map.get(safesearch, 1)
            
            params = {
                'q': query,
                'format': 'json',
                'safesearch': safesearch_value,
                'pageno': 1
            }
            
            url = f"{self.searxng_url}/search?{urlencode(params)}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=self.timeout)) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = data.get('results', [])[:max_results]
                        
                        formatted_results = []
                        for result in results:
                            formatted_results.append({
                                'title': result.get('title', ''),
                                'url': result.get('url', ''),
                                'snippet': result.get('content', ''),
                                'source': 'searxng'
                            })
                        
                        return formatted_results
                    else:
                        logger.error(f"SearXNG returned status {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Error searching with SearXNG: {e}")
            return []
    
    async def _search_yacy(
        self,
        query: str,
        max_results: int
    ) -> List[Dict[str, Any]]:
        """Search using Yacy instance"""
        try:
            params = {
                'query': query,
                'maximumRecords': max_results,
                'resource': 'global',
                'verify': 'false'
            }
            
            url = f"{self.yacy_url}/yacysearch.json?{urlencode(params)}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=self.timeout)) as response:
                    if response.status == 200:
                        data = await response.json()
                        channels = data.get('channels', [])
                        
                        formatted_results = []
                        for channel in channels:
                            items = channel.get('items', [])
                            for item in items[:max_results]:
                                formatted_results.append({
                                    'title': item.get('title', ''),
                                    'url': item.get('link', ''),
                                    'snippet': item.get('description', ''),
                                    'source': 'yacy'
                                })
                        
                        return formatted_results[:max_results]
                    else:
                        logger.error(f"Yacy returned status {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Error searching with Yacy: {e}")
            return []
    
    async def search_news(
        self,
        query: str,
        max_results: Optional[int] = None,
        region: str = "wt-wt",
        safesearch: str = "moderate"
    ) -> List[Dict[str, Any]]:
        """
        Search news
        
        Args:
            query: Search query
            max_results: Override default max results
            region: Region code
            safesearch: Safe search level
            
        Returns:
            List of news results
        """
        try:
            if max_results is None:
                max_results = self.max_results
            
            logger.info(f"Searching news for: {query}")
            
            def _search():
                with DDGS() as ddgs:
                    results = list(ddgs.news(
                        query,
                        region=region,
                        safesearch=safesearch,
                        max_results=max_results
                    ))
                return results
            
            results = await asyncio.to_thread(_search)
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'title': result.get('title', ''),
                    'url': result.get('url', ''),
                    'snippet': result.get('body', ''),
                    'date': result.get('date', ''),
                    'source': result.get('source', 'duckduckgo')
                })
            
            logger.info(f"Found {len(formatted_results)} news results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching news: {e}")
            return []
    
    def format_results_for_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Format search results as context for LLM
        
        Args:
            results: List of search results
            
        Returns:
            Formatted string for context injection
        """
        if not results:
            return ""
        
        context = "Web Search Results:\n\n"
        for i, result in enumerate(results, 1):
            context += f"{i}. {result['title']}\n"
            context += f"   URL: {result['url']}\n"
            context += f"   {result['snippet']}\n\n"
        
        return context


# Global singleton instance
_web_search_engine = None


def get_web_search_engine(
    max_results: int = 5,
    timeout: int = 10,
    default_engine: str = "duckduckgo",
    searxng_url: Optional[str] = None,
    yacy_url: Optional[str] = None
) -> WebSearchEngine:
    """
    Get or create singleton web search engine
    
    Args:
        max_results: Maximum results to return
        timeout: Request timeout
        default_engine: Default search engine
        searxng_url: URL for SearXNG instance
        yacy_url: URL for Yacy instance
        
    Returns:
        WebSearchEngine instance
    """
    global _web_search_engine
    
    if _web_search_engine is None:
        _web_search_engine = WebSearchEngine(
            max_results, 
            timeout,
            default_engine,
            searxng_url,
            yacy_url
        )
    
    return _web_search_engine
