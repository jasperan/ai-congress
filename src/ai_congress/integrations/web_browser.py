"""
Web Browsing Module
Fetches and parses web page content for RAG
"""
import logging
from typing import Optional, Dict, Any
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import trafilatura

logger = logging.getLogger(__name__)


class WebBrowser:
    """Fetch and parse web page content"""
    
    def __init__(self, timeout: int = 10):
        """
        Initialize web browser
        
        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        logger.info("Web browser initialized")
    
    async def fetch_url(
        self,
        url: str,
        extract_clean_text: bool = True
    ) -> Dict[str, Any]:
        """
        Fetch and parse URL content
        
        Args:
            url: URL to fetch
            extract_clean_text: Use trafilatura for clean text extraction
            
        Returns:
            Dictionary with page content and metadata
        """
        try:
            logger.info(f"Fetching URL: {url}")
            
            # Fetch page
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=self.timeout) as response:
                    if response.status != 200:
                        return {
                            'success': False,
                            'error': f'HTTP {response.status}',
                            'url': url
                        }
                    
                    html = await response.text()
                    content_type = response.headers.get('Content-Type', '')
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract title
            title = soup.title.string if soup.title else url
            
            # Extract clean text
            if extract_clean_text:
                # Use trafilatura for better text extraction
                text = await asyncio.to_thread(
                    trafilatura.extract,
                    html,
                    include_comments=False,
                    include_tables=True
                )
                
                if not text:
                    # Fallback to basic extraction
                    text = soup.get_text(separator='\n', strip=True)
            else:
                text = soup.get_text(separator='\n', strip=True)
            
            # Extract metadata
            metadata = {
                'url': url,
                'title': title,
                'content_type': content_type
            }
            
            # Extract meta description
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc and meta_desc.get('content'):
                metadata['description'] = meta_desc['content']
            
            logger.info(f"Successfully fetched and parsed: {url}")
            
            return {
                'success': True,
                'url': url,
                'title': title,
                'text': text,
                'metadata': metadata
            }
            
        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching URL: {url}")
            return {
                'success': False,
                'error': 'Timeout',
                'url': url
            }
        except Exception as e:
            logger.error(f"Error fetching URL {url}: {e}")
            return {
                'success': False,
                'error': str(e),
                'url': url
            }
    
    async def fetch_multiple_urls(
        self,
        urls: list[str],
        extract_clean_text: bool = True
    ) -> list[Dict[str, Any]]:
        """
        Fetch multiple URLs concurrently
        
        Args:
            urls: List of URLs to fetch
            extract_clean_text: Use trafilatura for clean text
            
        Returns:
            List of results
        """
        tasks = [self.fetch_url(url, extract_clean_text) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = []
        for result in results:
            if isinstance(result, dict):
                valid_results.append(result)
            else:
                logger.error(f"Exception in fetch_multiple_urls: {result}")
        
        return valid_results


# Global singleton instance
_web_browser = None


def get_web_browser(timeout: int = 10) -> WebBrowser:
    """
    Get or create singleton web browser
    
    Args:
        timeout: Request timeout
        
    Returns:
        WebBrowser instance
    """
    global _web_browser
    
    if _web_browser is None:
        _web_browser = WebBrowser(timeout)
    
    return _web_browser

