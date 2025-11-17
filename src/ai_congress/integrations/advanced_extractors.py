"""
Advanced Document Extraction Module
Provides Apache Tika and Docling integration for enhanced document parsing
These are optional extractors that can be used when available
"""
import logging
from typing import Optional, Dict, Any
import os
import requests
import json

logger = logging.getLogger(__name__)


class ApacheTikaExtractor:
    """
    Apache Tika document extraction
    Requires Tika server running (e.g., via Docker)
    
    Docker command:
    docker run -d -p 9998:9998 apache/tika:latest
    """
    
    def __init__(self, tika_server_url: str = "http://localhost:9998"):
        """
        Initialize Apache Tika extractor
        
        Args:
            tika_server_url: URL of Tika server
        """
        self.tika_server_url = tika_server_url
        self.available = self._check_availability()
        
        if self.available:
            logger.info(f"Apache Tika extractor initialized at {tika_server_url}")
        else:
            logger.warning(f"Apache Tika server not available at {tika_server_url}")
    
    def _check_availability(self) -> bool:
        """Check if Tika server is available"""
        try:
            response = requests.get(
                f"{self.tika_server_url}/tika",
                timeout=2
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def extract_text(self, file_path: str) -> Optional[str]:
        """
        Extract text from document using Tika
        
        Args:
            file_path: Path to document file
            
        Returns:
            Extracted text or None if extraction failed
        """
        if not self.available:
            logger.warning("Tika server not available, cannot extract text")
            return None
        
        try:
            with open(file_path, 'rb') as file:
                files = {'file': file}
                response = requests.put(
                    f"{self.tika_server_url}/tika",
                    files=files,
                    headers={'Accept': 'text/plain'},
                    timeout=30
                )
                
                if response.status_code == 200:
                    text = response.text
                    logger.info(f"Extracted {len(text)} characters from {file_path} using Tika")
                    return text
                else:
                    logger.error(f"Tika extraction failed with status {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error extracting text with Tika: {e}")
            return None
    
    def extract_metadata(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Extract metadata from document using Tika
        
        Args:
            file_path: Path to document file
            
        Returns:
            Metadata dictionary or None if extraction failed
        """
        if not self.available:
            return None
        
        try:
            with open(file_path, 'rb') as file:
                files = {'file': file}
                response = requests.put(
                    f"{self.tika_server_url}/meta",
                    files=files,
                    headers={'Accept': 'application/json'},
                    timeout=30
                )
                
                if response.status_code == 200:
                    metadata = response.json()
                    logger.info(f"Extracted metadata from {file_path} using Tika")
                    return metadata
                else:
                    logger.error(f"Tika metadata extraction failed with status {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error extracting metadata with Tika: {e}")
            return None


class DoclingExtractor:
    """
    Docling document extraction
    Requires Docling server running (e.g., via Docker)
    
    Docker command:
    docker run -p 5001:5001 -e DOCLING_SERVE_ENABLE_UI=true quay.io/docling-project/docling-serve
    
    With GPU:
    docker run --gpus all -p 5001:5001 -e DOCLING_SERVE_ENABLE_UI=true quay.io/docling-project/docling-serve-cu124
    """
    
    def __init__(self, docling_server_url: str = "http://localhost:5001"):
        """
        Initialize Docling extractor
        
        Args:
            docling_server_url: URL of Docling server
        """
        self.docling_server_url = docling_server_url
        self.available = self._check_availability()
        
        if self.available:
            logger.info(f"Docling extractor initialized at {docling_server_url}")
        else:
            logger.warning(f"Docling server not available at {docling_server_url}")
    
    def _check_availability(self) -> bool:
        """Check if Docling server is available"""
        try:
            response = requests.get(
                f"{self.docling_server_url}/health",
                timeout=2
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def extract_text(
        self,
        file_path: str,
        output_format: str = "markdown"
    ) -> Optional[str]:
        """
        Extract text from document using Docling
        
        Args:
            file_path: Path to document file
            output_format: Output format (markdown, json, text)
            
        Returns:
            Extracted text or None if extraction failed
        """
        if not self.available:
            logger.warning("Docling server not available, cannot extract text")
            return None
        
        try:
            with open(file_path, 'rb') as file:
                files = {'file': (os.path.basename(file_path), file)}
                data = {'output_format': output_format}
                
                response = requests.post(
                    f"{self.docling_server_url}/convert",
                    files=files,
                    data=data,
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Extract text based on format
                    if output_format == "markdown":
                        text = result.get('markdown', '')
                    elif output_format == "text":
                        text = result.get('text', '')
                    else:
                        text = json.dumps(result, indent=2)
                    
                    logger.info(f"Extracted {len(text)} characters from {file_path} using Docling")
                    return text
                else:
                    logger.error(f"Docling extraction failed with status {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error extracting text with Docling: {e}")
            return None
    
    def extract_with_structure(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Extract document with structure preservation
        
        Args:
            file_path: Path to document file
            
        Returns:
            Structured document data or None if extraction failed
        """
        if not self.available:
            return None
        
        try:
            with open(file_path, 'rb') as file:
                files = {'file': (os.path.basename(file_path), file)}
                data = {'output_format': 'json', 'preserve_structure': True}
                
                response = requests.post(
                    f"{self.docling_server_url}/convert",
                    files=files,
                    data=data,
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"Extracted structured data from {file_path} using Docling")
                    return result
                else:
                    logger.error(f"Docling structured extraction failed with status {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error extracting structured data with Docling: {e}")
            return None


class AdvancedDocumentExtractor:
    """
    Unified interface for advanced document extraction
    Falls back to available extractors
    """
    
    def __init__(
        self,
        tika_url: Optional[str] = None,
        docling_url: Optional[str] = None,
        prefer: str = "docling"
    ):
        """
        Initialize advanced document extractor
        
        Args:
            tika_url: Apache Tika server URL
            docling_url: Docling server URL
            prefer: Preferred extractor (docling or tika)
        """
        self.prefer = prefer
        
        # Initialize extractors
        self.tika = None
        if tika_url:
            self.tika = ApacheTikaExtractor(tika_url)
        
        self.docling = None
        if docling_url:
            self.docling = DoclingExtractor(docling_url)
        
        # Determine available extractors
        self.available_extractors = []
        if self.docling and self.docling.available:
            self.available_extractors.append("docling")
        if self.tika and self.tika.available:
            self.available_extractors.append("tika")
        
        if self.available_extractors:
            logger.info(f"Advanced extractors available: {', '.join(self.available_extractors)}")
        else:
            logger.info("No advanced extractors available, will use default parsers")
    
    def extract_text(self, file_path: str) -> Optional[str]:
        """
        Extract text using available extractors
        
        Args:
            file_path: Path to document file
            
        Returns:
            Extracted text or None
        """
        if not self.available_extractors:
            return None
        
        # Try preferred extractor first
        if self.prefer == "docling" and self.docling and self.docling.available:
            text = self.docling.extract_text(file_path)
            if text:
                return text
        elif self.prefer == "tika" and self.tika and self.tika.available:
            text = self.tika.extract_text(file_path)
            if text:
                return text
        
        # Fallback to other extractor
        if self.docling and self.docling.available:
            text = self.docling.extract_text(file_path)
            if text:
                return text
        
        if self.tika and self.tika.available:
            text = self.tika.extract_text(file_path)
            if text:
                return text
        
        return None
    
    def is_available(self) -> bool:
        """Check if any advanced extractors are available"""
        return len(self.available_extractors) > 0


# Global singleton instance
_advanced_extractor = None


def get_advanced_extractor(
    tika_url: Optional[str] = None,
    docling_url: Optional[str] = None,
    prefer: str = "docling"
) -> AdvancedDocumentExtractor:
    """
    Get or create singleton advanced extractor
    
    Args:
        tika_url: Apache Tika server URL
        docling_url: Docling server URL
        prefer: Preferred extractor
        
    Returns:
        AdvancedDocumentExtractor instance
    """
    global _advanced_extractor
    
    if _advanced_extractor is None:
        _advanced_extractor = AdvancedDocumentExtractor(
            tika_url,
            docling_url,
            prefer
        )
    
    return _advanced_extractor

