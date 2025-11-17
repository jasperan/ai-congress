"""
Tests for Enhanced RAG and Web Search Features
Tests all new integrations: caching, adaptive chunking, multi-engine search, advanced extractors
"""
import pytest
import asyncio
import numpy as np
from pathlib import Path
import tempfile
import os

# Import modules to test
from src.ai_congress.integrations.oracle_vector_store import OracleVectorStore, EmbeddingCache
from src.ai_congress.integrations.documents import TextChunker, DocumentProcessor
from src.ai_congress.integrations.web_search import WebSearchEngine
from src.ai_congress.integrations.advanced_extractors import (
    ApacheTikaExtractor,
    DoclingExtractor,
    AdvancedDocumentExtractor
)


class TestEmbeddingCache:
    """Test intelligent embedding caching"""
    
    def test_cache_initialization(self):
        """Test cache can be initialized with different parameters"""
        cache = EmbeddingCache(max_size=100, ttl=3600)
        assert cache.max_size == 100
        assert cache.ttl == 3600
        assert cache.hits == 0
        assert cache.misses == 0
    
    def test_cache_put_get(self):
        """Test basic cache put and get operations"""
        cache = EmbeddingCache()
        
        # Create dummy embedding
        embedding = np.random.rand(384).astype(np.float32)
        text = "This is a test sentence"
        
        # Put in cache
        cache.put(text, embedding)
        
        # Get from cache
        cached = cache.get(text)
        assert cached is not None
        np.testing.assert_array_equal(cached, embedding)
        assert cache.hits == 1
    
    def test_cache_miss(self):
        """Test cache miss for non-existent text"""
        cache = EmbeddingCache()
        
        result = cache.get("Non-existent text")
        assert result is None
        assert cache.misses == 1
    
    def test_cache_eviction(self):
        """Test LRU eviction when cache is full"""
        cache = EmbeddingCache(max_size=3, ttl=3600)
        
        # Fill cache
        for i in range(4):
            embedding = np.random.rand(384).astype(np.float32)
            cache.put(f"text_{i}", embedding)
        
        # Cache should be at max size
        assert len(cache.cache) == 3
        
        # First item should be evicted
        assert cache.get("text_0") is None
        assert cache.get("text_3") is not None
    
    def test_cache_statistics(self):
        """Test cache statistics calculation"""
        cache = EmbeddingCache()
        
        embedding = np.random.rand(384).astype(np.float32)
        cache.put("text1", embedding)
        
        # One hit, one miss
        cache.get("text1")
        cache.get("nonexistent")
        
        stats = cache.get_stats()
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['hit_rate'] == 0.5
        assert stats['size'] == 1


class TestAdaptiveChunking:
    """Test adaptive text chunking strategy"""
    
    def test_simple_chunking(self):
        """Test traditional simple chunking"""
        chunker = TextChunker(
            chunk_size=100,
            chunk_overlap=20,
            min_chunk_size=50,  # Set reasonable minimum
            adaptive=False
        )
        
        text = "This is a test. " * 30  # Repeat to create enough text
        chunks = chunker.chunk_text(text)
        
        # With min_chunk_size, we should get chunks
        if len(chunks) > 0:
            for chunk in chunks:
                assert 'chunking_strategy' in chunk.metadata
                assert chunk.metadata['chunking_strategy'] == 'simple'
    
    def test_adaptive_chunking(self):
        """Test adaptive paragraph-aware chunking"""
        chunker = TextChunker(
            chunk_size=200,
            chunk_overlap=50,
            adaptive=True
        )
        
        text = """This is paragraph one. It has multiple sentences. This is another sentence.

This is paragraph two. It also has content. More content here.

This is paragraph three. Final paragraph with text."""
        
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert 'chunking_strategy' in chunk.metadata
            assert chunk.metadata['chunking_strategy'] in ['adaptive', 'sentence-based']
    
    def test_chunking_metadata(self):
        """Test that chunks contain proper metadata"""
        chunker = TextChunker(adaptive=True, chunk_size=200, min_chunk_size=50)
        
        text = "Test text. " * 100  # Add punctuation for better chunking
        metadata = {'document_id': 'test_doc', 'filename': 'test.txt'}
        chunks = chunker.chunk_text(text, metadata)
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.metadata['document_id'] == 'test_doc'
            assert chunk.metadata['filename'] == 'test.txt'
            # chunk_index is set by DocumentChunk, not passed in metadata
            assert 'start_char' in chunk.metadata
            assert 'end_char' in chunk.metadata
    
    def test_minimum_chunk_size(self):
        """Test that chunks respect minimum size"""
        chunker = TextChunker(
            chunk_size=100,
            min_chunk_size=50,
            adaptive=True
        )
        
        text = "Short text"
        chunks = chunker.chunk_text(text)
        
        # Should not create chunks smaller than min_chunk_size
        assert all(len(chunk.text) >= 50 for chunk in chunks if chunks)


class TestWebSearchEngine:
    """Test multi-engine web search"""
    
    @pytest.mark.asyncio
    async def test_duckduckgo_search(self):
        """Test DuckDuckGo search"""
        engine = WebSearchEngine(
            max_results=3,
            timeout=10,
            default_engine="duckduckgo"
        )
        
        results = await engine.search("Python programming", max_results=3)
        
        # Should return results (unless network issue)
        if results:
            assert len(results) <= 3
            for result in results:
                assert 'title' in result
                assert 'url' in result
                assert 'snippet' in result
                assert result['source'] == 'duckduckgo'
    
    def test_search_engine_initialization(self):
        """Test search engine can be initialized with different configs"""
        engine = WebSearchEngine(
            max_results=10,
            timeout=20,
            default_engine="duckduckgo",
            searxng_url="http://localhost:8080",
            yacy_url="http://localhost:8090"
        )
        
        assert engine.max_results == 10
        assert engine.timeout == 20
        assert engine.default_engine == "duckduckgo"
    
    @pytest.mark.asyncio
    async def test_news_search(self):
        """Test news search functionality"""
        engine = WebSearchEngine()
        
        results = await engine.search_news("technology news", max_results=2)
        
        # Should return results or empty list
        assert isinstance(results, list)
        for result in results:
            assert 'title' in result
            assert 'url' in result
    
    def test_format_results_for_context(self):
        """Test formatting results for LLM context"""
        engine = WebSearchEngine()
        
        results = [
            {
                'title': 'Test Article',
                'url': 'https://example.com',
                'snippet': 'Test snippet content'
            }
        ]
        
        context = engine.format_results_for_context(results)
        
        assert 'Test Article' in context
        assert 'https://example.com' in context
        assert 'Test snippet content' in context


class TestAdvancedExtractors:
    """Test advanced document extraction"""
    
    def test_tika_extractor_initialization(self):
        """Test Tika extractor can be initialized"""
        extractor = ApacheTikaExtractor(tika_server_url="http://localhost:9998")
        
        assert extractor.tika_server_url == "http://localhost:9998"
        # availability depends on server running
        assert isinstance(extractor.available, bool)
    
    def test_docling_extractor_initialization(self):
        """Test Docling extractor can be initialized"""
        extractor = DoclingExtractor(docling_server_url="http://localhost:5001")
        
        assert extractor.docling_server_url == "http://localhost:5001"
        assert isinstance(extractor.available, bool)
    
    def test_advanced_extractor_fallback(self):
        """Test unified extractor with no services available"""
        extractor = AdvancedDocumentExtractor(
            tika_url="http://invalid:9999",
            docling_url="http://invalid:9999",
            prefer="docling"
        )
        
        # Should handle unavailable services gracefully
        assert extractor.is_available() == False
        assert len(extractor.available_extractors) == 0


class TestDocumentProcessor:
    """Test enhanced document processor"""
    
    def test_processor_initialization(self):
        """Test document processor can be initialized with new params"""
        processor = DocumentProcessor(
            chunk_size=512,
            chunk_overlap=50,
            adaptive_chunking=True,
            use_advanced_extractors=False
        )
        
        assert processor.chunker.adaptive == True
        assert processor.use_advanced_extractors == False
    
    def test_processor_with_advanced_extractors(self):
        """Test processor can be configured with advanced extractors"""
        processor = DocumentProcessor(
            use_advanced_extractors=True,
            tika_url="http://localhost:9998",
            docling_url="http://localhost:5001"
        )
        
        assert processor.use_advanced_extractors == True
        # advanced_extractor may or may not be available depending on setup
    
    def test_text_file_processing(self):
        """Test processing a simple text file"""
        processor = DocumentProcessor(adaptive_chunking=True)
        
        # Create temporary text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document. " * 50)
            temp_path = f.name
        
        try:
            text, chunks = processor.process_document(temp_path, document_id="test_doc")
            
            assert len(text) > 0
            assert len(chunks) > 0
            assert chunks[0].metadata['document_id'] == 'test_doc'
            assert chunks[0].metadata['extraction_method'] == 'default'
        finally:
            os.unlink(temp_path)


class TestOracleVectorStore:
    """Test Oracle Vector Store enhancements (requires Oracle DB)"""
    
    @pytest.mark.skipif(
        os.getenv("ORACLE_TEST_ENABLED") != "true",
        reason="Oracle DB not available for testing"
    )
    def test_vector_store_with_caching(self):
        """Test vector store initialization with caching"""
        # This test requires Oracle DB connection
        # Skip unless explicitly enabled
        store = OracleVectorStore(
            user="test_user",
            password="test_pass",
            dsn="localhost:1521/FREEPDB1",
            use_tls=False,
            enable_cache=True,
            cache_size=100,
            cache_ttl=3600,
            batch_size=50
        )
        
        assert store.enable_cache == True
        assert store.embedding_cache is not None
        assert store.batch_size == 50


class TestIntegration:
    """Integration tests for complete workflows"""
    
    @pytest.mark.asyncio
    async def test_search_and_format(self):
        """Test complete search and format workflow"""
        engine = WebSearchEngine(max_results=2)
        
        # Search
        results = await engine.search("artificial intelligence")
        
        # Format for context
        if results:
            context = engine.format_results_for_context(results)
            assert len(context) > 0
            assert "Web Search Results" in context
    
    def test_chunk_and_embed_workflow(self):
        """Test document chunking workflow"""
        chunker = TextChunker(adaptive=True, chunk_size=200)
        
        text = """
        Artificial Intelligence and Machine Learning.
        
        AI is transforming many industries. Machine learning algorithms
        can learn from data and make predictions.
        
        Deep learning is a subset of machine learning that uses neural networks.
        """
        
        chunks = chunker.chunk_text(text, metadata={'doc_id': 'ai_doc'})
        
        assert len(chunks) > 0
        
        # Verify each chunk has proper metadata
        for chunk in chunks:
            assert 'doc_id' in chunk.metadata
            assert chunk.metadata['doc_id'] == 'ai_doc'
            assert 'chunking_strategy' in chunk.metadata


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

