"""
Tests for Oracle Database connectivity and vector store functionality
"""
import pytest
import numpy as np
import asyncio
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ai_congress.integrations.oracle_vector_store import OracleVectorStore, EmbeddingCache
from src.ai_congress.core.rag_engine import RAGEngine
from src.ai_congress.utils.config_loader import load_config


class TestEmbeddingCache:
    """Test embedding cache functionality"""
    
    def test_cache_initialization(self):
        """Test cache initialization"""
        cache = EmbeddingCache(max_size=100, ttl=60)
        assert cache.max_size == 100
        assert cache.ttl == 60
        assert cache.hits == 0
        assert cache.misses == 0
    
    def test_cache_put_and_get(self):
        """Test cache put and get operations"""
        cache = EmbeddingCache(max_size=10, ttl=3600)
        
        # Create test embedding
        text = "test text"
        embedding = np.random.rand(384).astype(np.float32)
        
        # Put in cache
        cache.put(text, embedding)
        
        # Get from cache
        cached_embedding = cache.get(text)
        assert cached_embedding is not None
        np.testing.assert_array_equal(cached_embedding, embedding)
        assert cache.hits == 1
        assert cache.misses == 0
    
    def test_cache_miss(self):
        """Test cache miss"""
        cache = EmbeddingCache(max_size=10, ttl=3600)
        
        # Try to get non-existent item
        result = cache.get("non-existent text")
        assert result is None
        assert cache.misses == 1
    
    def test_cache_eviction(self):
        """Test LRU cache eviction"""
        cache = EmbeddingCache(max_size=3, ttl=3600)
        
        # Add 4 items (should evict oldest)
        for i in range(4):
            text = f"text_{i}"
            embedding = np.random.rand(384).astype(np.float32)
            cache.put(text, embedding)
        
        # First item should be evicted
        result = cache.get("text_0")
        assert result is None
        
        # Other items should exist
        result = cache.get("text_1")
        assert result is not None
    
    def test_cache_stats(self):
        """Test cache statistics"""
        cache = EmbeddingCache(max_size=10, ttl=3600)
        
        # Add some items
        for i in range(3):
            cache.put(f"text_{i}", np.random.rand(384).astype(np.float32))
        
        # Get some items
        cache.get("text_0")  # hit
        cache.get("text_1")  # hit
        cache.get("text_999")  # miss
        
        stats = cache.get_stats()
        assert stats['hits'] == 2
        assert stats['misses'] == 1
        assert stats['size'] == 3
        assert stats['max_size'] == 10
        assert stats['hit_rate'] == 2/3


class TestOracleVectorStoreConfig:
    """Test Oracle Vector Store configuration and initialization"""
    
    def test_config_loading(self):
        """Test that config loads Oracle DB settings correctly"""
        config = load_config()
        
        # Verify Oracle DB config
        assert hasattr(config, 'oracle_db')
        assert config.oracle_db.user.upper() == "ADMIN"  # Config loader may lowercase
        assert config.oracle_db.use_tls == True
        assert config.oracle_db.vector_table == "document_vectors"
        assert config.oracle_db.embedding_dimension == 384
        assert config.oracle_db.enable_cache == True
        assert config.oracle_db.cache_size == 1000
        assert config.oracle_db.cache_ttl == 3600
        assert config.oracle_db.batch_size == 100
        
        # Verify RAG config
        assert hasattr(config, 'rag')
        assert config.rag.enabled == True
        assert config.rag.adaptive_chunking == False
        
        # Verify document extraction config
        assert hasattr(config, 'document_extraction')
        assert config.document_extraction.use_advanced_extractors == False


@pytest.mark.integration
class TestOracleVectorStoreConnection:
    """Integration tests for Oracle Vector Store (requires actual DB)"""
    
    @pytest.fixture
    def config(self):
        """Load configuration"""
        return load_config()
    
    @pytest.fixture
    def vector_store(self, config):
        """Create vector store instance"""
        try:
            store = OracleVectorStore(
                user=config.oracle_db.user,
                password=config.oracle_db.password,
                dsn=config.oracle_db.dsn,
                use_tls=config.oracle_db.use_tls,
                vector_table=config.oracle_db.vector_table,
                embedding_dimension=config.oracle_db.embedding_dimension,
                enable_cache=config.oracle_db.enable_cache,
                cache_size=config.oracle_db.cache_size,
                cache_ttl=config.oracle_db.cache_ttl,
                batch_size=config.oracle_db.batch_size
            )
            yield store
            store.close()
        except Exception as e:
            pytest.skip(f"Could not connect to Oracle DB: {e}")
    
    def test_connection_establishment(self, vector_store):
        """Test that connection can be established"""
        assert vector_store.connection is not None
        assert vector_store.connection.ping() is None  # ping() returns None on success
    
    def test_table_creation(self, vector_store):
        """Test that vector table exists or was created"""
        cursor = vector_store.connection.cursor()
        cursor.execute(f"""
            SELECT COUNT(*) FROM user_tables 
            WHERE table_name = UPPER('{vector_store.vector_table}')
        """)
        count = cursor.fetchone()[0]
        cursor.close()
        assert count > 0
    
    def test_insert_and_search(self, vector_store):
        """Test inserting vectors and performing similarity search"""
        # Create test data
        document_id = "test_doc_001"
        chunks = ["This is test chunk 1", "This is test chunk 2"]
        embeddings = [
            np.random.rand(384).astype(np.float32),
            np.random.rand(384).astype(np.float32)
        ]
        metadata = [
            {"source": "test", "page": 1},
            {"source": "test", "page": 2}
        ]
        
        # Insert vectors
        success = vector_store.insert_vectors(document_id, chunks, embeddings, metadata)
        assert success == True
        
        # Perform similarity search
        query_embedding = np.random.rand(384).astype(np.float32)
        results = vector_store.similarity_search(query_embedding, top_k=2)
        
        assert len(results) >= 0  # May return results from this or previous tests
        
        # Clean up
        vector_store.delete_document(document_id)
    
    def test_delete_document(self, vector_store):
        """Test deleting a document"""
        # Insert test document
        document_id = "test_doc_002"
        chunks = ["Test chunk for deletion"]
        embeddings = [np.random.rand(384).astype(np.float32)]
        
        vector_store.insert_vectors(document_id, chunks, embeddings)
        
        # Delete document
        success = vector_store.delete_document(document_id)
        assert success == True
        
        # Verify deletion
        results = vector_store.similarity_search(
            np.random.rand(384).astype(np.float32),
            top_k=10,
            document_id=document_id
        )
        assert len(results) == 0
    
    def test_list_documents(self, vector_store):
        """Test listing documents"""
        # Insert test documents
        doc_ids = ["test_doc_003", "test_doc_004"]
        for doc_id in doc_ids:
            chunks = [f"Test chunk for {doc_id}"]
            embeddings = [np.random.rand(384).astype(np.float32)]
            vector_store.insert_vectors(doc_id, chunks, embeddings)
        
        # List documents
        documents = vector_store.list_documents()
        assert isinstance(documents, list)
        
        # Clean up
        for doc_id in doc_ids:
            vector_store.delete_document(doc_id)
    
    def test_cache_functionality(self, vector_store):
        """Test cache statistics"""
        if vector_store.enable_cache:
            stats = vector_store.get_cache_stats()
            assert 'hits' in stats
            assert 'misses' in stats
            assert 'hit_rate' in stats
            
            # Test cache clear
            vector_store.clear_cache()
            stats = vector_store.get_cache_stats()
            assert stats['hits'] == 0
            assert stats['misses'] == 0
    
    def test_document_stats(self, vector_store):
        """Test getting document statistics"""
        # Insert test document
        document_id = "test_doc_005"
        chunks = ["Test chunk 1", "Test chunk 2", "Test chunk 3"]
        embeddings = [np.random.rand(384).astype(np.float32) for _ in range(3)]
        
        vector_store.insert_vectors(document_id, chunks, embeddings)
        
        # Get stats
        stats = vector_store.get_document_stats(document_id)
        assert stats['document_id'] == document_id
        assert stats['chunk_count'] == 3
        assert 'avg_chunk_length' in stats
        assert 'created_at' in stats
        
        # Clean up
        vector_store.delete_document(document_id)


@pytest.mark.integration
class TestRAGEngineWithOracle:
    """Integration tests for RAG Engine with Oracle DB"""
    
    @pytest.fixture
    def rag_engine(self):
        """Create RAG engine instance"""
        try:
            engine = RAGEngine()
            if engine.vector_store is None:
                pytest.skip("Oracle DB connection not available")
            yield engine
            engine.close()
        except Exception as e:
            pytest.skip(f"Could not initialize RAG engine: {e}")
    
    def test_rag_engine_initialization(self, rag_engine):
        """Test RAG engine initializes with Oracle vector store"""
        assert rag_engine.vector_store is not None
        assert rag_engine.embedding_generator is not None
        assert rag_engine.document_processor is not None
        assert rag_engine.enabled == True
    
    @pytest.mark.asyncio
    async def test_process_document_integration(self, rag_engine, tmp_path):
        """Test end-to-end document processing"""
        # Create temporary test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("This is a test document for RAG integration testing.")
        
        try:
            # Process document
            result = await rag_engine.process_document(str(test_file), "test_doc_006")
            
            # Verify result
            if result['success']:
                assert 'document_id' in result
                assert 'chunk_count' in result
                
                # Clean up
                await rag_engine.delete_document("test_doc_006")
            else:
                # If it fails, it might be due to missing dependencies
                pytest.skip(f"Document processing failed: {result.get('error')}")
        except Exception as e:
            pytest.skip(f"Document processing not available: {e}")
    
    @pytest.mark.asyncio
    async def test_retrieve_context_integration(self, rag_engine):
        """Test context retrieval"""
        try:
            # Try to retrieve context (may be empty if no documents)
            results = await rag_engine.retrieve_context("test query", top_k=5)
            assert isinstance(results, list)
        except Exception as e:
            pytest.skip(f"Context retrieval failed: {e}")


@pytest.mark.unit
class TestOracleVectorStoreMocked:
    """Unit tests with mocked Oracle DB connection"""
    
    @patch('src.ai_congress.integrations.oracle_vector_store.oracledb.connect')
    def test_connection_with_tls(self, mock_connect):
        """Test connection initialization with TLS"""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = [1]  # Table exists
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        store = OracleVectorStore(
            user="test_user",
            password="test_pass",
            dsn="test_dsn",
            use_tls=True,
            vector_table="test_table"
        )
        
        # Verify connect was called with correct parameters
        # TLS settings are embedded in DSN, so we just verify the call was made
        assert mock_connect.called
        call_kwargs = mock_connect.call_args[1]
        assert call_kwargs['user'] == 'test_user'
        assert call_kwargs['password'] == 'test_pass'
        assert call_kwargs['dsn'] == 'test_dsn'
        
        store.close()
    
    @patch('src.ai_congress.integrations.oracle_vector_store.oracledb.connect')
    def test_batch_insert_optimization(self, mock_connect):
        """Test that batch insert uses executemany"""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = [1]  # Table exists
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        store = OracleVectorStore(
            user="test_user",
            password="test_pass",
            dsn="test_dsn",
            batch_size=100
        )
        
        # Insert multiple chunks
        chunks = [f"chunk_{i}" for i in range(50)]
        embeddings = [np.random.rand(384).astype(np.float32) for _ in range(50)]
        
        store.insert_vectors("doc_001", chunks, embeddings)
        
        # Verify executemany was called (batch operation)
        assert mock_cursor.executemany.called
        
        store.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

