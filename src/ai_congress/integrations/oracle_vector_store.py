"""
Oracle Vector Store Integration
Based on agentic_rag OraDBVectorStore.py
Provides vector storage and similarity search using Oracle Database 26ai
Enhanced with intelligent caching and batch operations optimization
"""
import logging
import oracledb
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from datetime import datetime
import json
import hashlib
from functools import lru_cache
from collections import OrderedDict
import time

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """LRU Cache for embeddings to reduce redundant computations"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        """
        Initialize embedding cache
        
        Args:
            max_size: Maximum number of cached embeddings
            ttl: Time to live for cache entries in seconds
        """
        self.cache = OrderedDict()
        self.max_size = max_size
        self.ttl = ttl
        self.hits = 0
        self.misses = 0
    
    def _hash_text(self, text: str) -> str:
        """Generate hash for text"""
        return hashlib.sha256(text.encode()).hexdigest()
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding if available and not expired"""
        key = self._hash_text(text)
        if key in self.cache:
            entry, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.hits += 1
                return entry
            else:
                # Expired, remove
                del self.cache[key]
        
        self.misses += 1
        return None
    
    def put(self, text: str, embedding: np.ndarray):
        """Cache embedding"""
        key = self._hash_text(text)
        
        # Remove oldest if at capacity
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        
        self.cache[key] = (embedding, time.time())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'size': len(self.cache),
            'max_size': self.max_size
        }
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0


class OracleVectorStore:
    """Oracle Database Vector Store for embeddings and similarity search with intelligent caching"""
    
    def __init__(
        self,
        user: str,
        password: str,
        dsn: str,
        use_tls: bool = True,
        vector_table: str = "document_vectors",
        embedding_dimension: int = 384,
        enable_cache: bool = True,
        cache_size: int = 1000,
        cache_ttl: int = 3600,
        batch_size: int = 100
    ):
        """
        Initialize Oracle Vector Store
        
        Args:
            user: Oracle DB username
            password: Oracle DB password
            dsn: Data Source Name (host:port/service)
            use_tls: Use TLS for connection
            vector_table: Name of the vector table
            embedding_dimension: Dimension of embedding vectors
            enable_cache: Enable intelligent caching
            cache_size: Maximum cache size
            cache_ttl: Cache time-to-live in seconds
            batch_size: Batch size for bulk operations
        """
        self.user = user
        self.password = password
        self.dsn = dsn
        self.use_tls = use_tls
        self.vector_table = vector_table
        self.embedding_dimension = embedding_dimension
        self.batch_size = batch_size
        self.connection = None
        
        # Initialize cache
        self.enable_cache = enable_cache
        if enable_cache:
            self.embedding_cache = EmbeddingCache(cache_size, cache_ttl)
            logger.info(f"Initialized embedding cache (size: {cache_size}, ttl: {cache_ttl}s)")
        else:
            self.embedding_cache = None
        
        self._init_connection()
        self._create_table_if_not_exists()
    
    def _init_connection(self):
        """Initialize database connection with TLS if enabled"""
        try:
            # The DSN connection string already contains TLS/security settings
            # For Oracle Cloud (Autonomous DB), the DSN includes: (security=(ssl_server_dn_match=yes))
            # So we can connect directly using the DSN
            self.connection = oracledb.connect(
                user=self.user,
                password=self.password,
                dsn=self.dsn
            )
            
            logger.info(f"Connected to Oracle Database {'(TLS enabled)' if self.use_tls else ''}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Oracle Database: {e}")
            raise
    
    def _create_table_if_not_exists(self):
        """Create vector table if it doesn't exist"""
        try:
            cursor = self.connection.cursor()
            
            # Check if table exists
            cursor.execute(f"""
                SELECT COUNT(*) FROM user_tables WHERE table_name = UPPER('{self.vector_table}')
            """)
            exists = cursor.fetchone()[0] > 0
            
            if not exists:
                # Create table with vector column
                create_sql = f"""
                CREATE TABLE {self.vector_table} (
                    id VARCHAR2(100) PRIMARY KEY,
                    document_id VARCHAR2(100),
                    chunk_index NUMBER,
                    content CLOB,
                    embedding VECTOR({self.embedding_dimension}, FLOAT32),
                    metadata CLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
                cursor.execute(create_sql)
                
                # Create index for vector similarity search
                cursor.execute(f"""
                    CREATE VECTOR INDEX {self.vector_table}_idx 
                    ON {self.vector_table}(embedding)
                    ORGANIZATION NEIGHBOR PARTITIONS
                    WITH DISTANCE COSINE
                """)
                
                self.connection.commit()
                logger.info(f"Created vector table: {self.vector_table}")
            else:
                logger.info(f"Vector table already exists: {self.vector_table}")
            
            cursor.close()
            
        except Exception as e:
            logger.error(f"Error creating table: {e}")
            raise
    
    def insert_vectors(
        self,
        document_id: str,
        chunks: List[str],
        embeddings: List[np.ndarray],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        Insert document chunks and their embeddings with batch optimization
        
        Args:
            document_id: Unique identifier for the document
            chunks: List of text chunks
            embeddings: List of embedding vectors
            metadata: Optional metadata for each chunk
            
        Returns:
            True if successful
        """
        try:
            cursor = self.connection.cursor()
            
            # Cache embeddings if enabled
            if self.enable_cache and self.embedding_cache:
                for chunk, embedding in zip(chunks, embeddings):
                    self.embedding_cache.put(chunk, embedding)
            
            # Prepare batch insert data
            insert_sql = f"""
                INSERT INTO {self.vector_table} 
                (id, document_id, chunk_index, content, embedding, metadata)
                VALUES (:1, :2, :3, :4, VECTOR(:5, {self.embedding_dimension}, FLOAT32), :6)
            """
            
            batch_data = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_id = f"{document_id}_chunk_{i}"
                
                # Convert numpy array to list for Oracle VECTOR type
                embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
                
                # Prepare metadata
                chunk_metadata = metadata[i] if metadata and i < len(metadata) else {}
                chunk_metadata['chunk_index'] = i
                chunk_metadata['total_chunks'] = len(chunks)
                metadata_json = json.dumps(chunk_metadata)
                
                batch_data.append([
                    chunk_id,
                    document_id,
                    i,
                    chunk,
                    embedding_list,
                    metadata_json
                ])
            
            # Execute batch insert
            cursor.executemany(insert_sql, batch_data)
            
            self.connection.commit()
            cursor.close()
            logger.info(f"Inserted {len(chunks)} vectors for document {document_id} (batch mode)")
            return True
            
        except Exception as e:
            logger.error(f"Error inserting vectors: {e}")
            self.connection.rollback()
            return False
    
    def similarity_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        document_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search using cosine distance
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            document_id: Optional filter by document ID
            
        Returns:
            List of results with content, metadata, and distance
        """
        try:
            cursor = self.connection.cursor()
            
            # Convert query embedding to list
            query_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
            
            # Build SQL query with vector similarity
            where_clause = f"WHERE document_id = '{document_id}'" if document_id else ""
            
            search_sql = f"""
                SELECT 
                    id,
                    document_id,
                    chunk_index,
                    content,
                    metadata,
                    VECTOR_DISTANCE(embedding, VECTOR(:1, {self.embedding_dimension}, FLOAT32), COSINE) as distance
                FROM {self.vector_table}
                {where_clause}
                ORDER BY distance
                FETCH FIRST :2 ROWS ONLY
            """
            
            cursor.execute(search_sql, [query_list, top_k])
            
            results = []
            for row in cursor:
                results.append({
                    'id': row[0],
                    'document_id': row[1],
                    'chunk_index': row[2],
                    'content': row[3],
                    'metadata': json.loads(row[4]) if row[4] else {},
                    'distance': float(row[5]),
                    'similarity': 1.0 - float(row[5])  # Convert distance to similarity
                })
            
            cursor.close()
            logger.info(f"Found {len(results)} similar chunks")
            return results
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {e}")
            return []
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete all vectors for a document
        
        Args:
            document_id: Document identifier
            
        Returns:
            True if successful
        """
        try:
            cursor = self.connection.cursor()
            
            delete_sql = f"""
                DELETE FROM {self.vector_table}
                WHERE document_id = :1
            """
            
            cursor.execute(delete_sql, [document_id])
            rows_deleted = cursor.rowcount
            
            self.connection.commit()
            cursor.close()
            
            logger.info(f"Deleted {rows_deleted} vectors for document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            self.connection.rollback()
            return False
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all documents in the vector store

        Returns:
            List of documents with metadata
        """
        try:
            cursor = self.connection.cursor()

            list_sql = f"""
                SELECT dv.document_id, dv.chunk_count, dv.created_at, m.metadata
                FROM (
                    SELECT document_id, COUNT(*) as chunk_count, MIN(created_at) as created_at
                    FROM {self.vector_table}
                    GROUP BY document_id
                ) dv
                LEFT JOIN (
                    SELECT document_id, metadata
                    FROM (
                        SELECT document_id, metadata,
                               ROW_NUMBER() OVER (PARTITION BY document_id ORDER BY created_at ASC) as rn
                        FROM {self.vector_table}
                    ) WHERE rn = 1
                ) m ON dv.document_id = m.document_id
                ORDER BY dv.created_at DESC
            """

            cursor.execute(list_sql)

            documents = []
            for row in cursor:
                metadata = json.loads(row[3]) if row[3] else {}
                documents.append({
                    'document_id': row[0],
                    'chunk_count': row[1],
                    'created_at': row[2].isoformat() if row[2] else None,
                    'metadata': metadata
                })

            cursor.close()
            return documents

        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return []
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Cache statistics dictionary
        """
        if self.enable_cache and self.embedding_cache:
            return self.embedding_cache.get_stats()
        return {'enabled': False}
    
    def clear_cache(self):
        """Clear embedding cache"""
        if self.enable_cache and self.embedding_cache:
            self.embedding_cache.clear()
            logger.info("Cleared embedding cache")
    
    def get_document_stats(self, document_id: str) -> Dict[str, Any]:
        """
        Get statistics for a specific document
        
        Args:
            document_id: Document identifier
            
        Returns:
            Document statistics
        """
        try:
            cursor = self.connection.cursor()
            
            stats_sql = f"""
                SELECT 
                    COUNT(*) as chunk_count,
                    AVG(LENGTH(content)) as avg_chunk_length,
                    MIN(created_at) as created_at,
                    MAX(created_at) as updated_at
                FROM {self.vector_table}
                WHERE document_id = :1
            """
            
            cursor.execute(stats_sql, [document_id])
            row = cursor.fetchone()
            cursor.close()
            
            if row:
                return {
                    'document_id': document_id,
                    'chunk_count': row[0],
                    'avg_chunk_length': float(row[1]) if row[1] else 0,
                    'created_at': row[2].isoformat() if row[2] else None,
                    'updated_at': row[3].isoformat() if row[3] else None
                }
            return {}
            
        except Exception as e:
            logger.error(f"Error getting document stats: {e}")
            return {}
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Closed Oracle Database connection")
        
        # Log cache stats before closing
        if self.enable_cache and self.embedding_cache:
            stats = self.get_cache_stats()
            logger.info(f"Final cache stats: {stats}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
