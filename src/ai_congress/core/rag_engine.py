"""
RAG (Retrieval-Augmented Generation) Engine
Orchestrates document processing, embedding, storage, and retrieval
"""
import logging
from typing import List, Dict, Any, Optional
import os
from pathlib import Path
import asyncio

from ..integrations.oracle_vector_store import OracleVectorStore
from ..integrations.embeddings import EmbeddingGenerator, get_embedding_generator
from ..integrations.documents import DocumentProcessor
from ..utils.config_loader import load_config

logger = logging.getLogger(__name__)
config = load_config()


class RAGEngine:
    """Retrieval-Augmented Generation Engine"""
    
    def __init__(
        self,
        vector_store: Optional[OracleVectorStore] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        document_processor: Optional[DocumentProcessor] = None
    ):
        """
        Initialize RAG engine
        
        Args:
            vector_store: Oracle vector store instance
            embedding_generator: Embedding generator instance
            document_processor: Document processor instance
        """
        # Initialize vector store
        if vector_store is None:
            try:
                self.vector_store = OracleVectorStore(
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
                logger.info("Initialized Oracle Vector Store with intelligent caching")
            except Exception as e:
                logger.error(f"Failed to initialize Oracle Vector Store: {e}")
                self.vector_store = None
        else:
            self.vector_store = vector_store
        
        # Initialize embedding generator
        if embedding_generator is None:
            self.embedding_generator = get_embedding_generator(config.rag.embedding_model)
        else:
            self.embedding_generator = embedding_generator
        
        # Initialize document processor
        if document_processor is None:
            self.document_processor = DocumentProcessor(
                chunk_size=config.rag.chunk_size,
                chunk_overlap=config.rag.chunk_overlap,
                min_chunk_size=config.rag.min_chunk_size,
                adaptive_chunking=config.rag.adaptive_chunking,
                use_advanced_extractors=config.document_extraction.use_advanced_extractors,
                tika_url=config.document_extraction.tika_url if config.document_extraction.tika_url else None,
                docling_url=config.document_extraction.docling_url if config.document_extraction.docling_url else None
            )
        else:
            self.document_processor = document_processor
        
        self.enabled = config.rag.enabled
        logger.info(f"RAG Engine initialized (enabled: {self.enabled})")
    
    async def process_document(
        self,
        file_path: str,
        document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a document: parse, chunk, embed, and store
        
        Args:
            file_path: Path to document file
            document_id: Optional document identifier
            
        Returns:
            Dictionary with processing results
        """
        try:
            if not self.enabled or self.vector_store is None:
                return {
                    'success': False,
                    'error': 'RAG is not enabled or vector store not available'
                }
            
            # Generate document ID if not provided
            if document_id is None:
                document_id = Path(file_path).stem
            
            logger.info(f"Processing document: {file_path} (ID: {document_id})")
            
            # Parse and chunk document
            print(f"[RAG] Parsing and chunking document: {file_path}")
            full_text, chunks = await asyncio.to_thread(
                self.document_processor.process_document,
                file_path,
                document_id
            )

            print(f"[RAG] Parsed text length: {len(full_text)} chars, chunks: {len(chunks)}")
            if not chunks:
                print("[RAG] ERROR: No chunks created from document")
                return {
                    'success': False,
                    'error': 'No chunks created from document'
                }

            logger.info(f"Created {len(chunks)} chunks from {file_path}")
            print(f"[RAG] Created {len(chunks)} chunks from {file_path}")

            # Generate embeddings for chunks
            chunk_texts = [chunk.text for chunk in chunks]
            print(f"[RAG] Generating embeddings for {len(chunk_texts)} chunks (batch_size=32)")
            embeddings = await asyncio.to_thread(
                self.embedding_generator.generate_embeddings,
                chunk_texts,
                batch_size=32
            )

            logger.info(f"Generated {len(embeddings)} embeddings")
            print(f"[RAG] Generated {len(embeddings)} embeddings")

            # Prepare metadata
            chunk_metadata = [chunk.metadata for chunk in chunks]

            # Store in vector database
            print(f"[RAG] Storing {len(chunk_texts)} vectors in database")
            success = await asyncio.to_thread(
                self.vector_store.insert_vectors,
                document_id,
                chunk_texts,
                embeddings,
                chunk_metadata
            )

            print(f"[RAG] Vector storage success: {success}")
            
            if success:
                logger.info(f"Successfully processed document {document_id}: {len(chunks)} chunks")
                return {
                    'success': True,
                    'document_id': document_id,
                    'chunk_count': len(chunks),
                    'filename': Path(file_path).name
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to store vectors'
                }
                
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def retrieve_context(
        self,
        query: str,
        top_k: int = None,
        document_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a query
        
        Args:
            query: Query text
            top_k: Number of results to return (default from config)
            document_id: Optional filter by document ID
            
        Returns:
            List of relevant chunks with metadata
        """
        try:
            if not self.enabled or self.vector_store is None:
                logger.warning("RAG not enabled or vector store not available")
                return []
            
            if top_k is None:
                top_k = config.rag.top_k
            
            # Generate query embedding
            query_embedding = await asyncio.to_thread(
                self.embedding_generator.generate_embedding,
                query
            )
            
            # Perform similarity search
            results = await asyncio.to_thread(
                self.vector_store.similarity_search,
                query_embedding,
                top_k,
                document_id
            )
            
            logger.info(f"Retrieved {len(results)} relevant chunks for query")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []
    
    async def augment_query(
        self,
        query: str,
        top_k: int = None,
        document_id: Optional[str] = None,
        format_template: Optional[str] = None
    ) -> str:
        """
        Augment query with retrieved context
        
        Args:
            query: Original query
            top_k: Number of results to retrieve
            document_id: Optional filter by document ID
            format_template: Optional custom format template
            
        Returns:
            Augmented query with context
        """
        try:
            # Retrieve relevant context
            context_chunks = await self.retrieve_context(query, top_k, document_id)
            
            if not context_chunks:
                logger.info("No context retrieved, returning original query")
                return query
            
            # Format context
            context_text = "\n\n".join([
                f"[Source {i+1} - {chunk['document_id']}]:\n{chunk['content']}"
                for i, chunk in enumerate(context_chunks)
            ])
            
            # Use default template if none provided
            if format_template is None:
                format_template = """Based on the following context, please answer the question.

Context:
{context}

Question: {query}

Answer:"""
            
            # Augment query
            augmented_query = format_template.format(
                context=context_text,
                query=query
            )
            
            logger.info(f"Augmented query with {len(context_chunks)} context chunks")
            return augmented_query
            
        except Exception as e:
            logger.error(f"Error augmenting query: {e}")
            return query
    
    async def delete_document(self, document_id: str) -> bool:
        """
        Delete document from vector store
        
        Args:
            document_id: Document identifier
            
        Returns:
            True if successful
        """
        try:
            if not self.enabled or self.vector_store is None:
                return False
            
            success = await asyncio.to_thread(
                self.vector_store.delete_document,
                document_id
            )
            
            if success:
                logger.info(f"Deleted document: {document_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False
    
    async def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all documents in vector store
        
        Returns:
            List of documents with metadata
        """
        try:
            if not self.enabled or self.vector_store is None:
                return []
            
            documents = await asyncio.to_thread(
                self.vector_store.list_documents
            )
            
            return documents
            
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return []
    
    def close(self):
        """Close connections"""
        if self.vector_store:
            self.vector_store.close()


# Global singleton instance
_rag_engine = None


def get_rag_engine() -> RAGEngine:
    """
    Get or create singleton RAG engine
    
    Returns:
        RAGEngine instance
    """
    global _rag_engine
    
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    
    return _rag_engine
