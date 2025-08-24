"""
Vector store service for managing document embeddings in Qdrant.
Provides high-performance similarity search and document retrieval for RAG systems.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np
from datetime import datetime
import json
import uuid
from dataclasses import asdict

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance, VectorParams, CreateCollection, 
    PointStruct, Filter, FieldCondition, Range, MatchValue
)

# Import related services
from services.chunking_service import DocumentChunk
from services.embedding_service import EmbeddingService

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class VectorStoreService:
    """
    Service for managing document embeddings in Qdrant vector database.
    Optimized for RAG applications with rich metadata filtering.
    """
    
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 6333,
                 collection_name: str = "financial_documents",
                 embedding_dimension: int = 384):
        """
        Initialize the vector store service.
        
        Args:
            host: Qdrant server host
            port: Qdrant server port
            collection_name: Name of the collection to use
            embedding_dimension: Dimension of embedding vectors
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        
        logger.info(f"Initializing Qdrant client: {host}:{port}")
        
        # Initialize Qdrant client
        try:
            self.client = QdrantClient(host=host, port=port)
            logger.info("Successfully connected to Qdrant")
        except Exception as e:
            logger.warning(f"Failed to connect to Qdrant at {host}:{port}: {e}")
            logger.info("Using in-memory Qdrant client")
            self.client = QdrantClient(":memory:")
        
        # Ensure collection exists
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self):
        """Create the collection if it doesn't exist"""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            existing_names = [collection.name for collection in collections]
            
            if self.collection_name not in existing_names:
                logger.info(f"Creating collection: {self.collection_name}")
                
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dimension,
                        distance=Distance.COSINE
                    )
                )
                
                # Create indexes for common filter fields
                self._create_payload_indexes()
                
                logger.info(f"Collection '{self.collection_name}' created successfully")
            else:
                logger.info(f"Collection '{self.collection_name}' already exists")
                
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
            raise
    
    def _create_payload_indexes(self):
        """Create indexes on payload fields for efficient filtering"""
        index_fields = [
            ("source_document", models.PayloadSchemaType.KEYWORD),
            ("chunk_type", models.PayloadSchemaType.KEYWORD),
            ("page_number", models.PayloadSchemaType.INTEGER),
            ("table_id", models.PayloadSchemaType.KEYWORD),
            ("created_at", models.PayloadSchemaType.DATETIME)
        ]
        
        for field_name, field_type in index_fields:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_type
                )
                logger.debug(f"Created index for field: {field_name}")
            except Exception as e:
                logger.debug(f"Index for {field_name} may already exist: {e}")
    
    def add_embedded_chunks(self, embedded_chunks: List[Dict[str, Any]]) -> List[str]:
        """
        Add embedded chunks to the vector store.
        
        Args:
            embedded_chunks: List of chunks with embeddings
            
        Returns:
            List of point IDs that were added
        """
        logger.info(f"Adding {len(embedded_chunks)} chunks to vector store")
        
        points = []
        point_ids = []
        
        for chunk in embedded_chunks:
            # Generate unique point ID
            point_id = str(uuid.uuid4())
            point_ids.append(point_id)
            
            # Extract embedding
            embedding = chunk.get('embedding')
            if not embedding:
                logger.error(f"No embedding found for chunk: {chunk.get('chunk_id', 'unknown')}")
                continue
            
            # Prepare payload (metadata)
            payload = self._prepare_payload(chunk)
            
            # Create point
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload
            )
            points.append(point)
        
        # Upload points to Qdrant
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Successfully added {len(points)} points to collection")
            return point_ids
        except Exception as e:
            logger.error(f"Error adding points to vector store: {e}")
            raise
    
    def _prepare_payload(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare chunk data as Qdrant payload"""
        # Core chunk information
        payload = {
            'chunk_id': chunk.get('chunk_id'),
            'content': chunk.get('content'),
            'chunk_type': chunk.get('chunk_type'),
            'source_document': chunk.get('source_document'),
            'page_number': chunk.get('page_number'),
            'table_id': chunk.get('table_id'),
            'context': chunk.get('context'),
            'created_at': chunk.get('created_at'),
            'embedding_model': chunk.get('embedding_model'),
            'embedding_dimension': chunk.get('embedding_dimension')
        }
        
        # Add metadata
        if 'metadata' in chunk and chunk['metadata']:
            payload['metadata'] = chunk['metadata']
        
        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}
        
        return payload
    
    def search_similar(self, 
                      query_embedding: Union[List[float], np.ndarray],
                      limit: int = 10,
                      score_threshold: float = 0.0,
                      filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents using vector similarity.
        
        Args:
            query_embedding: Query vector
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            filters: Optional filters for metadata
            
        Returns:
            List of similar documents with scores
        """
        # Convert numpy array to list if needed
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
        
        # Prepare filter conditions
        filter_conditions = None
        if filters:
            filter_conditions = self._build_filter_conditions(filters)
        
        try:
            # Perform search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=filter_conditions,
                limit=limit,
                score_threshold=score_threshold
            )
            
            # Format results
            results = []
            for scored_point in search_result:
                result = {
                    'id': scored_point.id,
                    'score': scored_point.score,
                    'payload': scored_point.payload
                }
                results.append(result)
            
            logger.info(f"Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            raise
    
    def _build_filter_conditions(self, filters: Dict[str, Any]) -> Filter:
        """Build Qdrant filter conditions from filter dictionary"""
        conditions = []
        
        for field, value in filters.items():
            if isinstance(value, str):
                conditions.append(FieldCondition(key=field, match=MatchValue(value=value)))
            elif isinstance(value, int):
                conditions.append(FieldCondition(key=field, match=MatchValue(value=value)))
            elif isinstance(value, list):
                # For list values, create OR conditions
                list_conditions = [FieldCondition(key=field, match=MatchValue(value=v)) for v in value]
                conditions.extend(list_conditions)
            elif isinstance(value, dict) and 'range' in value:
                # Range filter
                range_filter = value['range']
                conditions.append(FieldCondition(
                    key=field,
                    range=Range(
                        gte=range_filter.get('gte'),
                        lte=range_filter.get('lte')
                    )
                ))
        
        return Filter(must=conditions) if conditions else None
    
    def search_by_text(self, 
                      query_text: str,
                      embedding_service: EmbeddingService,
                      limit: int = 10,
                      score_threshold: float = 0.0,
                      filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search using text query (requires embedding service).
        
        Args:
            query_text: Text to search for
            embedding_service: Service to generate query embedding
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            filters: Optional metadata filters
            
        Returns:
            List of similar documents
        """
        logger.info(f"Searching for: {query_text}")
        
        # Generate query embedding
        query_embedding = embedding_service.embed_single_text(query_text)
        
        # Search using the embedding
        return self.search_similar(
            query_embedding=query_embedding,
            limit=limit,
            score_threshold=score_threshold,
            filters=filters
        )
    
    def get_document_chunks(self, source_document: str) -> List[Dict[str, Any]]:
        """
        Retrieve all chunks for a specific document.
        
        Args:
            source_document: Name of the source document
            
        Returns:
            List of document chunks
        """
        filters = {'source_document': source_document}
        
        # Use scroll to get all points (not limited by search limit)
        try:
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=self._build_filter_conditions(filters),
                limit=1000,  # Adjust based on expected document size
                with_payload=True,
                with_vectors=False
            )
            
            results = []
            for point in points:
                result = {
                    'id': point.id,
                    'payload': point.payload
                }
                results.append(result)
            
            logger.info(f"Retrieved {len(results)} chunks for document: {source_document}")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving document chunks: {e}")
            raise
    
    def delete_document(self, source_document: str) -> int:
        """
        Delete all chunks for a specific document.
        
        Args:
            source_document: Name of the source document
            
        Returns:
            Number of points deleted
        """
        filters = {'source_document': source_document}
        filter_conditions = self._build_filter_conditions(filters)
        
        try:
            operation_info = self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(filter=filter_conditions)
            )
            
            logger.info(f"Deleted chunks for document: {source_document}")
            return operation_info.operation_id
            
        except Exception as e:
            logger.error(f"Error deleting document chunks: {e}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            return {
                'name': collection_info.config.params.name,
                'vector_size': collection_info.config.params.vectors.size,
                'distance': collection_info.config.params.vectors.distance,
                'points_count': collection_info.points_count,
                'segments_count': collection_info.segments_count,
                'status': collection_info.status
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}
    
    def create_backup(self, output_dir: Path) -> Path:
        """
        Create a backup of all vectors and payloads.
        
        Args:
            output_dir: Directory to save the backup
            
        Returns:
            Path to the backup file
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        backup_file = output_dir / f"vector_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            # Scroll through all points
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,  # Adjust as needed
                with_payload=True,
                with_vectors=True
            )
            
            # Prepare backup data
            backup_data = {
                'metadata': {
                    'collection_name': self.collection_name,
                    'backup_created_at': datetime.now().isoformat(),
                    'points_count': len(points)
                },
                'points': []
            }
            
            for point in points:
                point_data = {
                    'id': str(point.id),
                    'vector': point.vector,
                    'payload': point.payload
                }
                backup_data['points'].append(point_data)
            
            # Save backup
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Created backup with {len(points)} points: {backup_file}")
            return backup_file
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            raise

# Example usage
if __name__ == "__main__":
    # Test vector store service
    vector_store = VectorStoreService(
        collection_name="test_financial_docs",
        embedding_dimension=384
    )
    
    # Load embedded chunks
    embeddings_file = Path("embedded_documents/embeddings.json")
    
    if embeddings_file.exists():
        from embedding_service import EmbeddingService
        
        embedding_service = EmbeddingService()
        embedded_chunks = embedding_service.load_embeddings(embeddings_file)
        
        # Add to vector store
        point_ids = vector_store.add_embedded_chunks(embedded_chunks['embeddings'])
        
        print(f"✅ Added {len(point_ids)} chunks to vector store")
        
        # Test search
        query = "What was Apple's revenue in Q3 2022?"
        results = vector_store.search_by_text(
            query_text=query,
            embedding_service=embedding_service,
            limit=5,
            score_threshold=0.1
        )
        
        print(f"\n🔍 Search results for: '{query}'")
        for i, result in enumerate(results):
            print(f"\n{i+1}. Score: {result['score']:.3f}")
            print(f"   Chunk: {result['payload']['chunk_id']}")
            print(f"   Type: {result['payload']['chunk_type']}")
            print(f"   Content: {result['payload']['content'][:200]}...")
        
        # Show collection info
        info = vector_store.get_collection_info()
        print(f"\n📊 Collection Info:")
        print(f"   Name: {info.get('name')}")
        print(f"   Points: {info.get('points_count')}")
        print(f"   Vector size: {info.get('vector_size')}")
    else:
        print("❌ No embedded documents found. Run embedding service first.")