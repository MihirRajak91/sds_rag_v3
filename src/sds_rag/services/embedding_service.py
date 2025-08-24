"""
Embedding service for converting document chunks into vector representations.
Uses sentence-transformers for high-quality semantic embeddings optimized for RAG.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np
import torch
from dataclasses import asdict
from sentence_transformers import SentenceTransformer
import json
from datetime import datetime

# Import the chunk dataclass
from .chunking_service import DocumentChunk

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Service for generating semantic embeddings from document chunks.
    Optimized for financial documents and RAG applications.
    """
    
    # Pre-configured models for different use cases
    MODEL_CONFIGS = {
        'default': {
            'name': 'all-MiniLM-L6-v2',
            'dimension': 384,
            'description': 'Fast and efficient, good for general use'
        },
        'high_quality': {
            'name': 'all-mpnet-base-v2', 
            'dimension': 768,
            'description': 'Higher quality embeddings, slower processing'
        },
        'multilingual': {
            'name': 'paraphrase-multilingual-MiniLM-L12-v2',
            'dimension': 384,
            'description': 'Supports multiple languages'
        },
        'financial': {
            'name': 'sentence-transformers/all-MiniLM-L6-v2',
            'dimension': 384,
            'description': 'Good general model for financial documents'
        }
    }
    
    def __init__(self, 
                 model_name: str = 'default',
                 device: Optional[str] = None,
                 batch_size: int = 32):
        """
        Initialize the embedding service.
        
        Args:
            model_name: Model to use ('default', 'high_quality', 'multilingual', 'financial', or custom model name)
            device: Device to use ('cpu', 'cuda', 'mps' or None for auto-detection)
            batch_size: Batch size for embedding generation
        """
        self.batch_size = batch_size
        
        # Resolve model name
        if model_name in self.MODEL_CONFIGS:
            self.model_config = self.MODEL_CONFIGS[model_name]
            self.model_name = self.model_config['name']
        else:
            self.model_name = model_name
            self.model_config = {'name': model_name, 'dimension': None, 'description': 'Custom model'}
        
        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        
        logger.info(f"Initializing embedding service with model: {self.model_name}")
        logger.info(f"Using device: {self.device}")
        
        # Load the model
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dimension}")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def embed_chunks(self, chunks: List[DocumentChunk]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for a list of document chunks.
        
        Args:
            chunks: List of DocumentChunk objects
            
        Returns:
            List of dictionaries containing chunk data and embeddings
        """
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        
        # Prepare texts for embedding
        texts = []
        chunk_data = []
        
        for chunk in chunks:
            # Create enhanced text for embedding based on chunk type
            enhanced_text = self._prepare_text_for_embedding(chunk)
            texts.append(enhanced_text)
            
            # Prepare chunk metadata
            chunk_dict = asdict(chunk)
            chunk_dict['enhanced_text'] = enhanced_text
            chunk_data.append(chunk_dict)
        
        # Generate embeddings in batches
        logger.info("Generating embeddings...")
        embeddings = self._generate_embeddings_batch(texts)
        
        # Combine chunk data with embeddings
        embedded_chunks = []
        for i, (chunk_dict, embedding) in enumerate(zip(chunk_data, embeddings)):
            embedded_chunk = {
                **chunk_dict,
                'embedding': embedding.tolist(),  # Convert numpy array to list for JSON serialization
                'embedding_model': self.model_name,
                'embedding_dimension': self.embedding_dimension,
                'embedding_created_at': datetime.now().isoformat()
            }
            embedded_chunks.append(embedded_chunk)
        
        logger.info(f"Successfully generated {len(embedded_chunks)} embeddings")
        return embedded_chunks
    
    def _prepare_text_for_embedding(self, chunk: DocumentChunk) -> str:
        """
        Prepare chunk text for optimal embedding generation based on chunk type.
        """
        if chunk.chunk_type == 'table':
            # For tables, combine context and structured data
            parts = []
            
            if chunk.context:
                parts.append(f"Context: {chunk.context}")
            
            # Add table indicator for better semantic understanding
            parts.append(f"Table from {chunk.source_document}")
            
            if chunk.page_number:
                parts.append(f"Page {chunk.page_number}")
            
            parts.append(chunk.content)
            
            return " | ".join(parts)
        
        elif chunk.chunk_type == 'text':
            # For text chunks, add document context
            context_parts = [f"Document: {chunk.source_document}"]
            
            if chunk.page_number:
                context_parts.append(f"Page {chunk.page_number}")
            
            context_prefix = " | ".join(context_parts)
            return f"{context_prefix} | {chunk.content}"
        
        elif chunk.chunk_type == 'summary':
            # For summary chunks
            return f"Summary of {chunk.source_document}: {chunk.content}"
        
        else:
            # Default: just return the content
            return chunk.content
    
    def _generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts in batches"""
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            logger.debug(f"Processing batch {i//self.batch_size + 1}/{(len(texts)-1)//self.batch_size + 1}")
            
            # Generate embeddings for this batch
            batch_embeddings = self.model.encode(
                batch_texts,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            all_embeddings.append(batch_embeddings)
        
        # Concatenate all batches
        return np.vstack(all_embeddings)
    
    def embed_single_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text (useful for query embedding).
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array containing the embedding
        """
        embedding = self.model.encode([text], convert_to_numpy=True)[0]
        return embedding
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Reshape to 2D arrays if needed
        if embedding1.ndim == 1:
            embedding1 = embedding1.reshape(1, -1)
        if embedding2.ndim == 1:
            embedding2 = embedding2.reshape(1, -1)
        
        similarity = cosine_similarity(embedding1, embedding2)[0][0]
        return float(similarity)
    
    def find_similar_chunks(self, 
                          query_text: str, 
                          embedded_chunks: List[Dict[str, Any]], 
                          top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find the most similar chunks to a query text.
        
        Args:
            query_text: Text to search for
            embedded_chunks: List of chunks with embeddings
            top_k: Number of similar chunks to return
            
        Returns:
            List of similar chunks with similarity scores
        """
        logger.info(f"Searching for similar chunks to: {query_text[:100]}...")
        
        # Generate query embedding
        query_embedding = self.embed_single_text(query_text)
        
        # Compute similarities
        similarities = []
        for chunk in embedded_chunks:
            chunk_embedding = np.array(chunk['embedding'])
            similarity = self.compute_similarity(query_embedding, chunk_embedding)
            
            similarities.append({
                **chunk,
                'similarity_score': similarity
            })
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        logger.info(f"Found {len(similarities)} chunks, returning top {top_k}")
        return similarities[:top_k]
    
    def save_embeddings(self, embedded_chunks: List[Dict[str, Any]], output_dir: Path) -> Path:
        """Save embedded chunks to JSON file"""
        output_dir.mkdir(parents=True, exist_ok=True)
        embeddings_file = output_dir / "embeddings.json"
        
        # Save with metadata
        output_data = {
            'metadata': {
                'model_name': self.model_name,
                'embedding_dimension': self.embedding_dimension,
                'device_used': self.device,
                'total_chunks': len(embedded_chunks),
                'created_at': datetime.now().isoformat()
            },
            'embeddings': embedded_chunks
        }
        
        with open(embeddings_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(embedded_chunks)} embeddings to {embeddings_file}")
        return embeddings_file
    
    def load_embeddings(self, embeddings_file: Path) -> List[Dict[str, Any]]:
        """Load embedded chunks from JSON file"""
        with open(embeddings_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        embedded_chunks = data['embeddings']
        metadata = data.get('metadata', {})
        
        logger.info(f"Loaded {len(embedded_chunks)} embeddings from {embeddings_file}")
        logger.info(f"Model used: {metadata.get('model_name', 'unknown')}")
        
        return embedded_chunks
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            'model_name': self.model_name,
            'model_config': self.model_config,
            'embedding_dimension': self.embedding_dimension,
            'device': self.device,
            'batch_size': self.batch_size
        }

# Example usage
if __name__ == "__main__":
    # Test embedding service with chunked data
    embedding_service = EmbeddingService(model_name='default', batch_size=16)
    
    # Load chunks from chunking service
    from chunking_service import ChunkingService
    
    chunks_file = Path("chunked_documents/chunks.json")
    
    if chunks_file.exists():
        chunking_service = ChunkingService()
        chunks = chunking_service.load_chunks(chunks_file)
        
        # Generate embeddings
        embedded_chunks = embedding_service.embed_chunks(chunks)
        
        # Save embeddings
        output_dir = Path("embedded_documents")
        embeddings_file = embedding_service.save_embeddings(embedded_chunks, output_dir)
        
        print(f"✅ Generated embeddings for {len(embedded_chunks)} chunks")
        print(f"📁 Saved to: {embeddings_file}")
        print(f"🤖 Model: {embedding_service.model_name}")
        print(f"📏 Embedding dimension: {embedding_service.embedding_dimension}")
        
        # Test similarity search
        query = "What was Apple's revenue in Q3 2022?"
        similar_chunks = embedding_service.find_similar_chunks(query, embedded_chunks, top_k=3)
        
        print(f"\n🔍 Similar chunks for query: '{query}'")
        for i, chunk in enumerate(similar_chunks):
            print(f"\n{i+1}. {chunk['chunk_id']} (similarity: {chunk['similarity_score']:.3f})")
            print(f"   Type: {chunk['chunk_type']}")
            print(f"   Preview: {chunk['content'][:200]}...")
    else:
        print("❌ No chunked documents found. Run chunking service first.")