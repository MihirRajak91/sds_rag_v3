"""
Test embedding service
"""
import sys
from pathlib import Path
sys.path.append('src')

from sds_rag.services.chunking_service import ChunkingService, DocumentChunk
from sds_rag.services.embedding_service import EmbeddingService

def test_embedding():
    print("1. Testing Embedding Service...")
    
    # Create some test chunks
    test_chunks = [
        DocumentChunk(
            chunk_id="test_chunk_1",
            content="Apple Inc. reported revenue of $82.959 billion for Q3 2022.",
            chunk_type="text",
            source_document="test_doc",
            page_number=1
        ),
        DocumentChunk(
            chunk_id="test_chunk_2", 
            content="iPhone revenue was $40.665 billion in Q3 2022.",
            chunk_type="text",
            source_document="test_doc",
            page_number=2
        ),
        DocumentChunk(
            chunk_id="test_table_1",
            content="Net sales Products: $63,355 Services: $19,604",
            chunk_type="table",
            source_document="test_doc",
            table_id="table_01",
            context="Apple's financial statements"
        )
    ]
    
    print(f"2. Created {len(test_chunks)} test chunks")
    
    print("3. Initializing EmbeddingService...")
    embedder = EmbeddingService(model_name='default', batch_size=4)
    
    print(f"   Model: {embedder.model_name}")
    print(f"   Device: {embedder.device}")
    print(f"   Dimension: {embedder.embedding_dimension}")
    
    print("4. Generating embeddings...")
    embedded_chunks = embedder.embed_chunks(test_chunks)
    
    print(f"SUCCESS: Generated {len(embedded_chunks)} embeddings")
    
    # Test similarity search
    print("5. Testing similarity search...")
    query = "What was Apple's revenue?"
    similar_chunks = embedder.find_similar_chunks(query, embedded_chunks, top_k=2)
    
    print(f"Query: {query}")
    print("Similar chunks:")
    for i, chunk in enumerate(similar_chunks):
        print(f"  {i+1}. {chunk['chunk_id']}: {chunk['similarity_score']:.3f}")
        print(f"     {chunk['content'][:60]}...")
    
    # Save embeddings for next test
    output_dir = Path('test_output')
    embeddings_file = embedder.save_embeddings(embedded_chunks, output_dir)
    print(f"6. Saved embeddings to: {embeddings_file}")
    
    return embedded_chunks

if __name__ == "__main__":
    embedded_chunks = test_embedding()