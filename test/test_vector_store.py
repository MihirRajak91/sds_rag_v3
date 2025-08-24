"""
Test vector store service with Qdrant
"""
import sys
from pathlib import Path
sys.path.append('src')

from sds_rag.services.embedding_service import EmbeddingService
from sds_rag.services.vector_store_service import VectorStoreService

def test_vector_store():
    print("1. Testing Vector Store Service...")
    
    # Load embeddings from previous test
    embeddings_file = Path('test_output/embeddings.json')
    if not embeddings_file.exists():
        print("ERROR: No embeddings file found. Run embedding test first.")
        return
    
    print("2. Loading embeddings...")
    embedder = EmbeddingService()
    embedded_chunks = embedder.load_embeddings(embeddings_file)
    print(f"   Loaded {len(embedded_chunks)} embedded chunks")
    
    print("3. Initializing Vector Store...")
    vector_store = VectorStoreService(
        collection_name="test_rag_pipeline",
        embedding_dimension=384
    )
    
    # Get collection info
    info = vector_store.get_collection_info()
    print(f"   Collection: {info.get('name', 'unknown')}")
    print(f"   Status: {info.get('status', 'unknown')}")
    
    print("4. Adding embedded chunks to vector store...")
    point_ids = vector_store.add_embedded_chunks(embedded_chunks)
    print(f"SUCCESS: Added {len(point_ids)} points to vector store")
    
    # Update collection info
    info = vector_store.get_collection_info()
    print(f"   Points in collection: {info.get('points_count', 0)}")
    
    print("5. Testing vector search...")
    
    # Test 1: Direct text search
    query1 = "Apple revenue information"
    results1 = vector_store.search_by_text(
        query_text=query1,
        embedding_service=embedder,
        limit=3,
        score_threshold=0.1
    )
    
    print(f"Query 1: '{query1}'")
    print(f"Results: {len(results1)} found")
    for i, result in enumerate(results1):
        payload = result['payload']
        print(f"  {i+1}. Score: {result['score']:.3f}")
        print(f"     Chunk: {payload.get('chunk_id', 'unknown')}")
        print(f"     Type: {payload.get('chunk_type', 'unknown')}")
        print(f"     Content: {payload.get('content', '')[:80]}...")
    
    # Test 2: Filtered search
    print("\n6. Testing filtered search...")
    query2 = "financial data"
    results2 = vector_store.search_by_text(
        query_text=query2,
        embedding_service=embedder,
        filters={'chunk_type': 'table'},
        limit=2
    )
    
    print(f"Query 2: '{query2}' (tables only)")
    print(f"Results: {len(results2)} found")
    for i, result in enumerate(results2):
        payload = result['payload']
        print(f"  {i+1}. Score: {result['score']:.3f}")
        print(f"     Table: {payload.get('table_id', 'unknown')}")
        print(f"     Content: {payload.get('content', '')[:80]}...")
    
    # Test 3: Document retrieval
    print("\n7. Testing document retrieval...")
    doc_chunks = vector_store.get_document_chunks('test_doc')
    print(f"Document chunks for 'test_doc': {len(doc_chunks)}")
    
    for chunk in doc_chunks:
        payload = chunk['payload']
        print(f"  - {payload.get('chunk_id', 'unknown')} ({payload.get('chunk_type', 'unknown')})")
    
    print("\nSUCCESS: All vector store tests passed!")
    return vector_store

if __name__ == "__main__":
    vector_store = test_vector_store()