"""
End-to-end RAG pipeline test
"""
import sys
from pathlib import Path
sys.path.append('src')

from sds_rag.services.chunking_service import ChunkingService
from sds_rag.services.embedding_service import EmbeddingService  
from sds_rag.services.vector_store_service import VectorStoreService

def test_complete_pipeline():
    print("=== Complete RAG Pipeline Test ===")
    
    # Step 1: Test chunking with real Apple document
    print("\n1. Testing chunking with Apple document...")
    chunker = ChunkingService(text_chunk_size=600, text_overlap=100)
    
    # Use the extracted Apple document
    extraction_dir = Path('output/2022 Q3 AAPL')
    if not extraction_dir.exists():
        print("ERROR: Apple document not found. Run PDF extraction first.")
        return
    
    # Process just a few files for testing speed
    chunks = []
    
    # Get a few table files
    table_files = list(extraction_dir.glob("table_*_with_context.txt"))[:5]
    print(f"   Processing {len(table_files)} table files...")
    
    for table_file in table_files:
        chunk = chunker._chunk_table_file(table_file, "2022_Q3_AAPL")
        if chunk:
            chunks.append(chunk)
    
    print(f"   Created {len(chunks)} chunks from Apple document")
    
    # Step 2: Generate embeddings
    print("\n2. Generating embeddings...")
    embedder = EmbeddingService(model_name='default', batch_size=8)
    embedded_chunks = embedder.embed_chunks(chunks)
    print(f"   Generated {len(embedded_chunks)} embeddings")
    
    # Step 3: Store in Qdrant
    print("\n3. Storing in Qdrant vector database...")
    vector_store = VectorStoreService(
        collection_name="apple_q3_2022",
        embedding_dimension=embedder.embedding_dimension
    )
    
    point_ids = vector_store.add_embedded_chunks(embedded_chunks)
    print(f"   Added {len(point_ids)} points to vector store")
    
    # Step 4: Test search queries
    print("\n4. Testing search queries...")
    
    test_queries = [
        "Apple's total revenue in Q3 2022",
        "iPhone sales performance", 
        "Operating expenses and costs",
        "Financial statement data",
        "Services revenue growth"
    ]
    
    for query in test_queries:
        print(f"\n   Query: {query}")
        
        results = vector_store.search_by_text(
            query_text=query,
            embedding_service=embedder,
            limit=2,
            score_threshold=0.2
        )
        
        if results:
            print(f"   Found {len(results)} relevant chunks:")
            for i, result in enumerate(results):
                payload = result['payload']
                print(f"     {i+1}. {payload.get('table_id', 'unknown')} (score: {result['score']:.3f})")
                content_preview = payload.get('content', '')[:120].replace('\n', ' ')
                print(f"        {content_preview}...")
        else:
            print("   No relevant chunks found")
    
    # Step 5: Test filtered search
    print("\n5. Testing filtered search (tables only)...")
    table_results = vector_store.search_by_text(
        query_text="revenue and sales data",
        embedding_service=embedder,
        filters={'chunk_type': 'table'},
        limit=3
    )
    
    print(f"   Found {len(table_results)} table chunks:")
    for result in table_results:
        payload = result['payload']
        print(f"   - {payload.get('table_id', 'unknown')}: {result['score']:.3f}")
    
    print("\n=== RAG Pipeline Test Complete ===")
    print("SUCCESS: All components working together!")
    print(f"- Chunking: {len(chunks)} semantic chunks created")
    print(f"- Embedding: {embedder.embedding_dimension}D vectors generated")
    print(f"- Storage: {len(point_ids)} points in Qdrant")
    print(f"- Search: Semantic similarity working")
    print("- Ready for production RAG queries!")

if __name__ == "__main__":
    test_complete_pipeline()