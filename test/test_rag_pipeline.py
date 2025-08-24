"""
Test script for the complete RAG pipeline.
Tests chunking, embedding, and vector store services with extracted PDF data.
"""

import sys
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from sds_rag.services.chunking_service import ChunkingService
from sds_rag.services.embedding_service import EmbeddingService
from sds_rag.services.vector_store_service import VectorStoreService

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

def test_complete_pipeline():
    """Test the complete RAG pipeline from extraction to search"""
    
    print("🚀 Testing Complete RAG Pipeline")
    print("=" * 50)
    
    # Step 1: Test Chunking Service
    print("\n📝 Step 1: Testing Chunking Service")
    print("-" * 30)
    
    try:
        chunking_service = ChunkingService(text_chunk_size=800, text_overlap=150)
        
        # Use existing extracted document
        extraction_dir = Path("src/sds_rag/services/output/2022 Q3 AAPL")
        
        if not extraction_dir.exists():
            print("❌ No extracted documents found. Please run PDF extraction first.")
            return
        
        # Create chunks
        chunks = chunking_service.chunk_extracted_document(extraction_dir)
        print(f"✅ Created {len(chunks)} chunks")
        
        # Save chunks for testing
        output_dir = Path("test_output")
        chunks_file = chunking_service.save_chunks(chunks, output_dir)
        print(f"📁 Saved chunks to: {chunks_file}")
        
        # Show sample chunk
        if chunks:
            sample_chunk = chunks[0]
            print(f"\n📄 Sample chunk ({sample_chunk.chunk_type}):")
            print(f"   ID: {sample_chunk.chunk_id}")
            print(f"   Content: {sample_chunk.content[:150]}...")
        
    except Exception as e:
        print(f"❌ Chunking failed: {e}")
        return
    
    # Step 2: Test Embedding Service
    print("\n🧠 Step 2: Testing Embedding Service")
    print("-" * 30)
    
    try:
        embedding_service = EmbeddingService(model_name='default', batch_size=8)
        
        # Generate embeddings
        embedded_chunks = embedding_service.embed_chunks(chunks[:10])  # Test with first 10 chunks
        print(f"✅ Generated embeddings for {len(embedded_chunks)} chunks")
        print(f"🤖 Model: {embedding_service.model_name}")
        print(f"📏 Embedding dimension: {embedding_service.embedding_dimension}")
        
        # Save embeddings
        embeddings_file = embedding_service.save_embeddings(embedded_chunks, output_dir)
        print(f"📁 Saved embeddings to: {embeddings_file}")
        
        # Test similarity search
        query = "What was Apple's revenue?"
        similar_chunks = embedding_service.find_similar_chunks(query, embedded_chunks, top_k=3)
        
        print(f"\n🔍 Top 3 similar chunks for: '{query}'")
        for i, chunk in enumerate(similar_chunks):
            print(f"   {i+1}. {chunk['chunk_id']} (score: {chunk['similarity_score']:.3f})")
            print(f"      Type: {chunk['chunk_type']}")
        
    except Exception as e:
        print(f"❌ Embedding failed: {e}")
        return
    
    # Step 3: Test Vector Store Service
    print("\n🗄️ Step 3: Testing Vector Store Service")
    print("-" * 30)
    
    try:
        vector_store = VectorStoreService(
            collection_name="test_rag_pipeline",
            embedding_dimension=embedding_service.embedding_dimension
        )
        
        # Add embedded chunks to vector store
        point_ids = vector_store.add_embedded_chunks(embedded_chunks)
        print(f"✅ Added {len(point_ids)} points to vector store")
        
        # Get collection info
        info = vector_store.get_collection_info()
        print(f"📊 Collection: {info.get('name', 'unknown')}")
        print(f"📊 Points: {info.get('points_count', 0)}")
        print(f"📊 Vector size: {info.get('vector_size', 'unknown')}")
        
        # Test vector search
        search_results = vector_store.search_by_text(
            query_text="Apple revenue Q3 2022",
            embedding_service=embedding_service,
            limit=5,
            score_threshold=0.1
        )
        
        print(f"\n🔍 Vector search results:")
        for i, result in enumerate(search_results):
            payload = result['payload']
            print(f"   {i+1}. Score: {result['score']:.3f}")
            print(f"      Chunk: {payload.get('chunk_id', 'unknown')}")
            print(f"      Type: {payload.get('chunk_type', 'unknown')}")
            print(f"      Document: {payload.get('source_document', 'unknown')}")
        
        # Test filtered search
        table_results = vector_store.search_by_text(
            query_text="financial data tables",
            embedding_service=embedding_service,
            filters={'chunk_type': 'table'},
            limit=3
        )
        
        print(f"\n📊 Table-only search results:")
        for i, result in enumerate(table_results):
            payload = result['payload']
            print(f"   {i+1}. Score: {result['score']:.3f}")
            print(f"      Table: {payload.get('table_id', 'unknown')}")
            print(f"      Page: {payload.get('page_number', 'unknown')}")
        
    except Exception as e:
        print(f"❌ Vector store failed: {e}")
        print(f"   Note: Make sure Qdrant is running or using in-memory mode")
        return
    
    # Step 4: Test End-to-End RAG Query
    print("\n🎯 Step 4: End-to-End RAG Query Test")
    print("-" * 30)
    
    try:
        test_queries = [
            "What was Apple's total revenue in Q3 2022?",
            "Show me iPhone revenue data",
            "What are Apple's operating expenses?",
            "Financial performance summary"
        ]
        
        for query in test_queries:
            print(f"\n❓ Query: {query}")
            
            # Search for relevant chunks
            results = vector_store.search_by_text(
                query_text=query,
                embedding_service=embedding_service,
                limit=3,
                score_threshold=0.2
            )
            
            if results:
                print(f"   📋 Found {len(results)} relevant chunks:")
                for i, result in enumerate(results[:2]):  # Show top 2
                    payload = result['payload']
                    content_preview = payload.get('content', '')[:100] + "..."
                    print(f"     {i+1}. {payload.get('chunk_id', 'unknown')} (score: {result['score']:.3f})")
                    print(f"        {content_preview}")
            else:
                print("   ⚠️ No relevant chunks found")
        
    except Exception as e:
        print(f"❌ End-to-end test failed: {e}")
        return
    
    print("\n🎉 RAG Pipeline Test Complete!")
    print("=" * 50)
    print("✅ All services working correctly")
    print(f"📁 Test outputs saved to: {output_dir}")
    print("\n🚀 Ready for production RAG queries!")

def test_individual_services():
    """Test each service individually for debugging"""
    
    print("\n🔧 Individual Service Tests")
    print("-" * 30)
    
    # Test chunking service
    try:
        chunking_service = ChunkingService()
        print("✅ ChunkingService: OK")
    except Exception as e:
        print(f"❌ ChunkingService: {e}")
    
    # Test embedding service
    try:
        embedding_service = EmbeddingService()
        test_text = "This is a test sentence."
        embedding = embedding_service.embed_single_text(test_text)
        print(f"✅ EmbeddingService: OK (dimension: {len(embedding)})")
    except Exception as e:
        print(f"❌ EmbeddingService: {e}")
    
    # Test vector store service
    try:
        vector_store = VectorStoreService(collection_name="test_individual")
        info = vector_store.get_collection_info()
        print(f"✅ VectorStoreService: OK (collection: {info.get('name', 'unknown')})")
    except Exception as e:
        print(f"❌ VectorStoreService: {e}")

if __name__ == "__main__":
    print("🧪 RAG Pipeline Test Suite")
    print("=" * 50)
    
    # Test individual services first
    test_individual_services()
    
    # Test complete pipeline
    test_complete_pipeline()