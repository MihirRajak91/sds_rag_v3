"""
Test RAG Chatbot functionality
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('src')

def test_chatbot():
    print("Testing RAG Chatbot system...")
    print("=" * 50)
    
    try:
        # Import required services
        from sds_rag.services.rag_chatbot_service import RAGChatbotService
        from sds_rag.services.vector_store_service import VectorStoreService
        from sds_rag.services.embedding_service import EmbeddingService
        from sds_rag.config.constants import DatabaseConfig, EmbeddingConfig, LLMConfig
        
        print("SUCCESS: All imports successful")
        
        # Check API key
        if not LLMConfig.API_KEY or LLMConfig.API_KEY == "your_gemini_api_key_here":
            print("ERROR: Please set your GOOGLE_API_KEY in the .env file")
            print("   Get your key from: https://aistudio.google.com/app/apikey")
            return
        
        print(f"SUCCESS: Gemini API key configured")
        
        # Initialize embedding service
        print("\n1. Initializing embedding service...")
        embedding_service = EmbeddingService(
            model_name=EmbeddingConfig.MODEL,
            device=EmbeddingConfig.DEVICE
        )
        print(f"   Model: {embedding_service.model_name}")
        print(f"   Dimension: {embedding_service.embedding_dimension}")
        
        # Initialize vector store (check if collection exists)
        print(f"\n2. Connecting to Qdrant at {DatabaseConfig.HOST}:{DatabaseConfig.PORT}...")
        vector_store = VectorStoreService(
            host=DatabaseConfig.HOST,
            port=DatabaseConfig.PORT,
            collection_name=DatabaseConfig.DEFAULT_COLLECTION,
            embedding_dimension=embedding_service.embedding_dimension
        )
        
        # Check if collection has data
        try:
            info = vector_store.get_collection_info()
            points_count = info.get('points_count', 0)
            print(f"   Collection '{DatabaseConfig.DEFAULT_COLLECTION}' found with {points_count} documents")
            
            if points_count == 0:
                print("WARNING: Collection is empty. Please process some documents first.")
                print("   Use the Document Processing interface to add documents.")
                return
                
        except Exception as e:
            print(f"ERROR: Could not access collection: {e}")
            print("   Make sure Qdrant is running and you've processed documents")
            return
        
        # Initialize chatbot
        print(f"\n3. Initializing RAG chatbot with {LLMConfig.MODEL}...")
        chatbot = RAGChatbotService(
            vector_store=vector_store,
            embedding_service=embedding_service
        )
        print("SUCCESS: Chatbot initialized successfully")
        
        # Display system info
        print(f"\nSystem Information:")
        info = chatbot.get_system_info()
        for key, value in info.items():
            print(f"   {key}: {value}")
        
        # Test query
        print(f"\n4. Testing chatbot with sample query...")
        test_query = "What is the main topic of the documents?"
        print(f"   Query: '{test_query}'")
        
        response = chatbot.chat(test_query)
        
        print(f"\nChatbot Response:")
        print(f"   Answer: {response.answer}")
        print(f"   Sources: {len(response.sources)}")
        print(f"   Confidence: {response.confidence_score:.3f}")
        print(f"   Processing time: {response.processing_time:.2f}s")
        
        if response.sources:
            print(f"\nSources:")
            for i, source in enumerate(response.sources[:3], 1):  # Show first 3
                print(f"   {i}. {source['document']} (Page {source['page']}, Score: {source['similarity_score']:.3f})")
                print(f"      Preview: {source['preview'][:100]}...")
        
        print(f"\nSUCCESS: RAG Chatbot test completed successfully!")
        print(f"\nTo use the full interface, run:")
        print(f"   streamlit run src/sds_rag/ui/main_app.py")
        
    except ImportError as e:
        print(f"ERROR: Import error: {e}")
        print("   Make sure all dependencies are installed:")
        print("   poetry install")
        
    except Exception as e:
        print(f"ERROR: Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_chatbot()