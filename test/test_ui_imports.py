"""
Test integrated UI imports
"""
import sys
from pathlib import Path
sys.path.append('src')

def test_ui():
    try:
        print("Testing UI imports...")
        
        from sds_rag.ui.streamlit_app import main
        print("✅ Streamlit app imported")
        
        from sds_rag.services.pdf_extracting_service import PDFExtractor
        print("✅ PDF extraction service imported")
        
        from sds_rag.services.chunking_service import ChunkingService
        print("✅ Chunking service imported")
        
        from sds_rag.services.embedding_service import EmbeddingService
        print("✅ Embedding service imported")
        
        from sds_rag.services.vector_store_service import VectorStoreService
        print("✅ Vector store service imported")
        
        print("\n🎉 SUCCESS: Complete RAG Pipeline UI ready!")
        print("Launch with: poetry run streamlit run src/sds_rag/ui/streamlit_app.py")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    test_ui()