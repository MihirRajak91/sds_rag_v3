"""
Simple UI test
"""
import sys
from pathlib import Path
sys.path.append('src')

try:
    from sds_rag.ui.streamlit_app import main
    from sds_rag.services.pdf_extracting_service import PDFExtractor
    from sds_rag.services.chunking_service import ChunkingService
    from sds_rag.services.embedding_service import EmbeddingService
    from sds_rag.services.vector_store_service import VectorStoreService
    
    print("SUCCESS: All services imported for integrated RAG UI")
    print("Ready to launch: poetry run streamlit run src/sds_rag/ui/streamlit_app.py")
    
except Exception as e:
    print(f"ERROR: {e}")