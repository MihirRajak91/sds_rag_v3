"""
Test script to verify all RAG service imports
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    try:
        from sds_rag.services.chunking_service import ChunkingService, DocumentChunk
        print("SUCCESS: ChunkingService imported")
        
        from sds_rag.services.embedding_service import EmbeddingService
        print("SUCCESS: EmbeddingService imported")
        
        from sds_rag.services.vector_store_service import VectorStoreService
        print("SUCCESS: VectorStoreService imported")
        
        print("SUCCESS: All RAG services ready!")
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False

if __name__ == "__main__":
    test_imports()