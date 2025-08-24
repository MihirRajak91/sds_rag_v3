"""
Example showing how to use the centralized configuration system
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from sds_rag.config import (
    DatabaseConfig, EmbeddingConfig, ChunkingConfig,
    PDFConfig, SearchConfig, get_all_config
)
from sds_rag.services.embedding_service import EmbeddingService
from sds_rag.services.vector_store_service import VectorStoreService
from sds_rag.services.chunking_service import ChunkingService

def example_with_config():
    """Example of using services with centralized config"""
    
    print("=== Using Centralized Configuration ===\n")
    
    # 1. Initialize services using config values
    print("1. Initializing services with config...")
    
    # Embedding service with config
    embedding_service = EmbeddingService(
        model_name=EmbeddingConfig.MODEL,
        device=EmbeddingConfig.DEVICE,
        batch_size=EmbeddingConfig.BATCH_SIZE
    )
    print(f"   Embedding: {EmbeddingConfig.MODEL} ({embedding_service.embedding_dimension}D)")
    
    # Vector store with config
    vector_store = VectorStoreService(
        host=DatabaseConfig.HOST,
        port=DatabaseConfig.PORT,
        collection_name=DatabaseConfig.DEFAULT_COLLECTION,
        embedding_dimension=embedding_service.embedding_dimension
    )
    print(f"   Vector Store: {DatabaseConfig.HOST}:{DatabaseConfig.PORT}")
    
    # Chunking service with config
    chunking_service = ChunkingService(
        text_chunk_size=ChunkingConfig.DEFAULT_SIZE,
        text_overlap=ChunkingConfig.DEFAULT_OVERLAP,
        min_chunk_size=ChunkingConfig.MIN_SIZE
    )
    print(f"   Chunking: {ChunkingConfig.DEFAULT_SIZE} chars, {ChunkingConfig.DEFAULT_OVERLAP} overlap")
    
    # 2. Show how to access config for file operations
    print(f"\n2. File paths from config:")
    print(f"   PDF Output: {PDFConfig.OUTPUT_DIR}")
    print(f"   Cache Dir: {EmbeddingConfig.MODEL_CONFIGS[EmbeddingConfig.MODEL]['name']}")
    
    # 3. Show search configuration
    print(f"\n3. Search settings:")
    print(f"   Default limit: {SearchConfig.DEFAULT_LIMIT}")
    print(f"   Similarity threshold: {SearchConfig.DEFAULT_THRESHOLD}")
    
    # 4. Show complete config dump
    print(f"\n4. Complete configuration:")
    config = get_all_config()
    for section, settings in config.items():
        print(f"   [{section.upper()}]")
        for key, value in settings.items():
            print(f"     {key}: {value}")

def customize_config_example():
    """Example of customizing configuration via environment variables"""
    
    print("\n=== Customizing Configuration ===\n")
    print("To customize settings, modify your .env file:")
    print()
    print("# Change embedding model to high quality:")
    print("EMBEDDING_MODEL=high_quality")
    print()
    print("# Change chunk size:")
    print("DEFAULT_CHUNK_SIZE=1200")
    print("DEFAULT_CHUNK_OVERLAP=200")
    print()
    print("# Change database settings:")
    print("QDRANT_HOST=my-qdrant-server.com")
    print("QDRANT_PORT=6334")
    print("QDRANT_DEFAULT_COLLECTION=my_documents")
    print()
    print("# Change search behavior:")
    print("DEFAULT_SEARCH_LIMIT=20")
    print("DEFAULT_SIMILARITY_THRESHOLD=0.2")

if __name__ == "__main__":
    example_with_config()
    customize_config_example()