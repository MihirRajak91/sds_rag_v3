"""
Test configuration system
"""
import sys
from pathlib import Path
sys.path.append('src')

def test_config():
    print("Testing configuration system...")
    
    try:
        from sds_rag.config import (
            DatabaseConfig, EmbeddingConfig, ChunkingConfig,
            PDFConfig, UIConfig, SearchConfig, get_all_config, print_config
        )
        
        print("SUCCESS: Configuration classes imported")
        
        # Test individual configs
        print(f"\nDatabase: {DatabaseConfig.HOST}:{DatabaseConfig.PORT}")
        print(f"Embedding Model: {EmbeddingConfig.MODEL}")
        print(f"Chunk Size: {ChunkingConfig.DEFAULT_SIZE}")
        print(f"PDF Output: {PDFConfig.OUTPUT_DIR}")
        print(f"UI Port: {UIConfig.PORT}")
        print(f"Search Limit: {SearchConfig.DEFAULT_LIMIT}")
        
        # Test complete config
        print("\n" + "="*50)
        print_config()
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_config()