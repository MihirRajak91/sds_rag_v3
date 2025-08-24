"""
Simple chunking test
"""
import sys
from pathlib import Path
sys.path.append('src')

from sds_rag.services.chunking_service import ChunkingService

def test_simple():
    print("Testing chunking service...")
    
    chunker = ChunkingService(text_chunk_size=500, text_overlap=100)
    extraction_dir = Path('output/2022 Q3 AAPL')
    
    # Check if files exist
    text_file = extraction_dir / "2022 Q3 AAPL_full_text.txt"
    table_files = list(extraction_dir.glob("table_*_with_context.txt"))
    
    print(f"Text file exists: {text_file.exists()}")
    print(f"Table files found: {len(table_files)}")
    
    if text_file.exists():
        print(f"Text file size: {text_file.stat().st_size} bytes")
    
    # Test with just first 3 table files
    chunks = []
    
    # Process just a few table files first
    for i, table_file in enumerate(table_files[:3]):
        print(f"Processing {table_file.name}...")
        chunk = chunker._chunk_table_file(table_file, "test_doc")
        if chunk:
            chunks.append(chunk)
    
    print(f"Created {len(chunks)} table chunks")
    
    for chunk in chunks:
        print(f"- {chunk.chunk_id}: {len(chunk.content)} chars")

if __name__ == "__main__":
    test_simple()