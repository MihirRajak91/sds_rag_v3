"""
Test chunking service
"""
import sys
from pathlib import Path
sys.path.append('src')

from sds_rag.services.chunking_service import ChunkingService

def test_chunking():
    print("1. Initializing ChunkingService...")
    chunker = ChunkingService(text_chunk_size=800, text_overlap=150)
    
    print("2. Locating extracted documents...")
    extraction_dir = Path('output/2022 Q3 AAPL')
    
    if not extraction_dir.exists():
        print(f"ERROR: Directory not found: {extraction_dir}")
        return
    
    print(f"   Found directory: {extraction_dir}")
    files = list(extraction_dir.glob("*"))
    print(f"   Files found: {len(files)}")
    
    print("3. Processing chunks...")
    chunks = chunker.chunk_extracted_document(extraction_dir)
    
    print("4. Saving chunks...")
    output_dir = Path('test_output')
    chunks_file = chunker.save_chunks(chunks, output_dir)
    
    print(f"SUCCESS: Created {len(chunks)} chunks")
    print(f"OUTPUT: Saved to {chunks_file}")
    
    # Show breakdown
    types = {}
    for chunk in chunks:
        types[chunk.chunk_type] = types.get(chunk.chunk_type, 0) + 1
    
    print("Chunk breakdown:")
    for chunk_type, count in types.items():
        print(f"  {chunk_type}: {count} chunks")
    
    # Show samples
    print("\nSample chunks:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"  {i+1}. {chunk.chunk_id} ({chunk.chunk_type})")
        print(f"     Length: {len(chunk.content)} chars")
        print(f"     Preview: {chunk.content[:100]}...")
    
    return chunks

if __name__ == "__main__":
    chunks = test_chunking()