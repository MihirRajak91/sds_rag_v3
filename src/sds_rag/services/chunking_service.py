"""
Chunking service for processing extracted PDF content into semantic chunks.
Handles both text and tabular data with context preservation for RAG systems.
"""

import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
from dataclasses import dataclass, asdict
from datetime import datetime

# Add models path for future use
sys.path.append(str(Path(__file__).parent.parent))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a semantic chunk of document content"""
    chunk_id: str
    content: str
    chunk_type: str  # 'text', 'table', 'mixed'
    source_document: str
    page_number: Optional[int] = None
    table_id: Optional[str] = None
    context: Optional[str] = None
    metadata: Dict[str, Any] = None
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.metadata is None:
            self.metadata = {}

class ChunkingService:
    """
    Service for chunking extracted PDF content into semantic units.
    Optimized for financial documents with text and tabular data.
    """
    
    def __init__(self, 
                 text_chunk_size: int = 1000,
                 text_overlap: int = 200,
                 min_chunk_size: int = 100):
        """
        Initialize the chunking service.
        
        Args:
            text_chunk_size: Maximum size for text chunks in characters
            text_overlap: Overlap between consecutive text chunks
            min_chunk_size: Minimum chunk size to keep
        """
        self.text_chunk_size = text_chunk_size
        self.text_overlap = text_overlap
        self.min_chunk_size = min_chunk_size
        
        logger.info(f"Initialized ChunkingService with chunk_size={text_chunk_size}, overlap={text_overlap}")
    
    def chunk_extracted_document(self, extraction_dir: Path) -> List[DocumentChunk]:
        """
        Process all extracted files from a document into chunks.
        
        Args:
            extraction_dir: Directory containing extracted PDF content
            
        Returns:
            List of DocumentChunk objects ready for embedding
        """
        chunks = []
        doc_name = extraction_dir.name
        
        logger.info(f"Processing extraction directory: {extraction_dir}")
        
        # Process full text file
        text_file = extraction_dir / f"{doc_name}_full_text.txt"
        if text_file.exists():
            text_chunks = self._chunk_full_text(text_file, doc_name)
            chunks.extend(text_chunks)
            logger.info(f"Created {len(text_chunks)} text chunks from full text")
        
        # Process individual table files
        table_files = list(extraction_dir.glob("table_*_with_context.txt"))
        for table_file in table_files:
            table_chunk = self._chunk_table_file(table_file, doc_name)
            if table_chunk:
                chunks.append(table_chunk)
        
        logger.info(f"Created {len(table_files)} table chunks")
        
        # Process summary file as context
        summary_file = extraction_dir / "extraction_summary.txt"
        if summary_file.exists():
            summary_chunk = self._chunk_summary_file(summary_file, doc_name)
            if summary_chunk:
                chunks.append(summary_chunk)
                logger.info("Created summary chunk")
        
        logger.info(f"Total chunks created: {len(chunks)}")
        return chunks
    
    def _chunk_full_text(self, text_file: Path, doc_name: str) -> List[DocumentChunk]:
        """Chunk the full text file into overlapping segments"""
        chunks = []
        
        with open(text_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by pages first
        page_sections = content.split('--- PAGE ')
        
        for page_idx, page_content in enumerate(page_sections):
            if not page_content.strip():
                continue
                
            # Extract page number from content
            page_match = re.match(r'^(\d+)', page_content)
            page_num = int(page_match.group(1)) if page_match else page_idx
            
            # Clean page content
            page_text = re.sub(r'^\d+\s*---\s*\n\n', '', page_content, flags=re.MULTILINE)
            
            # Create sliding window chunks for this page
            page_chunks = self._create_sliding_chunks(
                page_text, 
                doc_name, 
                page_num,
                chunk_type='text'
            )
            chunks.extend(page_chunks)
        
        return chunks
    
    def _create_sliding_chunks(self, text: str, doc_name: str, page_num: int, chunk_type: str) -> List[DocumentChunk]:
        """Create overlapping chunks from text using sliding window"""
        chunks = []
        
        if len(text) < self.min_chunk_size:
            return []
        
        start = 0
        chunk_counter = 1
        
        while start < len(text):
            # Define chunk boundaries
            end = start + self.text_chunk_size
            
            # Try to break at sentence or paragraph boundaries
            if end < len(text):
                # Look for sentence endings near the boundary
                sentence_end = text.rfind('.', start, end)
                paragraph_end = text.rfind('\n\n', start, end)
                
                # Choose the best break point
                if paragraph_end > start + self.min_chunk_size:
                    end = paragraph_end + 2
                elif sentence_end > start + self.min_chunk_size:
                    end = sentence_end + 1
            
            chunk_text = text[start:end].strip()
            
            if len(chunk_text) >= self.min_chunk_size:
                chunk = DocumentChunk(
                    chunk_id=f"{doc_name}_page{page_num}_chunk{chunk_counter}",
                    content=chunk_text,
                    chunk_type=chunk_type,
                    source_document=doc_name,
                    page_number=page_num,
                    metadata={
                        'chunk_size': len(chunk_text),
                        'chunk_index': chunk_counter,
                        'start_pos': start,
                        'end_pos': end
                    }
                )
                chunks.append(chunk)
                chunk_counter += 1
            
            # Move start position with overlap
            start = end - self.text_overlap
            
            if start >= len(text) or end >= len(text):
                break
        
        return chunks
    
    def _chunk_table_file(self, table_file: Path, doc_name: str) -> Optional[DocumentChunk]:
        """Process a table file into a single semantic chunk"""
        try:
            with open(table_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract metadata from table file content
            table_id = self._extract_table_id(content)
            page_num = self._extract_page_number(content)
            context = self._extract_context(content)
            table_data = self._extract_table_data(content)
            
            # Create enhanced content combining context and data
            enhanced_content = self._create_enhanced_table_content(content, context, table_data)
            
            chunk = DocumentChunk(
                chunk_id=f"{doc_name}_{table_id}",
                content=enhanced_content,
                chunk_type='table',
                source_document=doc_name,
                page_number=page_num,
                table_id=table_id,
                context=context,
                metadata={
                    'original_file': table_file.name,
                    'table_rows': table_data.count('|') // table_data.count('\n') if table_data else 0,
                    'has_context': bool(context),
                    'content_length': len(enhanced_content)
                }
            )
            
            return chunk
            
        except Exception as e:
            logger.error(f"Error processing table file {table_file}: {e}")
            return None
    
    def _chunk_summary_file(self, summary_file: Path, doc_name: str) -> Optional[DocumentChunk]:
        """Process the extraction summary as a document overview chunk"""
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            chunk = DocumentChunk(
                chunk_id=f"{doc_name}_summary",
                content=content,
                chunk_type='summary',
                source_document=doc_name,
                metadata={
                    'is_summary': True,
                    'content_length': len(content)
                }
            )
            
            return chunk
            
        except Exception as e:
            logger.error(f"Error processing summary file {summary_file}: {e}")
            return None
    
    def _extract_table_id(self, content: str) -> str:
        """Extract table ID from table file content"""
        match = re.search(r'\*\*Table ID:\*\* (table_\d+)', content)
        return match.group(1) if match else "unknown_table"
    
    def _extract_page_number(self, content: str) -> Optional[int]:
        """Extract page number from table file content"""
        match = re.search(r'\*\*Estimated Location in Document:\*\* Page (\d+)', content)
        return int(match.group(1)) if match else None
    
    def _extract_context(self, content: str) -> Optional[str]:
        """Extract surrounding text context from table file"""
        match = re.search(r'\*\*Context from Surrounding Text:\*\*\n(.*?)\n\n', content, re.DOTALL)
        return match.group(1).strip() if match else None
    
    def _extract_table_data(self, content: str) -> Optional[str]:
        """Extract the markdown table data"""
        match = re.search(r'\*\*Table Data \(in Markdown format\):\*\*\n(.*)', content, re.DOTALL)
        return match.group(1).strip() if match else None
    
    def _create_enhanced_table_content(self, original_content: str, context: str, table_data: str) -> str:
        """Create enhanced content for better semantic search"""
        parts = []
        
        # Add context for better understanding
        if context:
            parts.append(f"Context: {context}")
        
        # Add the table data
        if table_data:
            parts.append(f"Table Data:\n{table_data}")
        
        # Add original structured content as fallback
        if not parts:
            parts.append(original_content)
        
        return "\n\n".join(parts)
    
    def save_chunks(self, chunks: List[DocumentChunk], output_dir: Path) -> Path:
        """Save chunks to JSON file for further processing"""
        import json
        
        output_dir.mkdir(parents=True, exist_ok=True)
        chunks_file = output_dir / "chunks.json"
        
        # Convert chunks to JSON-serializable format
        chunks_data = [asdict(chunk) for chunk in chunks]
        
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(chunks)} chunks to {chunks_file}")
        return chunks_file
    
    def load_chunks(self, chunks_file: Path) -> List[DocumentChunk]:
        """Load chunks from JSON file"""
        import json
        
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        chunks = [DocumentChunk(**chunk_data) for chunk_data in chunks_data]
        logger.info(f"Loaded {len(chunks)} chunks from {chunks_file}")
        return chunks

# Example usage
if __name__ == "__main__":
    # Test chunking with extracted Apple document
    chunking_service = ChunkingService(text_chunk_size=800, text_overlap=150)
    
    # Process the existing extracted document
    extraction_dir = Path("src/sds_rag/services/output/2022 Q3 AAPL")
    
    if extraction_dir.exists():
        chunks = chunking_service.chunk_extracted_document(extraction_dir)
        
        # Save chunks
        output_dir = Path("chunked_documents")
        chunks_file = chunking_service.save_chunks(chunks, output_dir)
        
        print(f"✅ Created {len(chunks)} chunks")
        print(f"📁 Saved to: {chunks_file}")
        
        # Show sample chunks
        print("\n📄 Sample chunks:")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n{i+1}. {chunk.chunk_id} ({chunk.chunk_type})")
            print(f"   Content length: {len(chunk.content)} chars")
            print(f"   Preview: {chunk.content[:200]}...")
    else:
        print("❌ No extracted documents found. Run PDF extraction first.")