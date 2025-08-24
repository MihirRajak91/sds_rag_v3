"""
Streamlit UI for Complete RAG Pipeline
Upload PDFs → Extract → Chunk → Embed → Store in Qdrant → Ready for Search
"""

import streamlit as st
import tempfile
import os
from pathlib import Path
import sys
import time

# Add the services directory to path
sys.path.append(str(Path(__file__).parent.parent))
from services.pdf_extracting_service import PDFExtractor
from services.chunking_service import ChunkingService
from services.embedding_service import EmbeddingService
from services.vector_store_service import VectorStoreService

def main():
    st.set_page_config(
        page_title="RAG Pipeline",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Complete RAG Pipeline")
    st.markdown("Upload a PDF document → Extract → Chunk → Embed → Store in Vector Database → Ready for Search!")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ RAG Pipeline Configuration")
        
        # Extraction settings
        st.subheader("1. PDF Extraction")
        extract_text = st.checkbox("Extract Text", value=True, help="Extract narrative text from PDF")
        extract_tables = st.checkbox("Extract Tables", value=True, help="Extract tabular data using tabula-py")
        
        if extract_tables:
            pages_option = st.selectbox(
                "Pages to extract from:",
                ["all", "first 10", "custom"],
                index=0,
                help="Choose which pages to extract tables from"
            )
            
            if pages_option == "custom":
                custom_pages = st.text_input(
                    "Page numbers (e.g., 1,3-5,10):",
                    placeholder="1,3-5,10"
                )
                pages = custom_pages if custom_pages else "all"
            elif pages_option == "first 10":
                pages = "1-10"
            else:
                pages = "all"
        
        # Chunking settings
        st.subheader("2. Text Chunking")
        chunk_size = st.slider("Chunk Size (characters)", 400, 1500, 800, 50, help="Size of each text chunk")
        chunk_overlap = st.slider("Chunk Overlap", 50, 300, 150, 25, help="Overlap between chunks")
        
        # Embedding settings  
        st.subheader("3. Embeddings")
        embedding_model = st.selectbox(
            "Embedding Model:",
            ["default", "high_quality", "multilingual"],
            index=0,
            help="Choose embedding model quality vs speed"
        )
        
        # Vector store settings
        st.subheader("4. Vector Storage")
        collection_name = st.text_input(
            "Collection Name:",
            value="financial_docs",
            help="Name for the Qdrant collection"
        )
        
        # RAG pipeline control
        st.subheader("5. Pipeline Options")
        run_full_pipeline = st.checkbox("Run Complete RAG Pipeline", value=True, 
                                       help="Extract → Chunk → Embed → Store in Qdrant")
        
        if not run_full_pipeline:
            st.info("💡 Disable to run extraction only (no embedding/storage)")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Document")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a financial document (10-Q, 10-K, earnings report, etc.)"
        )
        
        if uploaded_file is not None:
            # Display file info
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            st.info(f"📊 File size: {uploaded_file.size:,} bytes")
            
            # Extract button
            if run_full_pipeline:
                extract_btn = st.button(
                    "🚀 Run Complete RAG Pipeline",
                    type="primary",
                    help="Extract → Chunk → Embed → Store in Qdrant"
                )
            else:
                extract_btn = st.button(
                    "📄 Extract Only",
                    type="primary", 
                    help="Extract text and tables only"
                )
            
            if extract_btn:
                pipeline_config = {
                    'extract_text': extract_text,
                    'extract_tables': extract_tables, 
                    'pages': pages,
                    'chunk_size': chunk_size,
                    'chunk_overlap': chunk_overlap,
                    'embedding_model': embedding_model,
                    'collection_name': collection_name,
                    'run_full_pipeline': run_full_pipeline
                }
                process_document_pipeline(uploaded_file, pipeline_config)
    
    with col2:
        st.header("🤖 RAG Pipeline")
        st.markdown("""
        ### Complete RAG Workflow:
        1. **📄 PDF Extraction**: Extract text and tables from your document
        2. **📝 Chunking**: Split content into semantic chunks
        3. **🧠 Embeddings**: Generate vector representations using AI models
        4. **🗄️ Vector Storage**: Store in Qdrant database for fast search
        5. **🔍 Ready for Search**: Query with natural language!
        
        ### Pipeline Features:
        - **Smart Chunking**: Preserves context and table relationships
        - **Multiple Models**: Choose embedding quality vs speed
        - **Rich Metadata**: Document source, page numbers, table IDs
        - **Scalable Storage**: Qdrant vector database with filtering
        
        ### Supported Documents:
        - Financial reports (10-Q, 10-K)
        - Earnings statements  
        - Annual reports
        - Research papers with tables
        - Any PDF with structured content
        
        ### After Processing:
        - **Search Ready**: Use natural language queries
        - **Filtered Search**: Search by document, table, page
        - **Context Preserved**: Each result includes surrounding context
        - **Production Ready**: Scalable for large document collections
        """)

def process_document_pipeline(uploaded_file, config):
    """Process the uploaded PDF through the complete RAG pipeline"""
    
    # Create progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Create columns for pipeline status
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        extraction_status = st.empty()
    with col2: 
        chunking_status = st.empty()
    with col3:
        embedding_status = st.empty()
    with col4:
        storage_status = st.empty()
    
    try:
        # === STEP 1: PDF EXTRACTION ===
        extraction_status.markdown("📄 **Extracting...**")
        status_text.text("📄 Step 1/4: PDF Extraction...")
        progress_bar.progress(10)
        
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Create output directory
        doc_name = uploaded_file.name.replace('.pdf', '')
        output_base = Path("extracted_documents")
        output_base.mkdir(exist_ok=True)
        
        class CustomPDFExtractor(PDFExtractor):
            def __init__(self, pdf_path: str, output_name: str):
                super().__init__(pdf_path)
                self.output_dir = output_base / output_name
                self.output_dir.mkdir(parents=True, exist_ok=True)
        
        extractor = CustomPDFExtractor(tmp_file_path, doc_name)
        progress_bar.progress(15)
        
        # Extract content
        full_text = ""
        tables = []
        
        if config['extract_text']:
            full_text = extractor.extract_text()
            progress_bar.progress(25)
        
        if config['extract_tables']:
            kwargs = {"pages": config['pages']} if config['pages'] != "all" else {}
            tables = extractor.extract_tables(**kwargs)
            progress_bar.progress(35)
        
        # Save extraction
        if config['extract_text'] or config['extract_tables']:
            extractor.save_extraction(full_text, tables)
        
        progress_bar.progress(40)
        extraction_status.markdown("✅ **Extracted**")
        
        # Cleanup temp file
        os.unlink(tmp_file_path)
        
        # If not running full pipeline, show results and stop
        if not config['run_full_pipeline']:
            status_text.text("✅ PDF Extraction completed!")
            display_extraction_results(extractor, full_text, tables, config)
            return
        
        # === STEP 2: CHUNKING ===
        chunking_status.markdown("📝 **Chunking...**")
        status_text.text("📝 Step 2/4: Creating semantic chunks...")
        progress_bar.progress(45)
        
        chunker = ChunkingService(
            text_chunk_size=config['chunk_size'],
            text_overlap=config['chunk_overlap']
        )
        
        chunks = chunker.chunk_extracted_document(extractor.output_dir)
        progress_bar.progress(55)
        chunking_status.markdown("✅ **Chunked**")
        
        # === STEP 3: EMBEDDINGS ===
        embedding_status.markdown("🧠 **Embedding...**")
        status_text.text("🧠 Step 3/4: Generating embeddings...")
        progress_bar.progress(60)
        
        embedder = EmbeddingService(
            model_name=config['embedding_model'],
            batch_size=8  # Smaller batch for UI responsiveness
        )
        
        embedded_chunks = embedder.embed_chunks(chunks)
        progress_bar.progress(75)
        embedding_status.markdown("✅ **Embedded**")
        
        # === STEP 4: VECTOR STORAGE ===
        storage_status.markdown("🗄️ **Storing...**")
        status_text.text("🗄️ Step 4/4: Storing in Qdrant...")
        progress_bar.progress(80)
        
        # Clean collection name
        clean_collection_name = config['collection_name'].lower().replace(' ', '_')
        
        vector_store = VectorStoreService(
            collection_name=clean_collection_name,
            embedding_dimension=embedder.embedding_dimension
        )
        
        point_ids = vector_store.add_embedded_chunks(embedded_chunks)
        progress_bar.progress(95)
        storage_status.markdown("✅ **Stored**")
        
        # === COMPLETION ===
        progress_bar.progress(100)
        status_text.text("🎉 Complete RAG Pipeline finished!")
        
        # Display comprehensive results
        display_pipeline_results(extractor, chunks, embedded_chunks, vector_store, point_ids, config)
        
    except Exception as e:
        st.error(f"❌ Pipeline Error: {str(e)}")
        status_text.text("❌ Pipeline failed")
        
        # Show which step failed
        import traceback
        with st.expander("Error Details"):
            st.code(traceback.format_exc())
    
def display_extraction_results(extractor, full_text, tables, config):
    """Display extraction results"""
    
    st.success("🎉 **PDF Extraction Completed Successfully!**")
    
    # Results summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📄 Pages", len(extractor.reader.pages))
    
    with col2:
        if config['extract_tables']:
            st.metric("📊 Tables Found", len(tables))
        else:
            st.metric("📊 Tables Found", "Skipped")
    
    with col3:
        if config['extract_text']:
            st.metric("📝 Text Length", f"{len(full_text):,} chars")
        else:
            st.metric("📝 Text Length", "Skipped")
    
    # Output location
    st.info(f"📁 **Extracted files saved to:** `{extractor.output_dir}`")
    
    # Preview sections
    if config['extract_text'] and full_text:
        with st.expander("📝 Text Preview (First 1000 characters)"):
            st.text(full_text[:1000] + "..." if len(full_text) > 1000 else full_text)
    
    if config['extract_tables'] and tables:
        with st.expander(f"📊 Tables Preview ({len(tables)} found)"):
            for i, table in enumerate(tables[:3]):  # Show first 3 tables
                st.subheader(f"Table {i+1}")
                if not table.empty:
                    st.dataframe(table.head(), use_container_width=True)
                else:
                    st.warning("Table appears to be empty")
            
            if len(tables) > 3:
                st.info(f"... and {len(tables) - 3} more tables saved to output directory")

def display_pipeline_results(extractor, chunks, embedded_chunks, vector_store, point_ids, config):
    """Display complete RAG pipeline results"""
    
    st.success("🎉 **Complete RAG Pipeline Finished Successfully!**")
    
    # Pipeline summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("📄 Pages", len(extractor.reader.pages))
    with col2:
        st.metric("📝 Chunks", len(chunks))
    with col3:
        st.metric("🧠 Embeddings", len(embedded_chunks))
    with col4:
        st.metric("🗄️ Vectors", len(point_ids))
    with col5:
        # Get collection info
        try:
            info = vector_store.get_collection_info()
            total_points = info.get('points_count', len(point_ids))
        except:
            total_points = len(point_ids)
        st.metric("📊 Total in DB", total_points)
    
    # Pipeline status
    st.info(f"✅ **Document ready for search in collection:** `{config['collection_name']}`")
    
    # Show processing details
    with st.expander("📋 Pipeline Processing Details"):
        st.markdown(f"""
        **Document Processing:**
        - **Source**: {extractor.pdf_path.name}  
        - **Output Directory**: `{extractor.output_dir}`
        - **Pages Processed**: {len(extractor.reader.pages)}
        
        **Chunking Results:**
        - **Chunk Size**: {config['chunk_size']} characters
        - **Overlap**: {config['chunk_overlap']} characters
        - **Total Chunks**: {len(chunks)}
        
        **Embedding Generation:**
        - **Model**: {config['embedding_model']}
        - **Dimension**: {embedded_chunks[0]['embedding_dimension'] if embedded_chunks else 'N/A'}
        - **Enhanced Text**: Context-aware embeddings
        
        **Vector Storage:**
        - **Collection**: {config['collection_name']}
        - **Points Added**: {len(point_ids)}
        - **Database**: Qdrant with metadata indexing
        """)
    
    # Show chunk samples
    if chunks:
        with st.expander(f"📝 Sample Chunks ({len(chunks)} total)"):
            chunk_types = {}
            for chunk in chunks:
                chunk_types[chunk.chunk_type] = chunk_types.get(chunk.chunk_type, 0) + 1
            
            st.markdown("**Chunk Type Distribution:**")
            for chunk_type, count in chunk_types.items():
                st.markdown(f"- **{chunk_type.title()}**: {count} chunks")
            
            st.markdown("**Sample Chunks:**")
            for i, chunk in enumerate(chunks[:3]):
                with st.container():
                    st.markdown(f"**{i+1}. {chunk.chunk_id}** ({chunk.chunk_type})")
                    st.markdown(f"*Length: {len(chunk.content)} chars*")
                    st.text(chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content)
    
    # Test search interface
    st.markdown("---")
    st.header("🔍 Test Your RAG System")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        test_query = st.text_input(
            "Try a search query:",
            placeholder="What was the revenue in Q3 2022?",
            help="Test your newly created RAG system with a natural language query"
        )
    
    with col2:
        search_limit = st.selectbox("Results to show:", [3, 5, 10], index=0)
    
    if test_query:
        with st.spinner("Searching..."):
            try:
                # Recreate embedder for search
                embedder = EmbeddingService(model_name=config['embedding_model'])
                
                results = vector_store.search_by_text(
                    query_text=test_query,
                    embedding_service=embedder,
                    limit=search_limit,
                    score_threshold=0.1
                )
                
                if results:
                    st.success(f"Found {len(results)} relevant results:")
                    
                    for i, result in enumerate(results):
                        payload = result['payload']
                        
                        with st.container():
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.markdown(f"**Result {i+1}: {payload.get('chunk_id', 'Unknown')}**")
                                st.markdown(f"*Type: {payload.get('chunk_type', 'unknown')} | Similarity: {result['score']:.3f}*")
                                
                                content = payload.get('content', '')
                                st.markdown(content[:300] + "..." if len(content) > 300 else content)
                            
                            with col2:
                                if payload.get('page_number'):
                                    st.metric("Page", payload.get('page_number'))
                                if payload.get('table_id'):
                                    st.metric("Table", payload.get('table_id', '').replace('table_', ''))
                        
                        st.markdown("---")
                else:
                    st.warning("No relevant results found. Try a different query.")
                    
            except Exception as e:
                st.error(f"Search error: {str(e)}")
    
    # Instructions for next steps
    with st.expander("🚀 Next Steps"):
        st.markdown(f"""
        Your document is now ready for production RAG queries! Here's how to use it:
        
        **Collection Information:**
        - **Name**: `{config['collection_name']}`  
        - **Total Chunks**: {len(chunks)}
        - **Search Ready**: Yes ✅
        
        **Query Examples:**
        - "What was the total revenue?"
        - "Show me the operating expenses"
        - "Financial performance summary"
        - "Tables about iPhone sales"
        
        **Advanced Features:**
        - **Filtered Search**: Search only tables, specific pages, etc.
        - **Metadata**: Each result includes source, page, table ID
        - **Context**: Results include surrounding text context
        - **Scalable**: Add more documents to the same collection
        
        **Integration:**
        - Use the `VectorStoreService` to query from your applications
        - Combine with LangChain for Q&A chatbots
        - Build custom interfaces for your users
        """)

if __name__ == "__main__":
    main()