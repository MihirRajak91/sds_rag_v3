"""
Simple Main Streamlit App with RAG Pipeline and Chatbot
"""

import streamlit as st
import sys
from pathlib import Path

# Add the services directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config.constants import UIConfig

def main():
    st.set_page_config(
        page_title="RAG System",
        page_icon="🤖",
        layout="wide"
    )
    
    # Initialize session state
    if "current_app" not in st.session_state:
        st.session_state.current_app = "home"
    
    # Top navigation
    st.title("🤖 RAG System - Document Intelligence Platform")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🏠 Home", type="primary" if st.session_state.current_app == "home" else "secondary"):
            st.session_state.current_app = "home"
            st.rerun()
    
    with col2:
        if st.button("📄 Process Documents", type="primary" if st.session_state.current_app == "pipeline" else "secondary"):
            st.session_state.current_app = "pipeline"
            st.rerun()
    
    with col3:
        if st.button("💬 Chat with Documents", type="primary" if st.session_state.current_app == "chatbot" else "secondary"):
            st.session_state.current_app = "chatbot"
            st.rerun()
            
    with col4:
        if st.button("🔄 Refresh", type="secondary"):
            st.rerun()
    
    st.markdown("---")
    
    # Page content
    if st.session_state.current_app == "home":
        show_home_page()
    elif st.session_state.current_app == "pipeline":
        show_pipeline_page()
    elif st.session_state.current_app == "chatbot":
        show_chatbot_page()

def show_home_page():
    """Display the home page"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📄 Document Processing")
        st.markdown("""
        **Transform your PDFs into searchable knowledge:**
        - 📊 Extract text and tables from PDFs
        - 🧩 Smart chunking with context preservation
        - 🧠 Generate vector embeddings using AI models
        - 🗄️ Store in Qdrant vector database
        - 🔍 Enable fast semantic search
        
        **Supported Documents:**
        - Financial reports (10-Q, 10-K, earnings)
        - Research papers
        - Technical documentation
        - Any PDF with structured content
        """)
        
        if st.button("🚀 Start Processing", key="home_pipeline"):
            st.session_state.current_app = "pipeline"
            st.rerun()
    
    with col2:
        st.header("💬 AI Chatbot")
        st.markdown("""
        **Chat with your documents using Gemma 2B (via OpenRouter):**
        - 🤖 Natural language questions
        - 📚 Source-backed answers
        - 🔗 Context-aware conversations
        - ⚡ Fast retrieval from vector database
        - 📊 Confidence scoring and metadata
        
        **Example Questions:**
        - "What was the revenue in Q3?"
        - "Summarize the main risk factors"
        - "Show me operating expenses by quarter"
        - "What are the key performance metrics?"
        """)
        
        if st.button("💬 Start Chatting", key="home_chatbot"):
            st.session_state.current_app = "chatbot"
            st.rerun()
    
    # System status
    st.markdown("---")
    st.header("📊 System Status")
    
    status_col1, status_col2, status_col3 = st.columns(3)
    
    with status_col1:
        try:
            from services.vector_store_service import VectorStoreService
            from config.constants import DatabaseConfig
            
            vector_store = VectorStoreService(
                host=DatabaseConfig.HOST,
                port=DatabaseConfig.PORT,
                collection_name="test"
            )
            vector_store.client.get_collections()
            st.success("✅ Qdrant Connected")
        except Exception as e:
            st.error(f"❌ Qdrant: {str(e)[:50]}...")
    
    with status_col2:
        try:
            from services.embedding_service import EmbeddingService
            embedding_service = EmbeddingService(model_name="default")
            st.success("✅ Embeddings Ready")
        except Exception as e:
            st.error(f"❌ Embeddings: {str(e)[:50]}...")
    
    with status_col3:
        try:
            from config.constants import LLMConfig
            if LLMConfig.API_KEY and LLMConfig.API_KEY != "your_gemini_api_key_here":
                st.success("✅ OpenRouter API Key Set")
            else:
                st.warning("⚠️ OpenRouter API Key Missing")
        except Exception as e:
            st.error(f"❌ OpenRouter: {str(e)[:50]}...")

def show_pipeline_page():
    """Show document processing pipeline"""
    st.header("📄 Document Processing Pipeline")
    st.markdown("Upload PDFs → Extract → Chunk → Embed → Store in Vector Database")
    
    # Configuration sidebar
    with st.sidebar:
        st.header("⚙️ Processing Configuration")
        
        # Extraction settings
        st.subheader("1. PDF Extraction")
        extract_text = st.checkbox("Extract Text", value=True)
        extract_tables = st.checkbox("Extract Tables", value=True)
        
        # Chunking settings
        st.subheader("2. Text Chunking")
        chunk_size = st.slider("Chunk Size", 400, 1500, 800, 50)
        chunk_overlap = st.slider("Chunk Overlap", 50, 300, 150, 25)
        
        # Embedding settings
        st.subheader("3. Embeddings")
        embedding_model = st.selectbox("Model:", ["default", "high_quality", "multilingual"])
        
        # Vector store settings
        st.subheader("4. Vector Storage")
        collection_name = st.text_input("Collection Name:", value="financial_docs")
    
    # Main content
    uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
    
    if uploaded_file is not None:
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        st.info(f"📊 File size: {uploaded_file.size:,} bytes")
        
        if st.button("🚀 Run Complete RAG Pipeline", type="primary"):
            config = {
                'extract_text': extract_text,
                'extract_tables': extract_tables,
                'chunk_size': chunk_size,
                'chunk_overlap': chunk_overlap,
                'embedding_model': embedding_model,
                'collection_name': collection_name
            }
            run_pipeline(uploaded_file, config)

def run_pipeline(uploaded_file, config):
    """Run the document processing pipeline"""
    import tempfile
    import os
    from pathlib import Path
    
    progress = st.progress(0)
    status = st.empty()
    
    try:
        # Step 1: Extract PDF
        status.text("📄 Extracting PDF content...")
        progress.progress(25)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        from services.pdf_extracting_service import PDFExtractor
        extractor = PDFExtractor(tmp_file_path)
        
        full_text = ""
        tables = []
        
        if config['extract_text']:
            full_text = extractor.extract_text()
        
        if config['extract_tables']:
            tables = extractor.extract_tables()
        
        # Step 2: Chunking
        status.text("📝 Creating semantic chunks...")
        progress.progress(50)
        
        from services.chunking_service import ChunkingService
        chunker = ChunkingService(
            text_chunk_size=config['chunk_size'],
            text_overlap=config['chunk_overlap']
        )
        
        # Create temp output directory
        output_dir = Path("temp_output") / uploaded_file.name.replace('.pdf', '')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save extraction temporarily
        extractor.output_dir = output_dir
        extractor.save_extraction(full_text, tables)
        
        chunks = chunker.chunk_extracted_document(output_dir)
        
        # Step 3: Embeddings
        status.text("🧠 Generating embeddings...")
        progress.progress(75)
        
        from services.embedding_service import EmbeddingService
        embedder = EmbeddingService(model_name=config['embedding_model'])
        embedded_chunks = embedder.embed_chunks(chunks)
        
        # Step 4: Vector Storage
        status.text("🗄️ Storing in Qdrant...")
        progress.progress(90)
        
        from services.vector_store_service import VectorStoreService
        vector_store = VectorStoreService(
            collection_name=config['collection_name'],
            embedding_dimension=embedder.embedding_dimension
        )
        
        point_ids = vector_store.add_embedded_chunks(embedded_chunks)
        
        progress.progress(100)
        status.text("🎉 Pipeline completed successfully!")
        
        # Results
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📝 Chunks", len(chunks))
        with col2:
            st.metric("🧠 Embeddings", len(embedded_chunks))
        with col3:
            st.metric("🗄️ Stored", len(point_ids))
        with col4:
            st.metric("📊 Tables", len(tables))
        
        st.success(f"✅ Document processed and stored in collection: `{config['collection_name']}`")
        
        # Cleanup
        os.unlink(tmp_file_path)
        
    except Exception as e:
        st.error(f"❌ Pipeline failed: {str(e)}")

def show_chatbot_page():
    """Show chatbot interface"""
    st.header("💬 Chat with Your Documents")
    st.markdown("Ask questions about your processed documents using Gemma 2B via OpenRouter")
    
    # Configuration
    with st.sidebar:
        st.header("🔧 Chatbot Settings")
        
        collection_name = st.text_input("Collection Name:", value="financial_docs")
        embedding_model = st.selectbox("Embedding Model:", ["default", "high_quality", "multilingual"])
        
        search_limit = st.slider("Sources per query:", 1, 10, 5)
        similarity_threshold = st.slider("Similarity threshold:", 0.0, 1.0, 0.3, 0.05)
        
        if st.button("🚀 Initialize Chatbot", type="primary"):
            try:
                from services.rag_chatbot_service import RAGChatbotService
                from services.vector_store_service import VectorStoreService  
                from services.embedding_service import EmbeddingService
                from config.constants import DatabaseConfig, EmbeddingConfig
                
                # Initialize services
                embedding_service = EmbeddingService(model_name=embedding_model, device=EmbeddingConfig.DEVICE)
                vector_store = VectorStoreService(
                    host=DatabaseConfig.HOST,
                    port=DatabaseConfig.PORT,
                    collection_name=collection_name,
                    embedding_dimension=embedding_service.embedding_dimension
                )
                
                chatbot = RAGChatbotService(vector_store=vector_store, embedding_service=embedding_service)
                
                st.session_state.chatbot = chatbot
                st.session_state.chat_messages = []
                st.success("✅ Chatbot initialized successfully!")
                
            except Exception as e:
                st.error(f"❌ Failed to initialize chatbot: {str(e)}")
    
    # Chat interface
    if "chatbot" not in st.session_state:
        st.info("👈 Please initialize the chatbot using the sidebar")
        return
    
    # Initialize chat messages
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    
    # Display chat history
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander(f"📚 Sources ({len(message['sources'])})", expanded=False):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**{i}.** {source['document']} (Page {source['page']}, Score: {source['similarity_score']:.3f})")
                        st.text(f"Preview: {source['preview']}")
                        st.divider()
    
    # Chat input
    user_input = st.chat_input("Ask a question about your documents...")
    
    if user_input:
        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.write(user_input)
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chatbot.chat(user_input)
            
            st.write(response.answer)
            
            if response.sources:
                with st.expander(f"📚 Sources ({len(response.sources)})", expanded=False):
                    for i, source in enumerate(response.sources, 1):
                        st.markdown(f"**{i}.** {source['document']} (Page {source['page']}, Score: {source['similarity_score']:.3f})")
                        st.text(f"Preview: {source['preview']}")
                        st.divider()
            
            st.caption(f"⏱️ {response.processing_time:.2f}s | 📊 Confidence: {response.confidence_score:.2f}")
        
        # Add assistant response
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": response.answer,
            "sources": response.sources
        })

if __name__ == "__main__":
    main()