"""
Main Streamlit App with RAG Pipeline and Chatbot
Complete solution for document processing and conversational AI
"""

import streamlit as st
import sys
from pathlib import Path

# Add the services directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config.constants import UIConfig

def main():
    st.set_page_config(
        page_title=UIConfig.PAGE_TITLE,
        page_icon=UIConfig.PAGE_ICON,
        layout=UIConfig.LAYOUT
    )
    
    # Initialize session state for page navigation
    if "current_page" not in st.session_state:
        st.session_state.current_page = "🏠 Home"
    
    # Sidebar for navigation
    with st.sidebar:
        st.title("🤖 RAG System")
        st.markdown("---")
        
        # Navigation
        page = st.selectbox(
            "Choose a page:",
            ["🏠 Home", "📄 Document Processing", "💬 Chat with Documents"],
            index=["🏠 Home", "📄 Document Processing", "💬 Chat with Documents"].index(st.session_state.current_page)
        )
        
        # Update session state when page changes
        if page != st.session_state.current_page:
            st.session_state.current_page = page
            st.rerun()
        
        st.markdown("---")
        st.markdown("""
        ### About This System
        
        **RAG Pipeline** processes your documents:
        - PDF extraction (text + tables)
        - Semantic chunking
        - Vector embeddings
        - Storage in Qdrant database
        
        **Chatbot** lets you:
        - Ask questions in natural language
        - Get answers from your documents
        - See source attributions
        - Chat with conversation context
        """)
    
    # Page routing
    if st.session_state.current_page == "🏠 Home":
        show_home_page()
    elif st.session_state.current_page == "📄 Document Processing":
        show_document_processing()
    elif st.session_state.current_page == "💬 Chat with Documents":
        show_chatbot_interface()

def show_home_page():
    """Display the home page"""
    st.title("🤖 RAG System - Document Intelligence Platform")
    
    st.markdown("""
    Welcome to the complete RAG (Retrieval-Augmented Generation) system! 
    This platform allows you to process documents and chat with them using AI.
    """)
    
    # Feature overview
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
        
        if st.button("🚀 Start Processing Documents", type="primary"):
            st.session_state.current_page = "📄 Document Processing"
            st.rerun()
    
    with col2:
        st.header("💬 AI Chatbot")
        st.markdown("""
        **Chat with your documents using Gemini AI:**
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
        
        if st.button("💬 Start Chatting", type="primary"):
            st.session_state.current_page = "💬 Chat with Documents"
            st.rerun()
    
    # System overview diagram
    st.markdown("---")
    st.header("🏗️ System Architecture")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        **📄 PDF Processing**
        - Extract text & tables
        - Clean and structure data
        - Preserve metadata
        """)
    
    with col2:
        st.markdown("""
        **🧩 Chunking**
        - Semantic segmentation
        - Context preservation
        - Metadata enrichment
        """)
    
    with col3:
        st.markdown("""
        **🧠 Embeddings**
        - Vector representations
        - Multiple model options
        - Batch processing
        """)
    
    with col4:
        st.markdown("""
        **💬 Chat Interface**
        - RAG with Gemini AI
        - Source attribution
        - Conversation memory
        """)
    
    # Quick start guide
    with st.expander("🚀 Quick Start Guide"):
        st.markdown("""
        ### Getting Started in 3 Steps:
        
        1. **📄 Process Your Documents**
           - Go to 'Document Processing'
           - Upload a PDF file
           - Configure extraction and chunking settings
           - Run the complete RAG pipeline
        
        2. **🔑 Setup Gemini API**
           - Get your API key from Google AI Studio
           - Enter it in the chatbot configuration
           - Choose your collection name
        
        3. **💬 Start Chatting**
           - Go to 'Chat with Documents'
           - Initialize the chatbot
           - Ask questions about your documents!
        
        ### Prerequisites:
        - ✅ Qdrant running (Docker: `docker run -d -p 6333:6333 qdrant/qdrant`)
        - ✅ Google Gemini API key
        - ✅ Python environment with dependencies installed
        """)
    
    # System status
    st.markdown("---")
    st.header("📊 System Status")
    
    # Check system components
    status_col1, status_col2, status_col3 = st.columns(3)
    
    with status_col1:
        # Check Qdrant connection
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
        # Check embedding service
        try:
            from services.embedding_service import EmbeddingService
            embedding_service = EmbeddingService(model_name="default")
            st.success("✅ Embeddings Ready")
        except Exception as e:
            st.error(f"❌ Embeddings: {str(e)[:50]}...")
    
    with status_col3:
        # Check Gemini API
        try:
            from config.constants import LLMConfig
            if LLMConfig.API_KEY:
                st.success("✅ Gemini API Key Set")
            else:
                st.warning("⚠️ Gemini API Key Missing")
        except Exception as e:
            st.error(f"❌ Gemini: {str(e)[:50]}...")

def show_document_processing():
    """Show the document processing interface"""
    # Import and run the original streamlit app
    try:
        # Import the pipeline processing functions directly
        import tempfile
        import os
        from pathlib import Path
        import time
        
        from services.pdf_extracting_service import PDFExtractor
        from services.chunking_service import ChunkingService
        from services.embedding_service import EmbeddingService
        from services.vector_store_service import VectorStoreService
        
        # Run the original pipeline interface
        run_document_pipeline()
        
    except ImportError as e:
        st.error(f"Could not load document processing interface: {e}")
        st.info("Please ensure all dependencies are installed")

def show_chatbot_interface():
    """Show the chatbot interface"""
    # Import and run the chatbot app
    try:
        from services.rag_chatbot_service import RAGChatbotService
        from services.vector_store_service import VectorStoreService
        from services.embedding_service import EmbeddingService
        from config.constants import DatabaseConfig, EmbeddingConfig, LLMConfig, ChatbotConfig
        
        # Run the chatbot interface
        run_chatbot_interface()
        
    except ImportError as e:
        st.error(f"Could not load chatbot interface: {e}")
        st.info("Please ensure all dependencies are installed")

if __name__ == "__main__":
    main()