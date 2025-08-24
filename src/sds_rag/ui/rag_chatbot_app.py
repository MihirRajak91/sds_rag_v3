"""
RAG Chatbot Streamlit Interface
Chat with your documents using RAG + Gemini
"""

import streamlit as st
import time
from datetime import datetime
from typing import List, Dict, Any
import sys
from pathlib import Path

# Add the services directory to path
sys.path.append(str(Path(__file__).parent.parent))

from services.rag_chatbot_service import RAGChatbotService, ChatMessage
from services.vector_store_service import VectorStoreService
from services.embedding_service import EmbeddingService
from config.constants import DatabaseConfig, EmbeddingConfig, LLMConfig, ChatbotConfig

def initialize_chatbot_session():
    """Initialize chatbot session state"""
    if "chatbot_service" not in st.session_state:
        st.session_state.chatbot_service = None
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "current_collection" not in st.session_state:
        st.session_state.current_collection = DatabaseConfig.DEFAULT_COLLECTION

def setup_chatbot_service(collection_name: str, embedding_model: str = "default"):
    """Setup or reinitialize the chatbot service"""
    try:
        # Initialize embedding service
        embedding_service = EmbeddingService(
            model_name=embedding_model,
            device=EmbeddingConfig.DEVICE,
            batch_size=EmbeddingConfig.BATCH_SIZE
        )
        
        # Initialize vector store
        vector_store = VectorStoreService(
            host=DatabaseConfig.HOST,
            port=DatabaseConfig.PORT,
            collection_name=collection_name,
            embedding_dimension=embedding_service.embedding_dimension
        )
        
        # Initialize chatbot service
        chatbot_service = RAGChatbotService(
            vector_store=vector_store,
            embedding_service=embedding_service
        )
        
        st.session_state.chatbot_service = chatbot_service
        st.session_state.current_collection = collection_name
        
        return True, "Chatbot initialized successfully!"
        
    except Exception as e:
        return False, f"Failed to initialize chatbot: {str(e)}"

def main():
    st.set_page_config(
        page_title="RAG Chatbot",
        page_icon="💬",
        layout="wide"
    )
    
    initialize_chatbot_session()
    
    st.title("💬 RAG Chatbot")
    st.markdown("Chat with your documents using RAG + Gemini AI")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔧 Chatbot Configuration")
        
        # Collection selection
        st.subheader("📊 Document Collection")
        collection_name = st.text_input(
            "Collection Name:",
            value=st.session_state.current_collection,
            help="Enter the Qdrant collection name containing your documents"
        )
        
        # Embedding model selection
        embedding_model = st.selectbox(
            "Embedding Model:",
            ["default", "high_quality", "multilingual"],
            index=0,
            help="Choose embedding model (must match documents)"
        )
        
        # API Key configuration
        st.subheader("🔑 API Configuration")
        api_key_input = st.text_input(
            "Gemini API Key:",
            type="password",
            value=LLMConfig.API_KEY,
            help="Enter your Google Gemini API key"
        )
        
        # Update API key if provided
        if api_key_input and api_key_input != LLMConfig.API_KEY:
            import os
            os.environ['GOOGLE_API_KEY'] = api_key_input
        
        # Initialize/Reinitialize chatbot
        if st.button("🚀 Initialize Chatbot", type="primary"):
            if not api_key_input:
                st.error("❌ Please provide a Gemini API key")
            else:
                with st.spinner("Initializing chatbot..."):
                    success, message = setup_chatbot_service(collection_name, embedding_model)
                    if success:
                        st.success(message)
                        st.session_state.chat_messages = []  # Clear chat history
                    else:
                        st.error(message)
        
        # Chatbot settings
        if st.session_state.chatbot_service:
            st.subheader("⚙️ Chat Settings")
            
            # Search parameters
            search_limit = st.slider(
                "Sources per query:", 
                1, 10, ChatbotConfig.SEARCH_LIMIT, 
                help="Number of document chunks to retrieve"
            )
            
            similarity_threshold = st.slider(
                "Similarity threshold:",
                0.0, 1.0, ChatbotConfig.SIMILARITY_THRESHOLD, 0.05,
                help="Minimum similarity score for relevant chunks"
            )
            
            # Update parameters if changed
            if st.button("Update Settings"):
                st.session_state.chatbot_service.set_search_parameters(
                    limit=search_limit,
                    threshold=similarity_threshold
                )
                st.success("Settings updated!")
            
            # Clear chat history
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chatbot_service.clear_history()
                st.session_state.chat_messages = []
                st.success("Chat history cleared!")
            
            # System info
            with st.expander("📋 System Information"):
                info = st.session_state.chatbot_service.get_system_info()
                for key, value in info.items():
                    st.text(f"{key}: {value}")
    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat Interface")
        
        if not st.session_state.chatbot_service:
            st.info("👈 Please initialize the chatbot using the sidebar configuration")
            st.markdown("""
            ### Getting Started:
            1. **Set Collection Name**: Enter the name of your Qdrant collection
            2. **Choose Embedding Model**: Select the same model used for your documents
            3. **Add API Key**: Enter your Google Gemini API key
            4. **Initialize**: Click 'Initialize Chatbot' to start
            """)
        else:
            # Chat messages display
            chat_container = st.container()
            
            with chat_container:
                # Display chat history
                for message in st.session_state.chat_messages:
                    with st.chat_message(message["role"]):
                        st.write(message["content"])
                        
                        # Show sources for assistant messages
                        if message["role"] == "assistant" and message.get("sources"):
                            with st.expander(f"📚 Sources ({len(message['sources'])})", expanded=False):
                                for i, source in enumerate(message["sources"], 1):
                                    st.markdown(f"""
                                    **Source {i}:** {source['document']} 
                                    (Page {source['page']}, {source['type']}, Score: {source['similarity_score']:.3f})
                                    
                                    *Preview:* {source['preview']}
                                    """)
                                    st.divider()
            
            # Chat input
            user_input = st.chat_input("Ask a question about your documents...")
            
            if user_input:
                # Add user message to chat
                st.session_state.chat_messages.append({
                    "role": "user", 
                    "content": user_input
                })
                
                # Display user message
                with st.chat_message("user"):
                    st.write(user_input)
                
                # Get chatbot response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = st.session_state.chatbot_service.chat(user_input)
                    
                    # Display response
                    st.write(response.answer)
                    
                    # Show sources
                    if response.sources:
                        with st.expander(f"📚 Sources ({len(response.sources)})", expanded=False):
                            for i, source in enumerate(response.sources, 1):
                                st.markdown(f"""
                                **Source {i}:** {source['document']} 
                                (Page {source['page']}, {source['type']}, Score: {source['similarity_score']:.3f})
                                
                                *Preview:* {source['preview']}
                                """)
                                st.divider()
                    
                    # Show metadata
                    st.caption(f"⏱️ Response time: {response.processing_time:.2f}s | "
                              f"📊 Confidence: {response.confidence_score:.2f} | "
                              f"🔍 Sources: {len(response.sources)}")
                
                # Add assistant response to chat history
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": response.answer,
                    "sources": response.sources,
                    "metadata": {
                        "processing_time": response.processing_time,
                        "confidence_score": response.confidence_score
                    }
                })
    
    with col2:
        st.header("📋 Chat Information")
        
        if st.session_state.chatbot_service:
            info = st.session_state.chatbot_service.get_system_info()
            
            # Current session info
            st.subheader("📊 Current Session")
            col2_1, col2_2 = st.columns(2)
            with col2_1:
                st.metric("Collection", info['vector_store_collection'])
                st.metric("Chat Messages", len(st.session_state.chat_messages))
            with col2_2:
                st.metric("LLM Model", info['llm_model'])
                st.metric("Search Limit", info['search_limit'])
            
            # Recent activity
            st.subheader("🕐 Recent Activity")
            if st.session_state.chat_messages:
                recent_messages = st.session_state.chat_messages[-3:]  # Last 3 messages
                for msg in recent_messages:
                    role_emoji = "👤" if msg["role"] == "user" else "🤖"
                    content_preview = msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
                    st.text(f"{role_emoji} {content_preview}")
            else:
                st.info("No recent activity")
        
        # Help section
        st.subheader("💡 Tips")
        st.markdown("""
        **Effective Questions:**
        - "What was the revenue in Q3?"
        - "Summarize the risk factors"
        - "Show me operating expenses"
        - "What are the key metrics?"
        
        **Features:**
        - 📚 **Source Attribution**: See which documents informed each answer
        - 🔍 **Similarity Scores**: Understand relevance of sources
        - ⏱️ **Response Times**: Monitor performance
        - 📊 **Confidence Scores**: Gauge answer quality
        
        **Pro Tips:**
        - Be specific in your questions
        - Use domain terminology
        - Ask follow-up questions for clarification
        - Check sources for verification
        """)
        
        # Example queries
        if st.session_state.chatbot_service:
            st.subheader("🎯 Quick Questions")
            example_queries = [
                "What was the total revenue?",
                "Summarize the main risks",
                "Show operating expenses",
                "Key financial metrics",
                "Management discussion"
            ]
            
            for query in example_queries:
                if st.button(f"💬 {query}", key=f"example_{query}"):
                    # Simulate user input
                    st.session_state.chat_messages.append({
                        "role": "user", 
                        "content": query
                    })
                    st.rerun()

def display_connection_status():
    """Display connection status to vector database"""
    try:
        vector_store = VectorStoreService(
            host=DatabaseConfig.HOST,
            port=DatabaseConfig.PORT,
            collection_name="test"
        )
        # Try to connect
        vector_store.client.get_collections()
        return True, "✅ Connected to Qdrant"
    except Exception as e:
        return False, f"❌ Connection failed: {str(e)}"

if __name__ == "__main__":
    main()