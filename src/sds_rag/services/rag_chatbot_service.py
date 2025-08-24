"""
RAG Chatbot Service using LangChain and Gemini API
Provides conversational interface over vector database
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config.constants import LLMConfig, ChatbotConfig
from services.vector_store_service import VectorStoreService
from services.embedding_service import EmbeddingService
from services.chunking_service import DocumentChunk

logger = logging.getLogger(__name__)

@dataclass
class ChatMessage:
    """Represents a single chat message"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime
    sources: Optional[List[Dict[str, Any]]] = None

@dataclass
class ChatResponse:
    """Response from the RAG chatbot"""
    answer: str
    sources: List[Dict[str, Any]]
    confidence_score: float
    processing_time: float

class RAGChatbotService:
    """
    RAG Chatbot service that combines vector search with LLM generation
    """
    
    def __init__(
        self,
        vector_store: VectorStoreService,
        embedding_service: EmbeddingService,
        api_key: Optional[str] = None
    ):
        """
        Initialize the RAG chatbot service
        
        Args:
            vector_store: Vector store service for document retrieval
            embedding_service: Embedding service for query encoding
            api_key: Google API key (optional, will use config if not provided)
        """
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        
        # Initialize LLM
        self.api_key = api_key or LLMConfig.API_KEY
        if not self.api_key:
            raise ValueError("OpenRouter API key is required. Set OPENROUTER_API_KEY in .env file")
            
        self.llm = ChatOpenAI(
            model=LLMConfig.MODEL,
            temperature=LLMConfig.TEMPERATURE,
            max_tokens=LLMConfig.MAX_TOKENS,
            openai_api_key=self.api_key,
            openai_api_base=LLMConfig.BASE_URL
        )
        
        # Chat history
        self.chat_history: List[ChatMessage] = []
        self.enable_history = ChatbotConfig.ENABLE_HISTORY
        self.max_history = ChatbotConfig.MAX_HISTORY
        
        # RAG settings
        self.search_limit = ChatbotConfig.SEARCH_LIMIT
        self.similarity_threshold = ChatbotConfig.SIMILARITY_THRESHOLD
        
        # Create RAG prompt template
        self.rag_prompt = self._create_rag_prompt()
        
        logger.info(f"RAG Chatbot initialized with OpenRouter model {LLMConfig.MODEL}")
    
    def _create_rag_prompt(self) -> ChatPromptTemplate:
        """Create the RAG prompt template"""
        template = """You are a helpful AI assistant that answers questions based on the provided context from financial documents.

**Context from financial documents:**
{context}

**Chat History:**
{chat_history}

**Current Question:** {question}

**Instructions:**
1. Answer the question using ONLY the information provided in the context
2. If the context doesn't contain enough information to answer the question, say so clearly
3. Cite specific information from the context when possible
4. Be concise but thorough in your explanations
5. If referencing specific data or numbers, mention the document source when available

**Answer:**"""

        return ChatPromptTemplate.from_template(template)
    
    def _format_context(self, chunks: List[DocumentChunk]) -> str:
        """Format retrieved chunks into context string"""
        if not chunks:
            return "No relevant information found in the documents."
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source_info = f"[Document: {chunk.source_document}"
            if chunk.page_number:
                source_info += f", Page: {chunk.page_number}"
            if chunk.chunk_type:
                source_info += f", Type: {chunk.chunk_type}"
            source_info += "]"
            
            context_parts.append(f"{i}. {source_info}\n{chunk.content}")
        
        return "\n\n".join(context_parts)
    
    def _format_chat_history(self) -> str:
        """Format chat history for context"""
        if not self.enable_history or not self.chat_history:
            return "No previous conversation."
        
        # Get recent history based on max_history setting
        recent_history = self.chat_history[-self.max_history:] if self.max_history > 0 else []
        
        history_parts = []
        for msg in recent_history:
            role = "User" if msg.role == "user" else "Assistant"
            history_parts.append(f"{role}: {msg.content}")
        
        return "\n".join(history_parts)
    
    def _retrieve_relevant_chunks(self, query: str) -> Tuple[List[DocumentChunk], List[Dict[str, Any]]]:
        """
        Retrieve relevant document chunks for the query
        
        Args:
            query: User query
            
        Returns:
            Tuple of (chunks, source_metadata)
        """
        try:
            # Search vector store
            search_results = self.vector_store.search_by_text(
                query_text=query,
                embedding_service=self.embedding_service,
                limit=self.search_limit,
                score_threshold=self.similarity_threshold
            )
            
            chunks = []
            sources = []
            
            for result in search_results:
                payload = result['payload']
                
                # Create DocumentChunk from payload
                chunk = DocumentChunk(
                    chunk_id=payload.get('chunk_id', ''),
                    content=payload.get('content', ''),
                    chunk_type=payload.get('chunk_type', 'unknown'),
                    source_document=payload.get('source_document', ''),
                    page_number=payload.get('page_number'),
                    table_id=payload.get('table_id'),
                    context=payload.get('context')
                )
                chunks.append(chunk)
                
                # Create source metadata
                source = {
                    'chunk_id': payload.get('chunk_id', ''),
                    'document': payload.get('source_document', ''),
                    'page': payload.get('page_number', 'N/A'),
                    'type': payload.get('chunk_type', 'unknown'),
                    'similarity_score': result['score'],
                    'preview': payload.get('content', '')[:200] + "..." if len(payload.get('content', '')) > 200 else payload.get('content', '')
                }
                sources.append(source)
            
            return chunks, sources
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            return [], []
    
    def _generate_response(self, query: str, context: str, chat_history: str) -> str:
        """Generate response using LLM"""
        try:
            # Create the chain
            chain = (
                RunnablePassthrough.assign(
                    context=lambda x: context,
                    chat_history=lambda x: chat_history,
                    question=lambda x: query
                )
                | self.rag_prompt
                | self.llm
                | StrOutputParser()
            )
            
            # Generate response
            response = chain.invoke({"question": query})
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I'm sorry, I encountered an error while processing your question: {str(e)}"
    
    def chat(self, query: str) -> ChatResponse:
        """
        Process a chat query and return response with sources
        
        Args:
            query: User question/query
            
        Returns:
            ChatResponse with answer, sources, and metadata
        """
        start_time = datetime.now()
        
        try:
            # Retrieve relevant chunks
            chunks, sources = self._retrieve_relevant_chunks(query)
            
            # Format context and history
            context = self._format_context(chunks)
            chat_history = self._format_chat_history()
            
            # Generate response
            answer = self._generate_response(query, context, chat_history)
            
            # Calculate confidence score (based on similarity scores of retrieved chunks)
            confidence_score = 0.0
            if sources:
                confidence_score = sum(s['similarity_score'] for s in sources) / len(sources)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Add to chat history
            if self.enable_history:
                self.chat_history.append(
                    ChatMessage(role="user", content=query, timestamp=start_time)
                )
                self.chat_history.append(
                    ChatMessage(
                        role="assistant", 
                        content=answer, 
                        timestamp=datetime.now(),
                        sources=sources
                    )
                )
                
                # Trim history if needed
                if len(self.chat_history) > self.max_history * 2:  # *2 for user+assistant pairs
                    self.chat_history = self.chat_history[-self.max_history * 2:]
            
            return ChatResponse(
                answer=answer,
                sources=sources,
                confidence_score=confidence_score,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error in chat processing: {e}")
            return ChatResponse(
                answer=f"I'm sorry, I encountered an error while processing your question: {str(e)}",
                sources=[],
                confidence_score=0.0,
                processing_time=(datetime.now() - start_time).total_seconds()
            )
    
    def clear_history(self):
        """Clear chat history"""
        self.chat_history.clear()
        logger.info("Chat history cleared")
    
    def get_chat_history(self) -> List[ChatMessage]:
        """Get current chat history"""
        return self.chat_history.copy()
    
    def set_search_parameters(self, limit: Optional[int] = None, threshold: Optional[float] = None):
        """
        Update search parameters
        
        Args:
            limit: Maximum number of chunks to retrieve
            threshold: Minimum similarity threshold
        """
        if limit is not None:
            self.search_limit = limit
        if threshold is not None:
            self.similarity_threshold = threshold
        
        logger.info(f"Search parameters updated: limit={self.search_limit}, threshold={self.similarity_threshold}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        return {
            'llm_model': LLMConfig.MODEL,
            'temperature': LLMConfig.TEMPERATURE,
            'max_tokens': LLMConfig.MAX_TOKENS,
            'search_limit': self.search_limit,
            'similarity_threshold': self.similarity_threshold,
            'history_enabled': self.enable_history,
            'max_history': self.max_history,
            'chat_history_count': len(self.chat_history),
            'vector_store_collection': self.vector_store.collection_name,
            'embedding_model': self.embedding_service.model_name
        }