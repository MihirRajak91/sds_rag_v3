# 🤖 RAG Document Intelligence System

A complete **Retrieval-Augmented Generation (RAG)** pipeline for processing documents and chatting with them using AI. Built with **Python**, **Streamlit**, **Qdrant**, and **OpenRouter/Gemma**.

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

### 🔍 **Document Processing**
- **PDF Text Extraction** - Extract narrative text from any PDF
- **Table Extraction** - Extract and parse tables with context preservation
- **Smart Chunking** - Semantic segmentation with overlap and metadata
- **Multi-format Support** - Handles financial reports, technical docs

### 🧠 **AI-Powered Search**
- **Vector Embeddings** - Multiple embedding models (default, high-quality, multilingual)
- **Semantic Search** - Find relevant content using natural language queries
- **Context Preservation** - Maintains document structure and relationships
- **Metadata Filtering** - Search by document, page, table, or content type

### 💬 **Intelligent Chatbot**
- **Free AI Model** - Uses Gemma 2B via OpenRouter (completely free)
- **Source Attribution** - Every answer shows which documents informed the response
- **Conversation Memory** - Maintains context across chat sessions
- **Real-time Processing** - Fast responses with processing time tracking

### 🚀 **Production Ready**
- **Scalable Architecture** - Modular services and clean data models
- **Configuration Management** - Centralized settings via environment variables
- **Error Handling** - Comprehensive error handling and logging
- **Web Interface** - Beautiful Streamlit UI for non-technical users

## 🏗️ Architecture

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   PDF Documents     │───▶│   RAG Pipeline      │───▶│   Chat Interface    │
│                     │    │                     │    │                     │
│ • Financial Reports │    │ • Text Extraction   │    │ • Natural Language  │
│ • Research Papers   │    │ • Table Processing  │    │ • Source Attribution│
│ • Technical Docs    │    │ • Smart Chunking    │    │ • Conversation Flow │
│ • Any PDF Content  │    │ • Vector Embeddings │    │ • Real-time Search  │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
                                       │
                           ┌─────────────────────┐
                           │   Vector Database   │
                           │                     │
                           │ • Qdrant Storage   │
                           │ • Similarity Search │
                           │ • Metadata Indexing │
                           │ • Scalable Retrieval│
                           └─────────────────────┘
```

### **Core Components**

- **PDF Extraction Service** - Uses `pypdf` and `tabula-py` for content extraction
- **Chunking Service** - Semantic segmentation with context preservation  
- **Embedding Service** - Vector embeddings using `sentence-transformers`
- **Vector Store Service** - Qdrant integration for similarity search
- **RAG Chatbot Service** - LangChain + OpenRouter for conversational AI
- **Streamlit UI** - Interactive web interface for document processing and chat

## Quick Start

### Prerequisites
- Python 3.10+
- Docker (for Qdrant)
- OpenRouter API key (free)

### 1. Clone & Install
```bash
git clone <repository-url>
cd sds_rag_v3
poetry install
```

### 2. Start Qdrant Database
```bash
docker run -d -p 6333:6333 qdrant/qdrant
```

### 3. Configure Environment
```bash
cp .env.example .env
# Edit .env with your OpenRouter API key
```

### 4. Launch Application
```bash
poetry run streamlit run src/sds_rag/ui/simple_main.py
```

### 5. Process Your First Document
1. Open http://localhost:8501
2. Go to "Document Processing"
3. Upload a PDF file
4. Click "Run Complete RAG Pipeline"
5. Go to "Chat with Documents" 
6. Initialize chatbot and start asking questions!

## Installation

### Using Poetry (Recommended)
```bash
# Clone repository
git clone <repository-url>
cd sds_rag_v3

# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

### Using pip
```bash
# Clone repository
git clone <repository-url>
cd sds_rag_v3

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Docker Setup
```bash
# Start Qdrant vector database
docker run -d -p 6333:6333 qdrant/qdrant

# Verify Qdrant is running
curl http://localhost:6333/health
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# ======================
# DATABASE SETTINGS
# ======================
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_TIMEOUT=10
QDRANT_DEFAULT_COLLECTION=financial_docs

# ======================
# EMBEDDING SETTINGS  
# ======================
EMBEDDING_MODEL=default
EMBEDDING_BATCH_SIZE=8
EMBEDDING_DEVICE=cpu

# Available models:
# - default: all-MiniLM-L6-v2 (384D, fast)
# - high_quality: all-mpnet-base-v2 (768D, slower)
# - multilingual: paraphrase-multilingual-MiniLM-L12-v2 (384D)

# ======================
# CHUNKING SETTINGS
# ======================
DEFAULT_CHUNK_SIZE=800
DEFAULT_CHUNK_OVERLAP=150
MIN_CHUNK_SIZE=100

# ======================
# LLM SETTINGS (OpenRouter)
# ======================
OPENROUTER_API_KEY=your_openrouter_api_key_here
LLM_MODEL=google/gemma-3n-e2b-it:free
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=1000
LLM_BASE_URL=https://openrouter.ai/api/v1

# ======================
# CHATBOT SETTINGS
# ======================
RAG_SEARCH_LIMIT=5
RAG_SIMILARITY_THRESHOLD=0.1
ENABLE_CHAT_HISTORY=true
MAX_CHAT_HISTORY=10
```

### Get OpenRouter API Key (Free)

1. Visit [OpenRouter.ai](https://openrouter.ai/)
2. Sign up for a free account
3. Get your API key from the dashboard
4. Add it to your `.env` file

### Configuration Classes

The system uses centralized configuration classes:

```python
from sds_rag.config import DatabaseConfig, EmbeddingConfig, LLMConfig

# Access configuration
print(f"Database: {DatabaseConfig.HOST}:{DatabaseConfig.PORT}")
print(f"Model: {LLMConfig.MODEL}")
print(f"Chunk Size: {ChunkingConfig.DEFAULT_SIZE}")
```

## Usage

### Document Processing Pipeline

```python
from sds_rag.services import (
    PDFExtractor,
    ChunkingService, 
    EmbeddingService,
    VectorStoreService
)

# 1. Extract PDF content
extractor = PDFExtractor("document.pdf")
full_text = extractor.extract_text()
tables = extractor.extract_tables()

# 2. Create semantic chunks
chunker = ChunkingService(chunk_size=800, overlap=150)
chunks = chunker.chunk_extracted_document(extractor.output_dir)

# 3. Generate embeddings
embedder = EmbeddingService(model_name="default")
embedded_chunks = embedder.embed_chunks(chunks)

# 4. Store in vector database
vector_store = VectorStoreService(collection_name="my_docs")
point_ids = vector_store.add_embedded_chunks(embedded_chunks)
```

### Chat with Documents

```python
from sds_rag.services import RAGChatbotService

# Initialize chatbot
chatbot = RAGChatbotService(
    vector_store=vector_store,
    embedding_service=embedder
)

# Ask questions
response = chatbot.chat("What was the revenue in Q3?")
print(f"Answer: {response.answer}")
print(f"Sources: {len(response.sources)}")
```

### Search Documents

```python
# Search for relevant content
results = vector_store.search_by_text(
    query_text="financial performance",
    embedding_service=embedder,
    limit=5
)

for result in results:
    print(f"Document: {result['payload']['source_document']}")
    print(f"Content: {result['payload']['content'][:200]}...")
```

## Use Cases

### Financial Document Analysis
- **Earnings Reports** - Extract revenue, expenses, and key metrics
- **10-Q/10-K Filings** - Analyze quarterly and annual reports
- **Investment Research** - Chat with research reports and analysis

### Business Intelligence
- **Contract Analysis** - Extract key terms and obligations
- **Policy Documents** - Navigate regulatory and compliance docs
- **Market Reports** - Analyze industry trends and forecasts

#### PDFExtractor
```python
class PDFExtractor:
    def extract_text() -> str
    def extract_tables(**kwargs) -> List[pd.DataFrame]
    def save_extraction(text: str, tables: List[pd.DataFrame])
```

#### ChunkingService
```python
class ChunkingService:
    def chunk_extracted_document(directory: Path) -> List[DocumentChunk]
    def chunk_text(text: str) -> List[DocumentChunk]
```

#### EmbeddingService
```python
class EmbeddingService:
    def embed_chunks(chunks: List[DocumentChunk]) -> List[Dict]
    def embed_single_text(text: str) -> np.ndarray
```

#### VectorStoreService
```python
class VectorStoreService:
    def add_embedded_chunks(chunks: List[Dict]) -> List[str]
    def search_by_text(query: str, **kwargs) -> List[Dict]
    def search_similar(query_embedding: np.ndarray, **kwargs) -> List[Dict]
```

#### RAGChatbotService
```python
class RAGChatbotService:
    def chat(query: str) -> ChatResponse
    def clear_history()
    def get_system_info() -> Dict[str, Any]
```

### Data Models

#### DocumentChunk
```python
@dataclass
class DocumentChunk:
    chunk_id: str
    content: str
    chunk_type: str  # 'text', 'table', 'mixed'
    source_document: str
    page_number: Optional[int] = None
    table_id: Optional[str] = None
    context: Optional[str] = None
```

#### ChatResponse
```python
@dataclass
class ChatResponse:
    answer: str
    sources: List[Dict[str, Any]]
    confidence_score: float
    processing_time: float
```

## 🧪 Testing

### Run Tests
```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src/sds_rag

# Run specific test file
poetry run pytest tests/test_chunking_service.py
```

### Test Configuration
```bash
# Test configuration loading
poetry run python test_config.py

# Test chatbot functionality  
poetry run python test_chatbot.py
```

## 📊 Performance

### Benchmarks

| Component | Processing Time | Memory Usage |
|-----------|----------------|--------------|
| PDF Extraction (10 pages) | ~2-5 seconds | ~50MB |
| Chunking (1000 chunks) | ~1-2 seconds | ~100MB |
| Embeddings (1000 chunks) | ~30-60 seconds | ~200MB |
| Vector Search (5 results) | ~100-200ms | ~10MB |
| Chat Response | ~2-5 seconds | ~50MB |

### Optimization Tips

- Use `default` embedding model for speed, `high_quality` for accuracy
- Adjust chunk size based on your documents (larger for technical docs)
- Use GPU if available: `EMBEDDING_DEVICE=cuda`
- Increase batch size for faster embedding: `EMBEDDING_BATCH_SIZE=32`

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd sds_rag_v3

# Install development dependencies
poetry install --with dev

# Install pre-commit hooks
pre-commit install

# Run tests
poetry run pytest
```

### Code Style
```bash
# Format code
poetry run black src/ tests/

# Check linting
poetry run flake8 src/ tests/

# Type checking
poetry run mypy src/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **LangChain** - RAG framework and LLM integration
- **Qdrant** - Vector database for similarity search
- **Sentence Transformers** - Text embedding models
- **Streamlit** - Interactive web interface
- **OpenRouter** - Free AI model access
- **Gemma** - Google's open-source language model

## Support

- **Documentation**: [Link to docs]
- **Issues**: [GitHub Issues](https://github.com/your-org/sds_rag_v3/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/sds_rag_v3/discussions)
- **Email**: support@yourcompany.com

---

**Built with ❤️ for document intelligence and conversational AI.**

🚀 **Ready to turn your documents into intelligent, searchable knowledge bases?** Get started now!