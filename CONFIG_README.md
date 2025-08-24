# Configuration Management

Centralized configuration system for the RAG pipeline using environment variables and constants.

## 📁 **Configuration Files**

### **`.env`** - Your Settings
Main configuration file where you customize all settings:
```bash
# Database
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Embedding Model  
EMBEDDING_MODEL=default
EMBEDDING_DEVICE=cpu

# Chunking
DEFAULT_CHUNK_SIZE=800
DEFAULT_CHUNK_OVERLAP=150
```

### **`.env.example`** - Template
Copy this to `.env` to get started with default settings.

### **`src/sds_rag/config/constants.py`** - Configuration Logic
Python module that loads `.env` values and provides structured configuration classes.

## 🔧 **How to Use**

### **1. Basic Usage**
```python
from sds_rag.config import DatabaseConfig, EmbeddingConfig

# Use in your services
vector_store = VectorStoreService(
    host=DatabaseConfig.HOST,
    port=DatabaseConfig.PORT,
    collection_name=DatabaseConfig.DEFAULT_COLLECTION
)

embedder = EmbeddingService(
    model_name=EmbeddingConfig.MODEL,
    device=EmbeddingConfig.DEVICE
)
```

### **2. Change Settings**
Edit `.env` file:
```bash
# Switch to high-quality embeddings
EMBEDDING_MODEL=high_quality

# Use larger chunks
DEFAULT_CHUNK_SIZE=1200

# Connect to remote Qdrant
QDRANT_HOST=my-server.com
QDRANT_PORT=6334
```

### **3. View Current Config**
```python
from sds_rag.config import print_config, get_all_config

# Print all settings
print_config()

# Get as dictionary
config = get_all_config()
```

## ⚙️ **Configuration Sections**

### **Database Settings**
```bash
QDRANT_HOST=localhost           # Qdrant server host
QDRANT_PORT=6333               # Qdrant server port  
QDRANT_TIMEOUT=10              # Connection timeout
QDRANT_DEFAULT_COLLECTION=financial_docs  # Default collection name
```

### **Embedding Settings**
```bash
EMBEDDING_MODEL=default        # Model: default, high_quality, multilingual
EMBEDDING_BATCH_SIZE=8         # Processing batch size
EMBEDDING_DEVICE=cpu          # Device: cpu, cuda, mps
```

**Available Models:**
- `default`: all-MiniLM-L6-v2 (384D, fast)
- `high_quality`: all-mpnet-base-v2 (768D, slower)
- `multilingual`: paraphrase-multilingual-MiniLM-L12-v2 (384D)

### **Chunking Settings**
```bash
DEFAULT_CHUNK_SIZE=800         # Characters per chunk
DEFAULT_CHUNK_OVERLAP=150      # Overlap between chunks
MIN_CHUNK_SIZE=100            # Minimum chunk size
```

### **PDF Processing**
```bash
PDF_EXTRACT_TEXT=true          # Extract narrative text
PDF_EXTRACT_TABLES=true        # Extract tables
PDF_DEFAULT_PAGES=all          # Pages to process
PDF_OUTPUT_DIR=extracted_documents  # Output directory
```

### **UI Settings**
```bash
STREAMLIT_HOST=localhost       # Streamlit host
STREAMLIT_PORT=8501           # Streamlit port
UI_MAX_UPLOAD_SIZE=100        # Max upload size (MB)
```

### **Search Settings**
```bash
DEFAULT_SEARCH_LIMIT=10        # Default result limit
DEFAULT_SIMILARITY_THRESHOLD=0.1  # Min similarity score
ENABLE_FILTERED_SEARCH=true    # Enable metadata filtering
```

### **Logging**
```bash
LOG_LEVEL=INFO                # Log level
LOG_FILE=logs/rag_pipeline.log  # Log file path
```

### **Performance**
```bash
MAX_WORKERS=4                 # Max concurrent workers
CACHE_EMBEDDINGS=true         # Cache embeddings
CACHE_DIR=.cache             # Cache directory
```

## 🎯 **Configuration Classes**

### **DatabaseConfig**
```python
DatabaseConfig.HOST           # localhost
DatabaseConfig.PORT           # 6333
DatabaseConfig.DEFAULT_COLLECTION  # financial_docs
```

### **EmbeddingConfig**
```python
EmbeddingConfig.MODEL         # default
EmbeddingConfig.DEVICE        # cpu
EmbeddingConfig.BATCH_SIZE    # 8
EmbeddingConfig.MODEL_CONFIGS # Dict of available models
```

### **ChunkingConfig**
```python
ChunkingConfig.DEFAULT_SIZE   # 800
ChunkingConfig.DEFAULT_OVERLAP # 150
ChunkingConfig.MIN_SIZE       # 100
```

### **All Available Classes**
- `DatabaseConfig` - Vector database settings
- `EmbeddingConfig` - Embedding model settings
- `ChunkingConfig` - Text chunking settings
- `PDFConfig` - PDF processing settings
- `UIConfig` - Streamlit UI settings
- `SearchConfig` - Search behavior settings
- `LoggingConfig` - Logging configuration
- `PerformanceConfig` - Performance tuning
- `SecurityConfig` - Security settings

## 🚀 **Quick Configuration Changes**

### **High-Performance Setup**
```bash
# .env
EMBEDDING_MODEL=default
EMBEDDING_DEVICE=cuda          # If you have GPU
EMBEDDING_BATCH_SIZE=32        # Larger batches
DEFAULT_CHUNK_SIZE=1000        # Larger chunks
MAX_WORKERS=8                  # More workers
```

### **High-Quality Setup**
```bash
# .env  
EMBEDDING_MODEL=high_quality   # Better embeddings
DEFAULT_CHUNK_SIZE=600         # Smaller chunks for precision
DEFAULT_CHUNK_OVERLAP=200      # More overlap
DEFAULT_SIMILARITY_THRESHOLD=0.2  # Higher threshold
```

### **Production Setup**
```bash
# .env
QDRANT_HOST=production-qdrant.company.com
QDRANT_PORT=6333
QDRANT_DEFAULT_COLLECTION=production_docs
LOG_LEVEL=WARNING
CACHE_EMBEDDINGS=true
```

### **Development Setup**
```bash
# .env
LOG_LEVEL=DEBUG
DEFAULT_SEARCH_LIMIT=3         # Fewer results for testing
UI_SHOW_PROGRESS=true         # Show detailed progress
```

## 🔍 **Validation**

The configuration system automatically validates settings on startup:

```python
from sds_rag.config import validate_config

errors = validate_config()
if errors:
    for error in errors:
        print(f"Config error: {error}")
```

**Common Validation Checks:**
- Port numbers in valid range
- Embedding model exists
- Chunk size > minimum size  
- Overlap < chunk size
- Required directories exist

## 💡 **Best Practices**

1. **Keep `.env` Local**: Never commit `.env` to git (already in `.gitignore`)
2. **Use `.env.example`**: Document all available settings
3. **Validate Early**: Check config on application startup
4. **Environment-Specific**: Use different `.env` files for dev/prod
5. **Secure Secrets**: Keep API keys and credentials in `.env`

## 🐛 **Troubleshooting**

### **Config Not Loading**
```bash
# Check if .env exists
ls -la .env

# Test configuration
poetry run python test_config.py
```

### **Invalid Settings**
```python
from sds_rag.config import validate_config
errors = validate_config()  # Shows validation errors
```

### **Reset to Defaults**
```bash
# Copy example file
cp .env.example .env
```

**Your RAG pipeline settings are now centralized and easily configurable!** 🎉