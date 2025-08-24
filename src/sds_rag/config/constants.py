"""
Constants and configuration management for the RAG pipeline.
Loads settings from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def str_to_bool(value: str) -> bool:
    """Convert string to boolean"""
    return value.lower() in ('true', '1', 'yes', 'on')

def get_env_list(key: str, default: List[str] = None) -> List[str]:
    """Get list from environment variable (comma-separated)"""
    if default is None:
        default = []
    value = os.getenv(key, '')
    return [item.strip() for item in value.split(',') if item.strip()] if value else default

# ======================
# PROJECT PATHS
# ======================
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
LOGS_DIR.mkdir(exist_ok=True)

# ======================
# DATABASE CONFIGURATION
# ======================
class DatabaseConfig:
    HOST = os.getenv('QDRANT_HOST', 'localhost')
    PORT = int(os.getenv('QDRANT_PORT', '6333'))
    TIMEOUT = int(os.getenv('QDRANT_TIMEOUT', '10'))
    DEFAULT_COLLECTION = os.getenv('QDRANT_DEFAULT_COLLECTION', 'financial_docs')
    
    # Connection settings
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0

# ======================
# EMBEDDING CONFIGURATION
# ======================
class EmbeddingConfig:
    MODEL = os.getenv('EMBEDDING_MODEL', 'default')
    BATCH_SIZE = int(os.getenv('EMBEDDING_BATCH_SIZE', '8'))
    DEVICE = os.getenv('EMBEDDING_DEVICE', 'auto')
    CACHE_EMBEDDINGS = str_to_bool(os.getenv('CACHE_EMBEDDINGS', 'true'))
    
    # Model configurations
    MODEL_CONFIGS = {
        'default': {
            'name': 'all-MiniLM-L6-v2',
            'dimension': 384,
            'description': 'Fast and efficient, good for general use'
        },
        'high_quality': {
            'name': 'all-mpnet-base-v2', 
            'dimension': 768,
            'description': 'Higher quality embeddings, slower processing'
        },
        'multilingual': {
            'name': 'paraphrase-multilingual-MiniLM-L12-v2',
            'dimension': 384,
            'description': 'Supports multiple languages'
        },
        'financial': {
            'name': 'sentence-transformers/all-MiniLM-L6-v2',
            'dimension': 384,
            'description': 'Good general model for financial documents'
        }
    }

# ======================
# CHUNKING CONFIGURATION
# ======================
class ChunkingConfig:
    DEFAULT_SIZE = int(os.getenv('DEFAULT_CHUNK_SIZE', '800'))
    DEFAULT_OVERLAP = int(os.getenv('DEFAULT_CHUNK_OVERLAP', '150'))
    MIN_SIZE = int(os.getenv('MIN_CHUNK_SIZE', '100'))
    
    # Advanced chunking settings
    RESPECT_SENTENCE_BOUNDARIES = True
    RESPECT_PARAGRAPH_BOUNDARIES = True

# ======================
# PDF PROCESSING CONFIGURATION
# ======================
class PDFConfig:
    EXTRACT_TEXT = str_to_bool(os.getenv('PDF_EXTRACT_TEXT', 'true'))
    EXTRACT_TABLES = str_to_bool(os.getenv('PDF_EXTRACT_TABLES', 'true'))
    DEFAULT_PAGES = os.getenv('PDF_DEFAULT_PAGES', 'all')
    OUTPUT_DIR = PROJECT_ROOT / os.getenv('PDF_OUTPUT_DIR', 'extracted_documents')
    TEMP_DIR = PROJECT_ROOT / os.getenv('PDF_TEMP_DIR', 'temp')
    
    # Ensure directories exist
    OUTPUT_DIR.mkdir(exist_ok=True)
    TEMP_DIR.mkdir(exist_ok=True)
    
    # Tabula settings
    TABULA_OPTIONS = {
        "multiple_tables": True,
        "pandas_options": {"header": None}
    }

# ======================
# UI CONFIGURATION
# ======================
class UIConfig:
    HOST = os.getenv('STREAMLIT_HOST', 'localhost')
    PORT = int(os.getenv('STREAMLIT_PORT', '8501'))
    SHOW_PROGRESS = str_to_bool(os.getenv('UI_SHOW_PROGRESS', 'true'))
    MAX_UPLOAD_SIZE = int(os.getenv('UI_MAX_UPLOAD_SIZE', '100'))  # MB
    
    # UI defaults
    DEFAULT_CHUNK_SIZE_RANGE = (400, 1500)
    DEFAULT_OVERLAP_RANGE = (50, 300)
    
    # Theme settings
    PAGE_TITLE = "RAG Pipeline"
    PAGE_ICON = "🤖"
    LAYOUT = "wide"

# ======================
# SEARCH CONFIGURATION
# ======================
class SearchConfig:
    DEFAULT_LIMIT = int(os.getenv('DEFAULT_SEARCH_LIMIT', '10'))
    DEFAULT_THRESHOLD = float(os.getenv('DEFAULT_SIMILARITY_THRESHOLD', '0.1'))
    ENABLE_FILTERED_SEARCH = str_to_bool(os.getenv('ENABLE_FILTERED_SEARCH', 'true'))
    
    # Search result limits
    MAX_RESULTS = 100
    MIN_SIMILARITY_SCORE = 0.0
    MAX_SIMILARITY_SCORE = 1.0

# ======================
# LOGGING CONFIGURATION
# ======================
class LoggingConfig:
    LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    FILE = LOGS_DIR / os.getenv('LOG_FILE', 'rag_pipeline.log')
    FORMAT = os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Rotation settings
    MAX_BYTES = 10 * 1024 * 1024  # 10MB
    BACKUP_COUNT = 5

# ======================
# PERFORMANCE CONFIGURATION
# ======================
class PerformanceConfig:
    MAX_WORKERS = int(os.getenv('MAX_WORKERS', '4'))
    CACHE_DIR = PROJECT_ROOT / os.getenv('CACHE_DIR', '.cache')
    
    # Ensure cache directory exists
    CACHE_DIR.mkdir(exist_ok=True)
    
    # Memory management
    BATCH_PROCESSING = True
    CLEANUP_TEMP_FILES = True

# ======================
# SECURITY CONFIGURATION
# ======================
class SecurityConfig:
    API_KEY = os.getenv('API_KEY')
    ENCRYPT_VECTORS = str_to_bool(os.getenv('ENCRYPT_VECTORS', 'false'))
    ALLOWED_ORIGINS = get_env_list('ALLOWED_ORIGINS', ['localhost', '127.0.0.1'])
    
    # File security
    ALLOWED_EXTENSIONS = ['.pdf']
    MAX_FILE_SIZE = UIConfig.MAX_UPLOAD_SIZE * 1024 * 1024  # Convert MB to bytes

# ======================
# LLM CONFIGURATION (GEMINI)
# ======================
class LLMConfig:
    """LLM configuration for OpenRouter"""
    API_KEY = os.getenv('OPENROUTER_API_KEY', '')
    MODEL = os.getenv('LLM_MODEL', 'google/gemma-3n-e2b-it:free')
    TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', '0.3'))
    MAX_TOKENS = int(os.getenv('LLM_MAX_TOKENS', '1000'))
    BASE_URL = os.getenv('LLM_BASE_URL', 'https://openrouter.ai/api/v1')

# ======================
# CHATBOT CONFIGURATION
# ======================
class ChatbotConfig:
    """RAG Chatbot configuration"""
    SEARCH_LIMIT = int(os.getenv('RAG_SEARCH_LIMIT', '5'))
    SIMILARITY_THRESHOLD = float(os.getenv('RAG_SIMILARITY_THRESHOLD', '0.3'))
    ENABLE_HISTORY = str_to_bool(os.getenv('ENABLE_CHAT_HISTORY', 'true'))
    MAX_HISTORY = int(os.getenv('MAX_CHAT_HISTORY', '10'))

# ======================
# VALIDATION FUNCTIONS
# ======================
def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Check required directories
    if not PROJECT_ROOT.exists():
        errors.append(f"Project root not found: {PROJECT_ROOT}")
    
    # Check database connection settings
    if not (1 <= DatabaseConfig.PORT <= 65535):
        errors.append(f"Invalid database port: {DatabaseConfig.PORT}")
    
    # Check embedding model
    if EmbeddingConfig.MODEL not in EmbeddingConfig.MODEL_CONFIGS:
        errors.append(f"Unknown embedding model: {EmbeddingConfig.MODEL}")
    
    # Check chunk sizes
    if ChunkingConfig.DEFAULT_SIZE < ChunkingConfig.MIN_SIZE:
        errors.append(f"Default chunk size ({ChunkingConfig.DEFAULT_SIZE}) cannot be less than minimum ({ChunkingConfig.MIN_SIZE})")
    
    if ChunkingConfig.DEFAULT_OVERLAP >= ChunkingConfig.DEFAULT_SIZE:
        errors.append(f"Chunk overlap ({ChunkingConfig.DEFAULT_OVERLAP}) cannot be greater than chunk size ({ChunkingConfig.DEFAULT_SIZE})")
    
    return errors

# ======================
# CONVENIENCE FUNCTIONS
# ======================
def get_all_config() -> Dict[str, Any]:
    """Get all configuration as a dictionary"""
    return {
        'database': {
            'host': DatabaseConfig.HOST,
            'port': DatabaseConfig.PORT,
            'timeout': DatabaseConfig.TIMEOUT,
            'default_collection': DatabaseConfig.DEFAULT_COLLECTION
        },
        'embedding': {
            'model': EmbeddingConfig.MODEL,
            'batch_size': EmbeddingConfig.BATCH_SIZE,
            'device': EmbeddingConfig.DEVICE,
            'dimension': EmbeddingConfig.MODEL_CONFIGS[EmbeddingConfig.MODEL]['dimension']
        },
        'chunking': {
            'default_size': ChunkingConfig.DEFAULT_SIZE,
            'default_overlap': ChunkingConfig.DEFAULT_OVERLAP,
            'min_size': ChunkingConfig.MIN_SIZE
        },
        'pdf': {
            'extract_text': PDFConfig.EXTRACT_TEXT,
            'extract_tables': PDFConfig.EXTRACT_TABLES,
            'output_dir': str(PDFConfig.OUTPUT_DIR)
        },
        'search': {
            'default_limit': SearchConfig.DEFAULT_LIMIT,
            'default_threshold': SearchConfig.DEFAULT_THRESHOLD
        }
    }

def print_config():
    """Print current configuration (for debugging)"""
    config = get_all_config()
    print("=== RAG Pipeline Configuration ===")
    for section, settings in config.items():
        print(f"\n[{section.upper()}]")
        for key, value in settings.items():
            print(f"  {key}: {value}")

# Validate configuration on import
config_errors = validate_config()
if config_errors:
    print("Configuration errors found:")
    for error in config_errors:
        print(f"  - {error}")
    print("Please check your .env file and fix these issues.")

# Export all config classes for easy importing
__all__ = [
    'DatabaseConfig',
    'EmbeddingConfig', 
    'ChunkingConfig',
    'PDFConfig',
    'UIConfig',
    'SearchConfig',
    'LoggingConfig',
    'PerformanceConfig',
    'SecurityConfig',
    'LLMConfig',
    'ChatbotConfig',
    'get_all_config',
    'print_config',
    'validate_config'
]