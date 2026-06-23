import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Project root directory (backend/app/core/config.py -> backend/app/core -> backend/app -> backend -> root)
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

# Storage Paths
UPLOAD_DIR = os.getenv("UPLOAD_DIR", str(BASE_DIR / "data" / "uploads"))
VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", str(BASE_DIR / "data" / "vectorstore"))

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")

# LLM Provider Configuration
DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "openai").lower()  # 'openai' or 'mistral'
DEFAULT_OPENAI_MODEL = os.getenv("DEFAULT_OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_MISTRAL_MODEL = os.getenv("DEFAULT_MISTRAL_MODEL", "mistral-large-latest")

# RAG Chunking Parameters
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Advanced Retrieval Configurations
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "10"))       # Number of chunks retrieved initially by dense/sparse
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))       # Number of chunks passed to the LLM after reranking

def get_api_key(provider: str, header_key: str = None) -> str:
    """
    Get the API key for the specified provider.
    Priority:
    1. Header key (passed from UI settings)
    2. Environment variable
    """
    provider = provider.lower()
    if header_key:
        return header_key
        
    if provider == "openai":
        return os.getenv("OPENAI_API_KEY", "") or OPENAI_API_KEY
    elif provider == "mistral":
        return os.getenv("MISTRAL_API_KEY", "") or MISTRAL_API_KEY
    return ""
