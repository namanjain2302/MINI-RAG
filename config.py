"""Configuration settings for the RAG Assistant."""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
HF_API_KEY = os.getenv("HF_API_KEY")

# Model configurations
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/Llama-3.2-1B-Instruct")

# Chunking parameters
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

# Retrieval parameters
TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", 3))

# Chroma DB settings
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "document_collection")

# Document directory
DATA_DIR = os.getenv("DATA_DIR", "./data")

# Validation
if not HF_API_KEY:
    print("⚠  Warning: HF_API_KEY not found in .env file. Some models may not work.")
