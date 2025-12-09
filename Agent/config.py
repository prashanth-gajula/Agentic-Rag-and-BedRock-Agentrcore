"""
Configuration file for the Agentic RAG system.
Contains all constants and environment setup.
"""
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Verify keys
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables")

# Paths
DATA_DIR = Path("Data_Source")
GROUND_TRUTH_JSON = "golden.json"

# PDF Files to process
PDFS = [
    "Attention_Is_All_You_Need.pdf",
    "BERT.pdf",
    "CLIP.pdf",
]

# Pinecone Configuration
PINECONE_INDEX_NAME = "research-docs-index"
DOCS_NAMESPACE = ""  # Default namespace for documents
GROUND_TRUTH_NAMESPACE = "ground-truth"  # Namespace for ground truth

# Embedding Configuration
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 1024

# Text Splitting Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Export Type
from langchain_docling.loader import ExportType
DOCLING_EXPORT_TYPE = ExportType.MARKDOWN


