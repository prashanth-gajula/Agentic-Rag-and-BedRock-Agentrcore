"""
Utility functions for the Agentic RAG system.
"""
from typing import List
from langchain_core.documents import Document


def print_document_sample(docs: List[Document], num_samples: int = 1):
    """
    Print sample documents for inspection.
    
    Args:
        docs: List of documents
        num_samples: Number of samples to print
    """
    print(f"\n{'='*60}")
    print(f"📄 DOCUMENT SAMPLES ({num_samples})")
    print(f"{'='*60}\n")
    
    for i, doc in enumerate(docs[:num_samples], 1):
        print(f"Sample {i}:")
        print(f"  Metadata: {doc.metadata}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:200]}...")
        print()


def print_section_header(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def print_success_message(message: str):
    """Print a success message."""
    print(f"\n{'='*80}")
    print(f"✅ {message}")
    print(f"{'='*80}\n")