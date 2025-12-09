"""
Document processing module using Docling.
Handles loading PDFs and splitting them into chunks.
"""
from pathlib import Path
from typing import List
from langchain_docling import DoclingLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import Agent.config as config


def load_papers() -> List[Document]:
    """
    Load research papers using Docling.
    
    Returns:
        List[Document]: List of loaded documents with metadata
    """
    docs = []
    
    print(f"\n{'='*60}")
    print("📄 LOADING RESEARCH PAPERS WITH DOCLING")
    print(f"{'='*60}\n")

    for fname in config.PDFS:
        pdf_path = config.DATA_DIR / fname
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"Missing file: {pdf_path.resolve()}")

        print(f"Processing: {fname}")
        
        # Load document with Docling
        loader = DoclingLoader(
            file_path=pdf_path,
            export_type=config.DOCLING_EXPORT_TYPE
        )
        
        file_docs = loader.load()
        print(f"  ✓ Loaded {len(file_docs)} document(s)")

        # Tag metadata - CONVERT ALL VALUES TO STRINGS
        for d in file_docs:
            # Convert any Path objects to strings
            if 'source' in d.metadata and isinstance(d.metadata['source'], Path):
                d.metadata['source'] = str(d.metadata['source'])
            else:
                d.metadata['source'] = fname  # Just the filename as string
            
            d.metadata['paper_id'] = fname.rsplit(".", 1)[0]
            
            # Clean up any other potential Path objects in metadata
            for key, value in d.metadata.items():
                if isinstance(value, Path):
                    d.metadata[key] = str(value)

        docs.extend(file_docs)
    
    print(f"\n✓ Total documents loaded: {len(docs)}")
    return docs


def split_documents(docs: List[Document]) -> List[Document]:
    """
    Split documents into smaller chunks for better retrieval.
    
    Args:
        docs: List of documents to split
        
    Returns:
        List[Document]: List of document chunks
    """
    print(f"\n{'='*60}")
    print("✂️  SPLITTING DOCUMENTS INTO CHUNKS")
    print(f"{'='*60}\n")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    splits = text_splitter.split_documents(docs)
    
    # Double-check metadata after splitting
    for split in splits:
        for key, value in split.metadata.items():
            if isinstance(value, Path):
                split.metadata[key] = str(value)
    
    print(f"✓ Split {len(docs)} documents into {len(splits)} chunks")
    print(f"  • Chunk size: {config.CHUNK_SIZE} characters")
    print(f"  • Chunk overlap: {config.CHUNK_OVERLAP} characters")
    
    return splits


def process_documents() -> List[Document]:
    """
    Main function to process documents: load and split.
    
    Returns:
        List[Document]: Processed document chunks ready for storage
    """
    # Load papers
    docs = load_papers()
    
    # Show sample document
    if docs:
        print(f"\n{'='*60}")
        print("📋 SAMPLE DOCUMENT")
        print(f"{'='*60}")
        print(f"Metadata: {docs[0].metadata}")
        print(f"Content preview: {docs[0].page_content[:200]}...")
    
    # Split documents
    chunks = split_documents(docs)
    
    return chunks


if __name__ == "__main__":
    # Test the processor
    chunks = process_documents()
    print(f"\n✅ Processing complete! Generated {len(chunks)} chunks ready for storage.")