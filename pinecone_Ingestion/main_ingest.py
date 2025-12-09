"""
Main ingestion script for the Agentic RAG system.
Orchestrates document processing and storage.
"""
import Agent.config as config
from document_processor import process_documents
from vectorstore_manager import VectorStoreManager
from utils import print_section_header, print_success_message, print_document_sample


def main():
    """Main function to orchestrate document and ground truth ingestion."""
    
    print_section_header("🚀 AGENTIC RAG DATA INGESTION PIPELINE")
    
    # ============================================
    # PART 1: PROCESS RESEARCH DOCUMENTS
    # ============================================
    
    print_section_header("PART 1: Processing Research Documents")
    
    # Process documents with Docling
    chunks = process_documents()
    
    # Show sample
    print_document_sample(chunks, num_samples=2)
    
    # ============================================
    # PART 2: INITIALIZE VECTOR STORE MANAGER
    # ============================================
    
    print_section_header("PART 2: Initializing Vector Store Manager")
    
    manager = VectorStoreManager()
    
    # ============================================
    # PART 3: STORE DOCUMENTS IN PINECONE
    # ============================================
    
    print_section_header("PART 3: Storing Documents in Pinecone")
    
    docs_vs = manager.store_documents(chunks)
    
    # ============================================
    # PART 4: STORE GROUND TRUTH IN PINECONE
    # ============================================
    
    print_section_header("PART 4: Storing Ground Truth in Pinecone")
    
    gt_vs = manager.store_ground_truth(config.GROUND_TRUTH_JSON)
    
    # ============================================
    # PART 5: VERIFY STORAGE
    # ============================================
    
    print_section_header("PART 5: Verification")
    
    stats = manager.check_index_stats()
    
    # ============================================
    # FINAL SUMMARY
    # ============================================
    
    print_success_message("DATA INGESTION COMPLETED SUCCESSFULLY!")
    
    print("📌 Summary:")
    print(f"  • Processed {len(chunks)} document chunks from {len(config.PDFS)} papers")
    print(f"  • Stored in Pinecone index: {config.PINECONE_INDEX_NAME}")
    print(f"  • Research documents namespace: '{config.DOCS_NAMESPACE or 'default'}'")
    print(f"  • Ground truth namespace: '{config.GROUND_TRUTH_NAMESPACE}'")
    print(f"  • Total vectors in index: {stats['total_vector_count']:,}")
    print("\n✨ System ready for Agentic RAG workflow!\n")


if __name__ == "__main__":
    main()