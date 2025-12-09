"""
Vector store management module.
Handles Pinecone storage operations for both documents and ground truth.
"""
from typing import List
import json
from pathlib import Path
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from pinecone import Pinecone

import Agent.config as config


class VectorStoreManager:
    """Manages Pinecone vector stores for documents and ground truth."""
    
    def __init__(self):
        """Initialize embeddings and vector stores."""
        print(f"\n{'='*60}")
        print("🔧 INITIALIZING VECTOR STORE MANAGER")
        print(f"{'='*60}\n")
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            dimensions=config.EMBEDDING_DIMENSIONS,
            openai_api_key=config.OPENAI_API_KEY
        )
        print(f"✓ Embeddings initialized: {config.EMBEDDING_MODEL} ({config.EMBEDDING_DIMENSIONS}D)")
        
        # Initialize Pinecone vector stores
        self.docs_vectorstore = PineconeVectorStore(
            index_name=config.PINECONE_INDEX_NAME,
            embedding=self.embeddings,
            namespace=config.DOCS_NAMESPACE
        )
        print(f"✓ Documents vector store ready (namespace: '{config.DOCS_NAMESPACE or 'default'}')")
        
        self.ground_truth_vectorstore = PineconeVectorStore(
            index_name=config.PINECONE_INDEX_NAME,
            embedding=self.embeddings,
            namespace=config.GROUND_TRUTH_NAMESPACE
        )
        print(f"✓ Ground truth vector store ready (namespace: '{config.GROUND_TRUTH_NAMESPACE}')")
    
    def store_documents(self, documents: List[Document]) -> PineconeVectorStore:
        """
        Store research documents in Pinecone (default namespace).
        
        Args:
            documents: List of document chunks to store
            
        Returns:
            PineconeVectorStore: The vector store with stored documents
        """
        print(f"\n{'='*60}")
        print("💾 STORING DOCUMENTS IN PINECONE")
        print(f"{'='*60}\n")
        
        print(f"Storing {len(documents)} document chunks...")
        print(f"  • Index: {config.PINECONE_INDEX_NAME}")
        print(f"  • Namespace: {config.DOCS_NAMESPACE or '(default)'}")
        
        # Add documents to default namespace
        self.docs_vectorstore.add_documents(documents)
        
        print(f"\n✓ Successfully stored {len(documents)} documents!")
        return self.docs_vectorstore
    
    def load_ground_truth(self, json_path: str) -> List[dict]:
        """
        Load ground truth Q&A pairs from JSON file.
        
        Supports two formats:
        1. Array format: [{"paper": "...", "question": "...", "answer": "...", "gold_chunks": [...]}, ...]
        2. Object format: {"qa_pairs": [{"question": "...", "answer": "..."}]}
        
        Args:
            json_path: Path to the golden.json file
            
        Returns:
            List[dict]: List of Q&A pairs
        """
        print(f"\n{'='*60}")
        print("📖 LOADING GROUND TRUTH")
        print(f"{'='*60}\n")
        
        if not Path(json_path).exists():
            raise FileNotFoundError(f"Ground truth file not found: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON formats
        if isinstance(data, list):
            # Array format: [{"paper": "...", "question": "...", ...}, ...]
            qa_pairs = data
            print(f"✓ Detected array format")
        elif isinstance(data, dict) and "qa_pairs" in data:
            # Object format: {"qa_pairs": [...]}
            qa_pairs = data["qa_pairs"]
            print(f"✓ Detected object format with 'qa_pairs' key")
        else:
            raise ValueError("Invalid JSON format. Expected either a list or an object with 'qa_pairs' key")
        
        if not qa_pairs:
            raise ValueError("No Q&A pairs found in ground truth JSON")
        
        print(f"✓ Loaded {len(qa_pairs)} ground truth Q&A pairs from {json_path}")
        
        # Show sample
        if qa_pairs:
            sample = qa_pairs[0]
            print(f"\n📄 Sample Q&A pair:")
            print(f"  • Paper: {sample.get('paper', 'N/A')}")
            print(f"  • Question: {sample.get('question', 'N/A')[:60]}...")
            print(f"  • Answer: {sample.get('answer', 'N/A')[:60]}...")
            print(f"  • Gold chunks: {len(sample.get('gold_chunks', []))} chunks")
        
        return qa_pairs
    
    def store_ground_truth(self, json_path: str) -> PineconeVectorStore:
        """
        Store ground truth Q&A pairs in Pinecone (ground-truth namespace).
        
        Args:
            json_path: Path to the golden.json file
            
        Returns:
            PineconeVectorStore: The vector store with stored ground truth
        """
        # Load ground truth
        qa_pairs = self.load_ground_truth(json_path)
        
        print(f"\n{'='*60}")
        print("💾 STORING GROUND TRUTH IN PINECONE")
        print(f"{'='*60}\n")
        
        # Create documents from Q&A pairs
        documents = []
        for i, qa in enumerate(qa_pairs):
            # Combine question and answer for embedding
            question = qa.get("question", "")
            answer = qa.get("answer", "")
            content = f"Question: {question}\nAnswer: {answer}"
            
            # Add gold chunks if available
            gold_chunks = qa.get("gold_chunks", [])
            if gold_chunks:
                content += "\n\nGold Chunks:\n" + "\n".join(gold_chunks)
            
            # Create metadata
            metadata = {
                "qa_id": f"gt_{i}",
                "question": question,
                "answer": answer,
                "type": "ground_truth"
            }
            
            # Add paper name
            if "paper" in qa:
                metadata["source_paper"] = qa["paper"]
            
            # Add gold chunks as JSON string
            if gold_chunks:
                metadata["gold_chunks"] = json.dumps(gold_chunks)
                metadata["num_gold_chunks"] = len(gold_chunks)
            
            # Optional: min chunks required (default to number of gold chunks)
            metadata["min_chunks_required"] = qa.get("min_chunks_required", len(gold_chunks))
            
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)
        
        print(f"Storing {len(documents)} ground truth Q&A pairs...")
        print(f"  • Index: {config.PINECONE_INDEX_NAME}")
        print(f"  • Namespace: {config.GROUND_TRUTH_NAMESPACE}")
        
        # Add documents to ground-truth namespace
        self.ground_truth_vectorstore.add_documents(documents)
        
        print(f"\n✓ Successfully stored {len(documents)} ground truth pairs!")
        return self.ground_truth_vectorstore
    
    def check_index_stats(self) -> dict:
        """
        Check statistics for both namespaces in the Pinecone index.
        
        Returns:
            dict: Index statistics
        """
        print(f"\n{'='*60}")
        print("📊 PINECONE INDEX STATISTICS")
        print(f"{'='*60}\n")
        
        pc = Pinecone(api_key=config.PINECONE_API_KEY)
        index = pc.Index(config.PINECONE_INDEX_NAME)
        stats = index.describe_index_stats()
        
        print(f"Index: {config.PINECONE_INDEX_NAME}")
        print(f"Total vectors: {stats['total_vector_count']}")
        print(f"Dimension: {stats['dimension']}")
        
        print(f"\n📁 Namespaces:")
        namespaces = stats.get('namespaces', {})
        
        # Default namespace (documents)
        default_count = namespaces.get('', {}).get('vector_count', 0)
        print(f"  • (default) - Research Documents: {default_count:,} vectors")
        
        # Ground truth namespace
        gt_count = namespaces.get(config.GROUND_TRUTH_NAMESPACE, {}).get('vector_count', 0)
        print(f"  • {config.GROUND_TRUTH_NAMESPACE} - Q&A Pairs: {gt_count:,} vectors")
        
        print(f"{'='*60}")
        
        return stats


if __name__ == "__main__":
    # Test the vector store manager
    manager = VectorStoreManager()
    manager.check_index_stats()