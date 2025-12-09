"""
All tools for the agentic RAG system.
LLM will decide which tools to call autonomously.
"""
from typing import List
from langchain_core.tools import tool
from langchain_core.documents import Document
#from langchain_pinecone import PineconeVectorStore
from langchain_pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from ragas_evaluator import RAGASEvaluator
import json

import config as config


# Initialize embeddings
embeddings = OpenAIEmbeddings(
    model=config.EMBEDDING_MODEL,
    dimensions=config.EMBEDDING_DIMENSIONS,
    openai_api_key=config.OPENAI_API_KEY
)

# Initialize vector stores
docs_vectorstore = Pinecone.from_existing_index(
    index_name=config.PINECONE_INDEX_NAME,
    embedding=embeddings,
    namespace=config.DOCS_NAMESPACE
)

gt_vectorstore = Pinecone.from_existing_index(
    index_name=config.PINECONE_INDEX_NAME,
    embedding=embeddings,
    namespace=config.GROUND_TRUTH_NAMESPACE
)

# Initialize RAGAS evaluator
ragas_eval = RAGASEvaluator()


# ============================================
# RETRIEVAL TOOLS
# ============================================

@tool
def retrieve_documents(query: str, k: int = 5) -> str:
    """
    Retrieve relevant document chunks from the research papers knowledge base.
    
    Use this tool to get information from research papers (Transformer, BERT, CLIP).
    
    Args:
        query: The search query or question
        k: Number of documents to retrieve (default: 5, can increase to 8 for broader search)
        
    Returns:
        JSON string containing retrieved documents with metadata
    """
    retriever = docs_vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)
    
    # Format as JSON for LLM
    results = []
    for i, doc in enumerate(docs):
        results.append({
            "chunk_id": i,
            "source": doc.metadata.get("source", "unknown"),
            "paper": doc.metadata.get("paper_id", "unknown"),
            "content": doc.page_content[:500]  # Truncate for readability
        })
    
    return json.dumps({
        "retrieved_count": len(docs),
        "documents": results
    }, indent=2)


@tool
def retrieve_ground_truth(query: str, k: int = 3) -> str:
    """
    Retrieve similar ground truth question-answer pairs for validation.
    
    Use this tool to find expected answers and evaluation criteria from the golden dataset.
    
    Args:
        query: The question to find similar ground truth for
        k: Number of ground truth examples to retrieve (default: 3)
        
    Returns:
        JSON string containing ground truth Q&A pairs with expected criteria
    """
    retriever = gt_vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)
    
    # Format as JSON for LLM
    results = []
    for i, doc in enumerate(docs):
        results.append({
            "gt_id": i,
            "question": doc.metadata.get("question", "N/A"),
            "answer": doc.metadata.get("answer", "N/A"),
            "min_chunks_required": doc.metadata.get("min_chunks_required", 2),
            "source_paper": doc.metadata.get("source_paper", "N/A")
        })
    
    return json.dumps({
        "ground_truth_count": len(docs),
        "examples": results
    }, indent=2)


# ============================================
# RAGAS EVALUATION TOOLS
# ============================================

@tool
def evaluate_retrieval_quality(question: str, retrieved_docs_json: str, ground_truth_answer: str) -> str:
    """
    Evaluate the quality of retrieved documents using RAGAS metrics.
    
    Use this tool after retrieving documents to check if they are sufficient.
    Returns context precision and recall scores.
    
    Args:
        question: The original question
        retrieved_docs_json: JSON string of retrieved documents (from retrieve_documents tool)
        ground_truth_answer: Expected answer from ground truth
        
    Returns:
        JSON with precision, recall scores and sufficiency decision
    """
    try:
        # Parse retrieved docs
        retrieved_data = json.loads(retrieved_docs_json)
        
        # Reconstruct Document objects for RAGAS
        docs = []
        for doc_data in retrieved_data.get("documents", []):
            doc = Document(
                page_content=doc_data["content"],
                metadata={"source": doc_data["source"]}
            )
            docs.append(doc)
        
        # Evaluate with RAGAS
        precision, recall = ragas_eval.evaluate_retrieval(question, docs, ground_truth_answer)
        
        # Determine sufficiency
        is_sufficient = precision >= 0.7 and recall >= 0.7
        
        return json.dumps({
            "context_precision": round(precision, 3),
            "context_recall": round(recall, 3),
            "is_sufficient": is_sufficient,
            "recommendation": "Proceed to generation" if is_sufficient else "Retrieve more documents"
        }, indent=2)
        
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


@tool
def evaluate_answer_quality(question: str, answer: str, retrieved_docs_json: str) -> str:
    """
    Evaluate the quality of a generated answer using RAGAS metrics.
    
    Use this tool after generating an answer to validate its quality.
    Returns faithfulness and answer relevancy scores.
    
    Args:
        question: The original question
        answer: The generated answer to evaluate
        retrieved_docs_json: JSON string of documents used for generation
        
    Returns:
        JSON with faithfulness, relevancy scores and quality decision
    """
    try:
        # Parse retrieved docs
        retrieved_data = json.loads(retrieved_docs_json)
        
        # Reconstruct Document objects
        docs = []
        for doc_data in retrieved_data.get("documents", []):
            doc = Document(
                page_content=doc_data["content"],
                metadata={"source": doc_data["source"]}
            )
            docs.append(doc)
        
        # Evaluate with RAGAS
        faithfulness, relevancy = ragas_eval.evaluate_answer(question, answer, docs)
        
        # Determine quality
        is_good_quality = faithfulness >= 0.7 and relevancy >= 0.7
        
        return json.dumps({
            "faithfulness": round(faithfulness, 3),
            "answer_relevancy": round(relevancy, 3),
            "is_good_quality": is_good_quality,
            "recommendation": "Return answer to user" if is_good_quality else "Answer quality insufficient"
        }, indent=2)
        
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


# Export all tools
ALL_TOOLS = [
    retrieve_documents,
    retrieve_ground_truth,
    evaluate_retrieval_quality,
    evaluate_answer_quality
]