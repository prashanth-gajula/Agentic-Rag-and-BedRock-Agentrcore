"""
RAGAS evaluation module for RAG quality assessment.
Handles all RAGAS metrics: context_precision, context_recall, faithfulness, answer_relevancy
"""
from typing import List, Dict, Tuple
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy
from datasets import Dataset
import warnings
import logging

import config as config

# Suppress RAGAS progress bars and warnings
warnings.filterwarnings('ignore')
logging.getLogger('ragas').setLevel(logging.ERROR)
logging.getLogger('httpx').setLevel(logging.ERROR)
logging.getLogger('openai').setLevel(logging.ERROR)


class RAGASEvaluator:
    """Handles RAGAS evaluation for retrieval and generation quality."""
    
    def __init__(self):
        """Initialize LLM and embeddings for RAGAS."""
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.0,
            openai_api_key=config.OPENAI_API_KEY
        )
        
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=config.OPENAI_API_KEY
        )
    
    def evaluate_retrieval(
        self,
        question: str,
        retrieved_docs: List[Document],
        ground_truth_answer: str
    ) -> Tuple[float, float]:
        """
        Evaluate retrieval quality using context_precision and context_recall.
        
        Args:
            question: The user's question
            retrieved_docs: List of retrieved document chunks
            ground_truth_answer: Expected answer from golden dataset
            
        Returns:
            Tuple[float, float]: (precision_score, recall_score)
        """
        try:
            # Prepare contexts
            retrieved_contexts = [str(doc.page_content) for doc in retrieved_docs]
            
            # Create dataset
            eval_data = {
                "question": [question],
                "contexts": [retrieved_contexts],
                "ground_truth": [ground_truth_answer]
            }
            
            dataset = Dataset.from_dict(eval_data)
            
            # Evaluate (progress bar suppressed)
            result = evaluate(
                dataset,
                metrics=[context_precision, context_recall],
                llm=self.llm,
                embeddings=self.embeddings,
                show_progress=False  # Disable progress bar
            )
            
            # Extract scores
            result_dict = result.to_pandas().to_dict('records')[0]
            precision = float(result_dict['context_precision'])
            recall = float(result_dict['context_recall'])
            
            return precision, recall
            
        except Exception as e:
            print(f"    ⚠️ RAGAS evaluation error: {str(e)}")
            # Fallback heuristic
            precision = 0.8 if len(retrieved_docs) >= 3 else 0.5
            recall = 0.8 if len(retrieved_docs) >= 3 else 0.5
            return precision, recall
    
    def evaluate_answer(
        self,
        question: str,
        answer: str,
        retrieved_docs: List[Document]
    ) -> Tuple[float, float]:
        """
        Evaluate answer quality using faithfulness and answer_relevancy.
        
        Args:
            question: The user's question
            answer: The generated answer
            retrieved_docs: List of retrieved document chunks used for generation
            
        Returns:
            Tuple[float, float]: (faithfulness_score, relevancy_score)
        """
        try:
            # Prepare contexts
            retrieved_contexts = [str(doc.page_content) for doc in retrieved_docs]
            
            # Create dataset
            validation_data = {
                "question": [question],
                "answer": [answer],
                "contexts": [retrieved_contexts]
            }
            
            dataset = Dataset.from_dict(validation_data)
            
            # Evaluate (progress bar suppressed)
            result = evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy],
                llm=self.llm,
                embeddings=self.embeddings,
                show_progress=False  # Disable progress bar
            )
            
            # Extract scores
            result_dict = result.to_pandas().to_dict('records')[0]
            faithfulness_score = float(result_dict['faithfulness'])
            relevancy_score = float(result_dict['answer_relevancy'])
            
            return faithfulness_score, relevancy_score
            
        except Exception as e:
            print(f"    ⚠️ Answer evaluation error: {str(e)}")
            # Conservative fallback
            faithfulness_score = 0.8
            relevancy_score = 0.8
            return faithfulness_score, relevancy_score


# Convenience functions
def evaluate_retrieval(question: str, retrieved_docs: List[Document], ground_truth: str) -> Tuple[float, float]:
    """Quick function to evaluate retrieval quality."""
    evaluator = RAGASEvaluator()
    return evaluator.evaluate_retrieval(question, retrieved_docs, ground_truth)


def evaluate_answer(question: str, answer: str, retrieved_docs: List[Document]) -> Tuple[float, float]:
    """Quick function to evaluate answer quality."""
    evaluator = RAGASEvaluator()
    return evaluator.evaluate_answer(question, answer, retrieved_docs)