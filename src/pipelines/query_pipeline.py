"""
Query pipeline for RAG system.
"""
from typing import List, Dict, Any
from pathlib import Path
from zenml.pipelines import pipeline
from zenml.steps import step
from zenml.logger import get_logger
from langchain.schema import Document

from ..models.embedding import initialize_embedding_model
from ..data.vector_store import (
    load_vector_store,
    set_embedding_function,
    search_vector_store
)

logger = get_logger(__name__)

@step
def validate_vector_store_path(vector_store_path: str) -> str:
    """
    ZenML step to validate vector store path.
    
    Args:
        vector_store_path: Path to vector store
        
    Returns:
        Validated vector store path
    """
    path = Path(vector_store_path)
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Vector store does not exist: {vector_store_path}")
    
    # Check for required files
    required_files = ["index.faiss", "config.pkl", "documents.pkl"]
    missing_files = [f for f in required_files if not (path / f).exists()]
    
    if missing_files:
        raise FileNotFoundError(
            f"Vector store is missing required files: {', '.join(missing_files)}"
        )
    
    logger.info(f"Using vector store at {vector_store_path}")
    return str(path)

@step
def prepare_query(query: str) -> str:
    """
    ZenML step to prepare query string.
    
    Args:
        query: Raw query string
        
    Returns:
        Processed query string
    """
    # Trim whitespace
    query = query.strip()
    
    if not query:
        raise ValueError("Query cannot be empty")
    
    logger.info(f"Processing query: {query}")
    return query

@step
def format_search_results(
    documents: List[Document],
    scores: List[float]
) -> List[Dict[str, Any]]:
    """
    ZenML step to format search results.
    
    Args:
        documents: List of retrieved documents
        scores: List of similarity scores
        
    Returns:
        Formatted search results
    """
    results = []
    
    for doc, score in zip(documents, scores):
        results.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": score
        })
    
    logger.info(f"Returning {len(results)} formatted search results")
    return results

@pipeline
def query_pipeline(
    query: str,
    vector_store_path: str,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    top_k: int = 5,
    hybrid_search: bool = True
) -> List[Dict[str, Any]]:
    """
    Pipeline for querying the vector store.
    
    Args:
        query: Query string
        vector_store_path: Path to vector store
        embedding_model_name: Name of embedding model
        top_k: Number of results to return
        hybrid_search: Whether to use hybrid search
        
    Returns:
        List of search results
    """
    # Validate inputs
    validated_path = validate_vector_store_path(vector_store_path)
    processed_query = prepare_query(query)
    
    # Load vector store
    vector_store = load_vector_store(validated_path)
    
    # Initialize embedding model
    embedding_model = initialize_embedding_model(
        model_name=embedding_model_name
    )
    
    # Set embedding function for vector store
    vector_store = set_embedding_function(vector_store, embedding_model)
    
    # Search vector store
    documents, scores = search_vector_store(
        vector_store,
        processed_query,
        k=top_k,
        hybrid_search=hybrid_search
    )
    
    # Format results
    return format_search_results(documents, scores)