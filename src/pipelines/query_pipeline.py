import logging
from typing import List, Tuple
from pathlib import Path

from zenml import pipeline
from zenml.steps import step
from zenml.artifacts import DataArtifact
from zenml.steps import Output

from data.vector_store import load_vector_store, search_vector_store
from models.embedding import get_embedding_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@step
def load_vector_store_step(
    vector_store_dir: str
) -> Output(vector_store=DataArtifact):
    """
    Load vector store from disk.
    
    Args:
        vector_store_dir: Directory containing vector store
        
    Returns:
        Loaded VectorStore instance
    """
    try:
        vector_store = load_vector_store(directory=vector_store_dir)
        logger.info(f"Loaded vector store from {vector_store_dir}")
        return vector_store
        
    except Exception as e:
        logger.error(f"Error loading vector store: {str(e)}")
        raise

@step
def process_query_step(
    query: str
) -> Output(query_embedding=np.ndarray):
    """
    Process and generate embedding for query.
    
    Args:
        query: User query text
        
    Returns:
        Query embedding
    """
    try:
        embedding_model = get_embedding_model()
        query_embedding = embedding_model.get_embedding(query)
        logger.info(f"Generated embedding for query: {query}")
        return query_embedding
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise

@pipeline
def query_pipeline(
    query: str,
    vector_store_dir: str,
    k: int = 5
) -> Output(results=Tuple[List[str], List[float]]):
    """
    Query pipeline for RAG.
    
    Args:
        query: User query text
        vector_store_dir: Directory containing vector store
        k: Number of results to return
        
    Returns:
        Tuple of (documents, similarity scores)
    """
    try:
        # Load vector store
        vector_store = load_vector_store_step(
            vector_store_dir=vector_store_dir
        )
        
        # Process query
        query_embedding = process_query_step(query=query)
        
        # Search vector store
        results = search_vector_store(
            vector_store=vector_store,
            query_embedding=query_embedding,
            k=k
        )
        
        logger.info(f"Found {len(results[0])} relevant documents")
        return results
        
    except Exception as e:
        logger.error(f"Query pipeline failed: {str(e)}")
        raise