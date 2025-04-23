"""
Utility functions for vector store operations.
"""
from typing import List, Dict, Any, Optional, Set
import os
import pickle
import numpy as np
from pathlib import Path

class VectorStoreError(Exception):
    """Base exception for vector store errors."""
    pass

class ConfigurationError(VectorStoreError):
    """Raised when there's a configuration issue."""
    pass

class SearchError(VectorStoreError):
    """Raised when there's an error during search."""
    pass

class StorageError(VectorStoreError):
    """Raised when there's an error during storage operations."""
    pass

def validate_embeddings(
    embeddings: Optional[np.ndarray],
    documents: List[Any],
    embedding_dimension: int
) -> np.ndarray:
    """
    Validate and normalize embeddings.
    
    Args:
        embeddings: Optional array of embeddings
        documents: List of documents
        embedding_dimension: Expected embedding dimension
        
    Returns:
        Validated embeddings array
        
    Raises:
        ValueError: If embeddings are invalid
    """
    if embeddings is None:
        return None
        
    if not isinstance(embeddings, np.ndarray):
        raise ValueError("Embeddings must be a NumPy array")
        
    if embeddings.shape[0] != len(documents):
        raise ValueError(
            f"Number of embeddings ({embeddings.shape[0]}) does not match "
            f"number of documents ({len(documents)})"
        )
        
    if embeddings.shape[1] != embedding_dimension:
        raise ValueError(
            f"Embedding dimension ({embeddings.shape[1]}) does not match "
            f"expected dimension ({embedding_dimension})"
        )
        
    return embeddings

def validate_directory(directory: str) -> str:
    """
    Validate and create directory if needed.
    
    Args:
        directory: Directory path
        
    Returns:
        Absolute path to directory
        
    Raises:
        ValueError: If directory cannot be created
    """
    try:
        abs_path = os.path.abspath(directory)
        os.makedirs(abs_path, exist_ok=True)
        return abs_path
    except Exception as e:
        raise ValueError(f"Failed to create directory {directory}: {str(e)}")

def create_config(
    embedding_dimension: int,
    index_type: str,
    cache_size: int,
    num_documents: int
) -> Dict[str, Any]:
    """
    Create configuration dictionary for vector store.
    
    Args:
        embedding_dimension: Dimension of embeddings
        index_type: Type of FAISS index
        cache_size: Size of LRU cache
        num_documents: Number of documents
        
    Returns:
        Configuration dictionary
    """
    return {
        "embedding_dimension": embedding_dimension,
        "index_type": index_type,
        "cache_size": cache_size,
        "num_documents": num_documents
    }

def save_config(
    config: Dict[str, Any],
    directory: str,
    filename: str = "config.pkl"
) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        directory: Directory to save in
        filename: Name of config file
    """
    config_path = os.path.join(directory, filename)
    with open(config_path, "wb") as f:
        pickle.dump(config, f)

def load_config(
    directory: str,
    filename: str = "config.pkl"
) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        directory: Directory containing config
        filename: Name of config file
        
    Returns:
        Configuration dictionary
    """
    config_path = os.path.join(directory, filename)
    with open(config_path, "rb") as f:
        return pickle.load(f)

def save_index(
    index: Any,
    directory: str,
    filename: str = "index.faiss"
) -> None:
    """
    Save FAISS index to file.
    
    Args:
        index: FAISS index to save
        directory: Directory to save in
        filename: Name of index file
    """
    index_path = os.path.join(directory, filename)
    faiss.write_index(index, index_path)

def load_index(
    directory: str,
    filename: str = "index.faiss"
) -> Any:
    """
    Load FAISS index from file.
    
    Args:
        directory: Directory containing index
        filename: Name of index file
        
    Returns:
        Loaded FAISS index
    """
    index_path = os.path.join(directory, filename)
    return faiss.read_index(index_path)

def save_data(
    data: Any,
    directory: str,
    filename: str
) -> None:
    """
    Save data to pickle file.
    
    Args:
        data: Data to save
        directory: Directory to save in
        filename: Name of data file
    """
    data_path = os.path.join(directory, filename)
    with open(data_path, "wb") as f:
        pickle.dump(data, f)

def load_data(
    directory: str,
    filename: str
) -> Any:
    """
    Load data from pickle file.
    
    Args:
        directory: Directory containing data
        filename: Name of data file
        
    Returns:
        Loaded data
    """
    data_path = os.path.join(directory, filename)
    with open(data_path, "rb") as f:
        return pickle.load(f)
