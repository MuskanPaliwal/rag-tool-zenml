"""
Utility functions for vector operations and storage.
"""
from typing import Dict, Any
import os
import pickle
import numpy as np
import faiss
from zenml.logger import get_logger

logger = get_logger(__name__)

# Custom exceptions
class VectorStoreError(Exception):
    """Base exception for vector store errors."""

class ConfigurationError(VectorStoreError):
    """Exception for configuration errors."""

class SearchError(VectorStoreError):
    """Exception for search errors."""

class StorageError(VectorStoreError):
    """Exception for storage errors."""

def validate_embeddings(embeddings: np.ndarray, dimension: int) -> bool:
    """
    Validate that embeddings have the correct shape and type.
    
    Args:
        embeddings: Array of embeddings
        dimension: Expected dimension
        
    Returns:
        True if embeddings are valid
        
    Raises:
        ConfigurationError: If embeddings are invalid
    """
    if not isinstance(embeddings, np.ndarray):
        raise ConfigurationError("Embeddings must be a numpy array")
        
    if len(embeddings.shape) != 2:
        raise ConfigurationError(f"Embeddings must be 2D, got shape {embeddings.shape}")
        
    if embeddings.shape[1] != dimension:
        raise ConfigurationError(
            f"Embeddings have dimension {embeddings.shape[1]}, expected {dimension}"
        )
        
    return True
    
def validate_directory(directory: str, create: bool = False) -> bool:
    """
    Validate that a directory exists or create it.
    
    Args:
        directory: Directory path
        create: Whether to create the directory if it doesn't exist
        
    Returns:
        True if directory is valid
        
    Raises:
        StorageError: If directory is invalid and not created
    """
    if os.path.exists(directory):
        if not os.path.isdir(directory):
            raise StorageError(f"Path exists but is not a directory: {directory}")
        return True
        
    if create:
        try:
            os.makedirs(directory, exist_ok=True)
            return True
        except Exception as e:
            raise StorageError(f"Failed to create directory {directory}: {str(e)}") from e
            
    raise StorageError(f"Directory does not exist: {directory}")
    
def create_config(
    embedding_dimension: int,
    index_type: str,
    cache_size: int,
    num_documents: int
) -> Dict[str, Any]:
    """
    Create a configuration dictionary.
    
    Args:
        embedding_dimension: Dimension of embeddings
        index_type: Type of FAISS index
        cache_size: Size of LRU cache
        num_documents: Number of documents in the store
        
    Returns:
        Configuration dictionary
    """
    return {
        "embedding_dimension": embedding_dimension,
        "index_type": index_type,
        "cache_size": cache_size,
        "num_documents": num_documents,
        "version": "1.0"
    }
    
def save_config(config: Dict[str, Any], file_path: str) -> None:
    """
    Save configuration to disk.
    
    Args:
        config: Configuration dictionary
        file_path: Path to save to
    """
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(config, f)
        logger.info("Saved configuration to %s", file_path)
    except Exception as e:
        raise StorageError(f"Failed to save configuration to {file_path}: {str(e)}") from e
        
def load_config(file_path: str) -> Dict[str, Any]:
    """
    Load configuration from disk.
    
    Args:
        file_path: Path to load from
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(file_path, 'rb') as f:
            config = pickle.load(f)
        logger.info("Loaded configuration from %s", file_path)
        return config
    except Exception as e:
        raise StorageError(f"Failed to load configuration from {file_path}: {str(e)}") from e
        
def save_index(index: faiss.Index, file_path: str) -> None:
    """
    Save FAISS index to disk.
    
    Args:
        index: FAISS index
        file_path: Path to save to
    """
    try:
        faiss.write_index(index, file_path)
        logger.info(f"Saved FAISS index to {file_path}")
    except Exception as e:
        raise StorageError(f"Failed to save FAISS index to {file_path}: {str(e)}") from e
        
def load_index(file_path: str) -> faiss.Index:
    """
    Load FAISS index from disk.
    
    Args:
        file_path: Path to load from
        
    Returns:
        FAISS index
    """
    try:
        index = faiss.read_index(file_path)
        logger.info(f"Loaded FAISS index from {file_path}")
        return index
    except Exception as e:
        raise StorageError(f"Failed to load FAISS index from {file_path}: {str(e)}") from e
        
def save_data(data: Any, file_path: str) -> None:
    """
    Save data to disk using pickle.
    
    Args:
        data: Data to save
        file_path: Path to save to
    """
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Saved data to {file_path}")
    except Exception as e:
        raise StorageError(f"Failed to save data to {file_path}: {str(e)}") from e
        
def load_data(file_path: str) -> Any:
    """
    Load data from disk using pickle.
    
    Args:
        file_path: Path to load from
        
    Returns:
        Loaded data
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"Loaded data from {file_path}")
        return data
    except Exception as e:
        raise StorageError(f"Failed to load data from {file_path}: {str(e)}") from e