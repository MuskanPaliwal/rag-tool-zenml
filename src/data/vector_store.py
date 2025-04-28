"""
Vector store module for storing and retrieving document embeddings.
"""
from typing import List, Dict, Any, Tuple, Optional
from functools import lru_cache
from collections import defaultdict
import os
import numpy as np
import faiss
from langchain.schema import Document
from zenml.steps import step
from zenml.logger import get_logger
from ..utils.vector_utils import (
    ConfigurationError, SearchError, StorageError,
    validate_embeddings, validate_directory,
    create_config, save_config, load_config,
    save_index, load_index,
    save_data, load_data
)

logger = get_logger(__name__)

class VectorStore:
    """
    A class to store and retrieve document embeddings using FAISS.
    """
    
    def __init__(
        self, 
        embedding_dimension: int,
        index_type: str = "L2",
        cache_size: int = 1000
    ):
        """
        Initialize the vector store.
        
        Args:
            embedding_dimension: Dimension of the embeddings
            index_type: Type of FAISS index to use (L2 or IP)
            cache_size: Size of LRU cache for embeddings
        """
        if embedding_dimension <= 0:
            raise ConfigurationError("Embedding dimension must be positive")
            
        if cache_size <= 0:
            raise ConfigurationError("Cache size must be positive")
            
        if index_type not in ["L2", "IP"]:
            raise ConfigurationError(f"Unsupported index type: {index_type}")
            
        self.embedding_dimension = embedding_dimension
        self.index_type = index_type
        self.cache_size = cache_size
        self._embedding_func = None  # Initialize embedding function to None
        self.embedding_cache = None  # Initialize embedding cache to None
        
        # Initialize FAISS index
        if index_type == "L2":
            self.index = faiss.IndexFlatL2(embedding_dimension)
        elif index_type == "IP":
            self.index = faiss.IndexFlatIP(embedding_dimension)
        
        self.documents = []  # List of Document objects
        self.metadata = []   # List of metadata dictionaries
        self.tag_index = defaultdict(set)  # Mapping from tags to document indices
        self.batch_size = 100  # Default batch size for batch operations
        
        logger.info(
            f"Initialized VectorStore with dimension={embedding_dimension}, "
            f"index_type={index_type}, cache_size={cache_size}"
        )
        
    def set_embedding_function(self, embedding_func):
        """
        Set the function to use for generating embeddings.
        
        Args:
            embedding_func: Function that takes a string and returns an embedding vector
        """
        self._embedding_func = embedding_func
        if self._embedding_func is not None:
            self.embedding_cache = lru_cache(maxsize=self.cache_size)(self._embedding_func)
        else:
            raise ConfigurationError("Embedding function cannot be None.")
        self.embedding_cache = lru_cache(maxsize=self.cache_size)(self._embedding_func)
        logger.info("Set embedding function for VectorStore")
        
    def add_documents(
        self, 
        documents: List[Document], 
        embeddings: Optional[np.ndarray] = None,
        tags: Optional[List[str]] = None
    ) -> None:
        """
        Add documents and their embeddings to the vector store.
        
        Args:
            documents: List of Document objects
            embeddings: Optional array of embeddings
            tags: Optional list of tags for filtering
        """
        if not documents:
            logger.warning("No documents provided to add_documents")
            return
            
        # Validate embeddings if provided
        if embeddings is not None:
            if len(documents) != embeddings.shape[0]:
                raise ValueError(
                    f"Number of documents ({len(documents)}) does not match "
                    f"number of embeddings ({embeddings.shape[0]})"
                )
            validate_embeddings(embeddings, self.embedding_dimension)
        else:
            # If no embedding function is set, raise an error
            if not hasattr(self, '_embedding_func'):
                raise ConfigurationError(
                    "No embedding function set. Use set_embedding_function() before adding documents."
                )
            
            # Generate embeddings using the cached function
            embeddings_list = []
            for doc in documents:
                embeddings_list.append(self.embedding_cache(doc.page_content))
            embeddings = np.array(embeddings_list)
        
        # Add to index
        self.index.add(embeddings)
        
        # Add documents and metadata
        start_idx = len(self.documents)
        self.documents.extend(documents)
        self.metadata.extend([doc.metadata for doc in documents])
        
        # Add tags to index if provided
        if tags:
            for i, doc_idx in enumerate(range(start_idx, start_idx + len(documents))):
                for tag in tags:
                    self.tag_index[tag].add(doc_idx)
        
        logger.info(f"Added {len(documents)} documents to VectorStore")
        
    def search(
        self, 
        query: str,
        k: int = 5,
        tags: Optional[List[str]] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        hybrid_search: bool = False,
        keyword_weight: float = 0.3
    ) -> Tuple[List[Document], List[float]]:
        """
        Search for similar documents with advanced filtering.
        
        Args:
            query: Search query
            k: Number of results to return
            tags: Optional tags to filter by
            filter_metadata: Optional metadata filters
            hybrid_search: Whether to use hybrid search
            keyword_weight: Weight for keyword search in hybrid mode
            
        Returns:
            Tuple of (list of documents, list of similarity scores)
        """
        if self.index.ntotal == 0:
            logger.warning("Vector store is empty, returning empty results")
            return [], []
            
        if not hasattr(self, 'embedding_cache'):
            raise ConfigurationError(
                "No embedding function set. Use set_embedding_function() before searching."
            )
        
        # Get query embedding
        query_embedding = self.embedding_cache(query)
        
        # Extract keywords for hybrid search if enabled
        keywords = []
        if hybrid_search:
            keywords = [w.lower() for w in query.split() if len(w) > 2]
        
        # Search the index - retrieve more results than needed for filtering
        search_k = min(k * 4, self.index.ntotal)  # Adjust based on total docs
        query_vector = np.array([query_embedding]).astype('float32')
        
        try:
            distances, indices = self.index.search(query_vector, search_k)
        except Exception as e:
            logger.error(f"FAISS search error: {str(e)}")
            raise SearchError(f"Failed to search index: {str(e)}") from e
        
        # Filter and rank results
        results = []
        
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self.documents):
                continue  # Skip invalid indices
                
            doc = self.documents[idx]
            meta = self.metadata[idx]
            
            # Apply tag filter if provided
            if tags and not any(idx in self.tag_index.get(tag, set()) for tag in tags):
                continue
                
            # Apply metadata filter if provided
            if filter_metadata:
                skip = False
                for key, value in filter_metadata.items():
                    if key not in meta or meta[key] != value:
                        skip = True
                        break
                if skip:
                    continue
            
            # Calculate similarity score (convert distance to similarity)
            if self.index_type == "L2":
                # For L2 distance, smaller is better, so invert
                base_score = 1.0 / (1.0 + dist)
            else:
                # For IP (inner product), higher is better
                base_score = max(0.0, dist)
                
            # Apply hybrid search if enabled
            final_score = base_score
            if hybrid_search and keywords:
                keyword_matches = sum(
                    1 for keyword in keywords 
                    if keyword in doc.page_content.lower()
                )
                keyword_score = keyword_matches / len(keywords)
                final_score = (1 - keyword_weight) * base_score + keyword_weight * keyword_score
            
            results.append((idx, doc, final_score))
        
        # Sort by score (descending) and get top k
        results.sort(key=lambda x: x[2], reverse=True)
        results = results[:k]
        
        # Extract documents and scores
        documents = [item[1] for item in results]
        scores = [item[2] for item in results]
        
        logger.info(f"Search returned {len(documents)} results")
        return documents, scores
        
    def batch_search(
        self, 
        queries: List[str],
        k: int = 5,
        **kwargs
    ) -> List[Tuple[List[Document], List[float]]]:
        """
        Perform batch search for multiple queries.
        
        Args:
            queries: List of search queries
            k: Number of results to return per query
            kwargs: Additional search parameters
            
        Returns:
            List of (documents, scores) tuples for each query
        """
        if not queries:
            return []
            
        results = []
        for query in queries:
            results.append(self.search(query, k, **kwargs))
            
        return results
        
    def save(self, directory: str) -> None:
        """
        Save the vector store to disk.
        
        Args:
            directory: Directory to save to
        """
        try:
            validate_directory(directory, create=True)
            
            # Save the index
            index_path = os.path.join(directory, "index.faiss")
            save_index(self.index, index_path)
            
            # Save the documents and metadata
            docs_path = os.path.join(directory, "documents.pkl")
            meta_path = os.path.join(directory, "metadata.pkl")
            save_data(self.documents, docs_path)
            save_data(self.metadata, meta_path)
            
            # Save tag index
            tags_path = os.path.join(directory, "tag_index.pkl")
            save_data(dict(self.tag_index), tags_path)
            
            # Save config
            config = create_config(
                embedding_dimension=self.embedding_dimension,
                index_type=self.index_type,
                cache_size=self.cache_size,
                num_documents=len(self.documents)
            )
            config_path = os.path.join(directory, "config.pkl")
            save_config(config, config_path)
            
            logger.info(f"Saved vector store to {directory}")
            
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            raise StorageError(f"Failed to save vector store: {str(e)}") from e
        
    @classmethod
    def load(cls, directory: str) -> "VectorStore":
        """
        Load a vector store from disk.
        
        Args:
            directory: Directory to load from
            
        Returns:
            Loaded VectorStore
        """
        try:
            validate_directory(directory, create=False)
            
            # Load config
            config_path = os.path.join(directory, "config.pkl")
            config = load_config(config_path)
            
            # Create instance
            instance = cls(
                embedding_dimension=config["embedding_dimension"],
                index_type=config["index_type"],
                cache_size=config["cache_size"]
            )
            
            # Load index
            index_path = os.path.join(directory, "index.faiss")
            instance.index = load_index(index_path)
            
            # Load documents and metadata
            docs_path = os.path.join(directory, "documents.pkl")
            meta_path = os.path.join(directory, "metadata.pkl")
            instance.documents = load_data(docs_path)
            instance.metadata = load_data(meta_path)
            
            # Load tag index
            tags_path = os.path.join(directory, "tag_index.pkl")
            tag_data = load_data(tags_path)
            instance.tag_index = defaultdict(set, tag_data)
            
            logger.info(
                f"Loaded vector store from {directory} with "
                f"{len(instance.documents)} documents"
            )
            
            return instance
            
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            raise StorageError(f"Failed to load vector store: {str(e)}") from e


# ZenML steps for vector store operations
@step
def initialize_vector_store(
    embedding_dimension: int,
    index_type: str = "L2",
    cache_size: int = 1000
) -> VectorStore:
    """
    ZenML step to initialize the vector store.
    
    Args:
        embedding_dimension: Dimension of the embeddings
        index_type: Type of FAISS index to use
        cache_size: Size of LRU cache for embeddings
        
    Returns:
        Initialized VectorStore
    """
    return VectorStore(
        embedding_dimension=embedding_dimension,
        index_type=index_type,
        cache_size=cache_size
    )

@step
def set_embedding_function(
    vector_store: VectorStore,
    embedding_model: Any
) -> VectorStore:
    """
    ZenML step to set the embedding function for the vector store.
    
    Args:
        vector_store: VectorStore instance
        embedding_model: Model with an embed_text method
        
    Returns:
        Updated VectorStore
    """
    # Create a wrapper function for the embedding model
    def embedding_function(text: str) -> np.ndarray:
        embedding = embedding_model.embed_single_text(text)
        return np.array(embedding)
        
    vector_store.set_embedding_function(embedding_function)
    return vector_store

@step
def add_documents_to_vector_store(
    vector_store: VectorStore,
    documents: List[Document],
    embeddings: Optional[List[List[float]]] = None,
    tags: Optional[List[str]] = None
) -> VectorStore:
    """
    ZenML step to add documents to the vector store.
    
    Args:
        vector_store: VectorStore instance
        documents: List of Document objects
        embeddings: Optional list of embedding vectors
        tags: Optional list of tags for filtering
        
    Returns:
        Updated VectorStore
    """
    # Convert embeddings to numpy array if provided
    embeddings_array = None
    if embeddings:
        embeddings_array = np.array(embeddings)
        
    vector_store.add_documents(documents, embeddings_array, tags)
    return vector_store

@step
def save_vector_store(
    vector_store: VectorStore,
    directory: str
) -> str:
    """
    ZenML step to save the vector store
    
    Args:
        vector_store: VectorStore instance
        directory: Directory to save to
        
    Returns:
        Path to the saved vector store
    """
    vector_store.save(directory)
    return directory

@step
def load_vector_store(
    directory: str
) -> VectorStore:
    """
    ZenML step to load a vector store from disk
    
    Args:
        directory: Directory to load from
        
    Returns:
        Loaded VectorStore
    """
    return VectorStore.load(directory)

@step
def search_vector_store(
    vector_store: VectorStore,
    query: str,
    k: int = 5,
    tags: Optional[List[str]] = None,
    filter_metadata: Optional[Dict[str, Any]] = None,
    hybrid_search: bool = False
) -> Tuple[List[Document], List[float]]:
    """
    ZenML step to search the vector store
    
    Args:
        vector_store: VectorStore instance
        query: Search query
        k: Number of results to return
        tags: Optional tags to filter by
        filter_metadata: Optional metadata filters
        hybrid_search: Whether to use hybrid search
        
    Returns:
        Tuple of (list of documents, list of similarity scores)
    """
    return vector_store.search(
        query=query, 
        k=k, 
        tags=tags, 
        filter_metadata=filter_metadata,
        hybrid_search=hybrid_search
    )