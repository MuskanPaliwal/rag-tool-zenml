"""
Vector store module for storing and retrieving document embeddings.
"""
from typing import List, Dict, Any, Tuple, Optional, Set
import os
import pickle
import numpy as np
import faiss
from langchain.schema import Document
from zenml.steps import step
import hashlib
from functools import lru_cache
from collections import defaultdict

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
            index_type: Type of FAISS index to use
            cache_size: Size of LRU cache for embeddings
        """
        self.embedding_dimension = embedding_dimension
        self.index_type = index_type
        self.cache_size = cache_size
        
        if index_type == "L2":
            self.index = faiss.IndexFlatL2(embedding_dimension)
        elif index_type == "IP":
            self.index = faiss.IndexFlatIP(embedding_dimension)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        self.documents = []
        self.metadata = []
        self.tag_index = defaultdict(set)  # For tag-based filtering
        self.embedding_cache = lru_cache(maxsize=cache_size)(self._get_embedding)
        self.batch_size = 100  # Default batch size
        
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text with caching."""
        # This will be overridden by the actual embedding model
        return np.random.rand(self.embedding_dimension)
        
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
        if len(documents) != embeddings.shape[0] if embeddings is not None else len(documents):
            raise ValueError(
                f"Number of documents ({len(documents)}) does not match "
                f"number of embeddings ({embeddings.shape[0] if embeddings is not None else len(documents)})"
            )
        
        # Generate embeddings if not provided
        if embeddings is None:
            embeddings = np.array([self.embedding_cache(doc.page_content) for doc in documents])
        
        # Add to index
        self.index.add(embeddings)
        
        # Add documents and metadata
        start_idx = len(self.documents)
        self.documents.extend(documents)
        self.metadata.extend([doc.metadata for doc in documents])
        
        # Add tags to index if provided
        if tags:
            for idx, tag in enumerate(tags):
                self.tag_index[tag].add(start_idx + idx)
        
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
            return [], []
        
        # Get query embedding and optionally process keywords
        query_embedding = self.embedding_cache(query)
        keywords = []
        if hybrid_search:
            keywords = query.lower().split()
        
        # Search the index
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1),
            k * 2  # Search more to filter later
        )
        
        # Apply filters
        filtered_indices = []
        filtered_distances = []
        
        for idx, dist in zip(indices[0], distances[0]):
            if idx >= len(self.documents):
                continue
                
            doc = self.documents[idx]
            meta = self.metadata[idx]
            
            # Apply tag filter
            if tags and not any(tag in self.tag_index for tag in tags):
                continue
                
            # Apply metadata filter
            if filter_metadata:
                skip = False
                for key, value in filter_metadata.items():
                    if key not in meta or meta[key] != value:
                        skip = True
                        break
                if skip:
                    continue
                    
            # Apply hybrid search scoring
            score = 1.0 - dist
            if hybrid_search:
                keyword_score = sum(
                    1 for keyword in keywords
                    if keyword.lower() in doc.page_content.lower()
                ) / max(len(keywords), 1)
                score = (1 - keyword_weight) * score + keyword_weight * keyword_score
                
            filtered_indices.append(idx)
            filtered_distances.append(score)
            
            # Stop if we have enough results
            if len(filtered_indices) >= k:
                break
        
        # Sort by score and get top k
        sorted_indices = [x for _, x in sorted(
            zip(filtered_distances, filtered_indices),
            reverse=True
        )[:k]]
        
        # Get the documents and scores
        documents = [self.documents[i] for i in sorted_indices]
        scores = [filtered_distances[filtered_indices.index(i)] for i in sorted_indices]
        
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
        results = []
        batch_size = min(len(queries), self.batch_size)
        
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
            batch_results = []
            
            for query in batch:
                batch_results.append(self.search(query, k, **kwargs))
            
            results.extend(batch_results)
        
        return results
        
    def save(self, directory: str) -> None:
        """
        Save the vector store to disk.
        
        Args:
            directory: Directory to save to
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save the index
        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
        
        # Save the documents and metadata
        with open(os.path.join(directory, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)
        
        with open(os.path.join(directory, "metadata.pkl"), "wb") as f:
            pickle.dump(self.metadata, f)
        
        # Save tag index
        with open(os.path.join(directory, "tag_index.pkl"), "wb") as f:
            pickle.dump(dict(self.tag_index), f)
        
        # Save config
        config = {
            "embedding_dimension": self.embedding_dimension,
            "index_type": self.index_type,
            "cache_size": self.cache_size,
            "num_documents": len(self.documents)
        }
        
        with open(os.path.join(directory, "config.pkl"), "wb") as f:
            pickle.dump(config, f)
        
    @classmethod
    def load(cls, directory: str) -> "VectorStore":
        """
        Load a vector store from disk.
        
        Args:
            directory: Directory to load from
            
        Returns:
            Loaded VectorStore
        """
        # Load config
        with open(os.path.join(directory, "config.pkl"), "rb") as f:
            config = pickle.load(f)
        
        # Create instance
        instance = cls(
            embedding_dimension=config["embedding_dimension"],
            index_type=config["index_type"],
            cache_size=config["cache_size"]
        )
        
        # Load index
        instance.index = faiss.read_index(os.path.join(directory, "index.faiss"))
        
        # Load documents and metadata
        with open(os.path.join(directory, "documents.pkl"), "rb") as f:
            instance.documents = pickle.load(f)
        
        with open(os.path.join(directory, "metadata.pkl"), "rb") as f:
            instance.metadata = pickle.load(f)
        
        # Load tag index
        with open(os.path.join(directory, "tag_index.pkl"), "rb") as f:
            instance.tag_index = defaultdict(set, pickle.load(f))
        
        return instance

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
def add_documents_to_vector_store(
    vector_store: VectorStore,
    documents: List[Document],
    embeddings_data: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None
) -> VectorStore:
    """
    ZenML step to add documents to the vector store.
    
    Args:
        vector_store: VectorStore instance
        documents: List of Document objects
        embeddings_data: Optional dictionary containing embeddings and metadata
        tags: Optional list of tags for filtering
        
    Returns:
        Updated VectorStore
    """
    embeddings = embeddings_data["embeddings"] if embeddings_data else None
    vector_store.add_documents(documents, embeddings, tags)
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
    return os.path.join(directory, "vector_store")

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
    **kwargs
) -> Tuple[List[Document], List[float]]:
    """
    ZenML step to search the vector store
    
    Args:
        vector_store: VectorStore instance
        query: Search query
        k: Number of results to return
        kwargs: Additional search parameters
        
    Returns:
        Tuple of (list of documents, list of similarity scores)
    """
    return vector_store.search(query, k, **kwargs)

@step
def batch_search_vector_store(
    vector_store: VectorStore,
    queries: List[str],
    k: int = 5,
    **kwargs
) -> List[Tuple[List[Document], List[float]]]:
    """
    ZenML step to perform batch search on the vector store
    
    Args:
        vector_store: VectorStore instance
        queries: List of search queries
        k: Number of results to return per query
        kwargs: Additional search parameters
        
    Returns:
        List of (documents, scores) tuples for each query
    """
    return vector_store.batch_search(queries, k, **kwargs)