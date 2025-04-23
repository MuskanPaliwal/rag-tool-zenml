"""
Vector store module for storing and retrieving document embeddings.
"""
from typing import List, Dict, Any, Tuple
import os
import pickle
import numpy as np
import faiss
from langchain.schema import Document
from zenml.steps import step

class VectorStore:
    """
    A class to store and retrieve document embeddings using FAISS.
    """
    
    def __init__(
        self, 
        embedding_dimension: int,
        index_type: str = "L2"
    ):
        """
        Initialize the vector store.
        
        Args:
            embedding_dimension: Dimension of the embeddings
            index_type: Type of FAISS index to use
        """
        self.embedding_dimension = embedding_dimension
        self.index_type = index_type
        
        if index_type == "L2":
            self.index = faiss.IndexFlatL2(embedding_dimension)
        elif index_type == "IP":
            self.index = faiss.IndexFlatIP(embedding_dimension)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        self.documents = []
        self.metadata = []
    
    def add_documents(
        self, 
        documents: List[Document], 
        embeddings: np.ndarray
    ) -> None:
        """
        Add documents and their embeddings to the vector store.
        
        Args:
            documents: List of Document objects
            embeddings: Array of embeddings
        """
        if len(documents) != embeddings.shape[0]:
            raise ValueError(
                f"Number of documents ({len(documents)}) does not match "
                f"number of embeddings ({embeddings.shape[0]})"
            )
        
        self.index.add(embeddings)
        self.documents.extend(documents)
        self.metadata.extend([doc.metadata for doc in documents])
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        k: int = 5
    ) -> Tuple[List[Document], List[float]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Embedding of the query
            k: Number of results to return
            
        Returns:
            Tuple of (list of documents, list of similarity scores)
        """
        if self.index.ntotal == 0:
            return [], []
        
        # Reshape query embedding to (1, dimension)
        query_embedding = query_embedding.reshape(1, -1)
        
        # Search the index
        distances, indices = self.index.search(query_embedding, k)
        
        # Get the documents and scores
        documents = [self.documents[i] for i in indices[0] if i < len(self.documents)]
        scores = [float(1.0 - d) for d in distances[0]]  # Convert distance to similarity
        
        return documents, scores
    
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
        
        # Save config
        config = {
            "embedding_dimension": self.embedding_dimension,
            "index_type": self.index_type,
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
            index_type=config["index_type"]
        )
        
        # Load index
        instance.index = faiss.read_index(os.path.join(directory, "index.faiss"))
        
        # Load documents and metadata
        with open(os.path.join(directory, "documents.pkl"), "rb") as f:
            instance.documents = pickle.load(f)
        
        with open(os.path.join(directory, "metadata.pkl"), "rb") as f:
            instance.metadata = pickle.load(f)
        
        return instance

# ZenML steps for vector store operations
@step
def initialize_vector_store(
    embedding_dimension: int,
    index_type: str = "L2"
) -> VectorStore:
    """
    ZenML step to initialize the vector store.
    
    Args:
        embedding_dimension: Dimension of the embeddings
        index_type: Type of FAISS index to use
        
    Returns:
        Initialized VectorStore
    """
    return VectorStore(
        embedding_dimension=embedding_dimension,
        index_type=index_type
    )

@step
def add_documents_to_vector_store(
    vector_store: VectorStore,
    documents: List[Document],
    embeddings_data: Dict[str, Any]
) -> VectorStore:
    """
    ZenML step to add documents to the vector store.
    
    Args:
        vector_store: VectorStore instance
        documents: List of Document objects
        embeddings_data: Dictionary containing embeddings and metadata
        
    Returns:
        Updated VectorStore
    """
    embeddings = embeddings_data["embeddings"]
    vector_store.add_documents(documents, embeddings)
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
    query_embedding: np.ndarray,
    k: int = 5
) -> Tuple[List[Document], List[float]]:
    """
    ZenML step to search the vector store
    
    Args:
        vector_store: VectorStore instance
        query_embedding: Embedding of the query
        k: Number of results to return
        
    Returns:
        Tuple of (list of documents, list of similarity scores)
    """
    return vector_store.search(query_embedding, k)