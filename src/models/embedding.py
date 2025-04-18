"""
Embeddings model module for generating text embeddings.
"""
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from zenml.steps import step

class Embedder:
    """
    A class to generate embeddings for text using sentence-transformers.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Generate embeddings using the model
        embeddings = self.model.encode(texts)
        
        # Convert numpy arrays to lists for easier serialization
        return embeddings.tolist()
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embeddings.
        
        Returns:
            Dimension of the embeddings
        """
        return self.model.get_sentence_embedding_dimension()


@step
def initialize_embedding_model(
    model_name: str = "all-MiniLM-L6-v2"
) -> Embedder:
    """
    ZenML step to initialize the embedding model.
    
    Args:
        model_name: Name of the sentence-transformers model to use
        
    Returns:
        Initialized Embedder
    """
    return Embedder(model_name=model_name)


@step
def generate_embeddings(
    texts: List[str],
    embedding_model: Embedder
) -> Dict[str, Any]:
    """
    ZenML step to generate embeddings for texts.
    
    Args:
        texts: List of text strings
        embedding_model: Embedder instance
        
    Returns:
        Dictionary containing embeddings and metadata
    """
    embeddings = embedding_model.generate_embeddings(texts)
    
    return {
        "embeddings": embeddings,
        "dimension": embedding_model.get_embedding_dimension(),
        "model_name": embedding_model.model_name,
        "num_texts": len(texts)
    }
