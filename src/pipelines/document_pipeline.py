"""
Document processing pipeline for RAG system.
"""
from typing import List, Dict, Any
from pathlib import Path
from zenml.pipelines import pipeline
from zenml.steps import step
from zenml.logger import get_logger
from langchain.schema import Document

from ..utils.document_processor import process_document
from ..utils.text_splitter import initialize_text_splitter, split_documents
from ..models.embedding import initialize_embedding_model, generate_embeddings
from ..data.vector_store import (
    initialize_vector_store, 
    set_embedding_function,
    add_documents_to_vector_store,
    save_vector_store
)

logger = get_logger(__name__)

@step
def get_document_path(document_path: str) -> str:
    """
    ZenML step to validate and return document path.
    
    Args:
        document_path: Path to document or directory
        
    Returns:
        Validated document path
    """
    path = Path(document_path)
    if not path.exists():
        raise FileNotFoundError(f"Document path does not exist: {document_path}")
    
    logger.info(f"Processing document path: {document_path}")
    return str(path)

@step
def combine_pipeline_outputs(
    documents: List[Document],
    vector_store_path: str
) -> Dict[str, Any]:
    """
    ZenML step to combine outputs from pipeline steps.
    
    Args:
        documents: Processed documents
        vector_store_path: Path to saved vector store
        
    Returns:
        Dictionary with pipeline outputs
    """
    return {
        "num_documents": len(documents),
        "vector_store_path": vector_store_path
    }

@pipeline
def document_processing_pipeline(
    document_path: str,
    output_dir: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    index_type: str = "L2"
):
    """
    Pipeline for processing documents and creating a vector store.
    
    Args:
        document_path: Path to document or directory
        output_dir: Directory to save vector store
        chunk_size: Size of document chunks
        chunk_overlap: Overlap between chunks
        embedding_model_name: Name of embedding model
        index_type: Type of FAISS index
    """
    # Get validated document path
    doc_path = get_document_path(document_path)
    
    # Process documents
    documents = process_document(doc_path)
    
    # Initialize text splitter and split documents
    text_splitter = initialize_text_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    split_docs = split_documents(documents, text_splitter)
    
    # Initialize embedding model
    embedding_model = initialize_embedding_model(
        model_name=embedding_model_name
    )
    
    # Get embedding dimension
    embeddings_data = generate_embeddings(
        [split_docs[0].page_content] if split_docs else [""],
        embedding_model
    )
    embedding_dimension = embeddings_data["dimension"]
    
    # Initialize vector store
    vector_store = initialize_vector_store(
        embedding_dimension=embedding_dimension,
        index_type=index_type
    )
    
    # Set embedding function
    vector_store = set_embedding_function(vector_store, embedding_model)
    
    # Generate embeddings and add to vector store
    embeddings_data = generate_embeddings(
        [doc.page_content for doc in split_docs],
        embedding_model
    )
    
    # Add documents to vector store
    vector_store = add_documents_to_vector_store(
        vector_store,
        split_docs,
        embeddings_data["embeddings"]
    )
    
    # Save vector store
    vector_store_path = save_vector_store(vector_store, output_dir)
    
    # Combine outputs
    return combine_pipeline_outputs(split_docs, vector_store_path)