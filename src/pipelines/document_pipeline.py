import logging
from typing import List
from pathlib import Path
import numpy as np

from zenml import pipeline
from zenml.steps import step
from zenml.artifacts import DataArtifact
from zenml.steps import Output

from data.vector_store import initialize_vector_store, add_documents_to_vector_store, save_vector_store
from models.embedding import get_embedding_model
from utils.text_splitter import TextSplitter
from utils.document_processor import DocumentProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@step
def process_documents_step(
    documents: List[str],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> Output(processed_docs=List[str], chunks=List[str]):
    """
    Process and split documents into chunks.
    
    Args:
        documents: List of document texts
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        
    Returns:
        Tuple of processed documents and chunks
    """
    try:
        # Process documents
        document_processor = DocumentProcessor()
        processed_docs = document_processor.process_documents(documents)
        
        # Split into chunks
        text_splitter = TextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_text(processed_docs)
        
        logger.info(f"Processed {len(documents)} documents into {len(chunks)} chunks")
        return processed_docs, chunks
        
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
        raise

@step
def generate_embeddings_step(
    chunks: List[str]
) -> Output(embeddings=np.ndarray):
    """
    Generate embeddings for document chunks.
    
    Args:
        chunks: List of document chunks
        
    Returns:
        Array of embeddings
    """
    try:
        embedding_model = get_embedding_model()
        embeddings = embedding_model.get_embeddings(chunks)
        logger.info(f"Generated embeddings for {len(chunks)} chunks")
        return embeddings
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise

@pipeline
def document_processing_pipeline(
    documents: List[str],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embedding_dimension: int = 768,
    vector_store_dir: str = "vector_store"
):
    """
    Document processing pipeline for RAG.
    
    Args:
        documents: List of document texts to process
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        embedding_dimension: Dimension of embeddings
        vector_store_dir: Directory to save vector store
    """
    try:
        # Create output directory
        output_dir = Path(vector_store_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process documents and get chunks
        processed_docs, chunks = process_documents_step(
            documents=documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Generate embeddings
        embeddings = generate_embeddings_step(chunks=chunks)
        
        # Initialize vector store
        vector_store = initialize_vector_store(
            embedding_dimension=embedding_dimension
        )
        
        # Add documents to vector store
        vector_store = add_documents_to_vector_store(
            vector_store=vector_store,
            documents=chunks,
            embeddings_data={"embeddings": embeddings}
        )
        
        # Save vector store
        save_path = save_vector_store(
            vector_store=vector_store,
            directory=str(output_dir)
        )
        
        logger.info(f"Vector store saved to: {save_path}")
        return save_path
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise