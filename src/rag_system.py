"""
RAG (Retrieval Augmented Generation) system using ZenML pipelines.
This module provides a simple interface to the document processing and query pipelines.
"""
import os
import tempfile
from typing import Dict, Any, Optional
from pathlib import Path
import shutil

from zenml.logger import get_logger

from src.pipelines.document_pipeline import document_processing_pipeline
from src.pipelines.query_pipeline import query_pipeline

logger = get_logger(__name__)

class RAGSystem:
    """
    A Retrieval-Augmented Generation (RAG) system for document question-answering.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the RAG system.
        
        Args:
            storage_path: Path to store vector database. If None, a temporary directory is created.
        """
        if storage_path:
            self.storage_path = storage_path
            self.vector_store_path = os.path.join(storage_path, "vector_store")
            self.temp_dir = None
            os.makedirs(self.vector_store_path, exist_ok=True)
        else:
            self.temp_dir = tempfile.mkdtemp()
            self.storage_path = self.temp_dir
            self.vector_store_path = os.path.join(self.temp_dir, "vector_store")
            os.makedirs(self.vector_store_path, exist_ok=True)
            
        self.documents_processed = False
        logger.info(f"Initialized RAG system with storage at {self.storage_path}")
        
    def process_documents(self, 
                          document_path: str, 
                          chunk_size: int = 1000, 
                          chunk_overlap: int = 200,
                          embedding_model: str = "all-MiniLM-L6-v2",
                          index_type: str = "L2") -> Dict[str, Any]:
        """
        Process a document or directory of documents and create a vector store.
        
        Args:
            document_path: Path to document or directory of documents
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
            embedding_model: Name of embedding model to use
            index_type: Type of FAISS index to use (L2 or IP)
            
        Returns:
            Dictionary with processing results
        """
        # Validate document path
        doc_path = Path(document_path)
        if not doc_path.exists():
            raise FileNotFoundError(f"Document path does not exist: {document_path}")
        
        # Run document processing pipeline
        processing_result = document_processing_pipeline(
            document_path=str(doc_path),
            output_dir=self.vector_store_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model_name=embedding_model,
            index_type=index_type
        )
        
        self.documents_processed = True
        
        return {
            "status": "success",
            "message": f"Processed {processing_result['num_documents']} document chunks",
            "num_chunks": processing_result['num_documents'],
            "vector_store_path": processing_result['vector_store_path']
        }
        
    def query(self, 
              query: str, 
              top_k: int = 5,
              hybrid_search: bool = True,
              embedding_model: str = "all-MiniLM-L6-v2") -> Dict[str, Any]:
        """
        Query processed documents.
        
        Args:
            query: Query string
            top_k: Number of results to return
            hybrid_search: Whether to use hybrid search (combine semantic and keyword)
            embedding_model: Name of embedding model to use
            
        Returns:
            Dictionary with query results
        """
        if not self.documents_processed:
            raise RuntimeError("No documents have been processed yet. Call process_documents first.")
        
        # Run query pipeline
        results = query_pipeline(
            query=query,
            vector_store_path=self.vector_store_path,
            embedding_model_name=embedding_model,
            top_k=top_k,
            hybrid_search=hybrid_search
        )
        
        # Format results for better readability
        formatted_results = []
        for i, res in enumerate(results):
            formatted_results.append({
                "rank": i + 1,
                "content": res["content"],
                "score": res["score"],
                "source": res["metadata"].get("source", "Unknown"),
                "page": res["metadata"].get("page", "N/A"),
                "chunk_index": res["metadata"].get("chunk_index", "N/A"),
                "metadata": res["metadata"]
            })
        
        return {
            "query": query,
            "results": formatted_results,
            "result_count": len(formatted_results)
        }
    
    def cleanup(self):
        """Clean up temporary files if used."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up temporary directory {self.temp_dir}")
            self.temp_dir = None
    
    def __del__(self):
        """Ensure cleanup on object destruction."""
        self.cleanup()


# Example usage
if __name__ == "__main__":
    # Initialize RAG system
    rag = RAGSystem()
    
    # Process documents
    result = rag.process_documents(
        document_path="./documents/",
        chunk_size=1000,
        chunk_overlap=200
    )
    print(f"Processed documents: {result}")
    
    # Query documents
    query_result = rag.query(
        query="What is the main topic of these documents?",
        top_k=3
    )
    
    # Print results
    print(f"\nQuery: {query_result['query']}")
    print(f"Found {query_result['result_count']} results:")
    
    for result in query_result['results']:
        print(f"\nRank {result['rank']} (Score: {result['score']:.4f})")
        print(f"Source: {result['source']}, Page: {result['page']}")
        print(f"Content: {result['content'][:200]}...")