"""
Text splitter module for chunking document text.
"""
from typing import List, Optional, Dict, Any
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from zenml.steps import step
from zenml.logger import get_logger

logger = get_logger(__name__)

class TextSplitter:
    """
    A class to split text documents into chunks suitable for embedding.
    """
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize the text splitter.
        
        Args:
            chunk_size: The size of each text chunk
            chunk_overlap: The overlap between chunks
            separators: Custom separators to use for splitting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        if separators is None:
            separators = ["\n\n", "\n", " ", ""]
            
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=separators
        )
        
        logger.info("Initialized TextSplitter with chunk_size=%d, chunk_overlap=%d", chunk_size, chunk_overlap)
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split a list of documents into chunks while preserving metadata.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of chunked Document objects with enhanced metadata
        """
        logger.info("Splitting %d documents into chunks", len(documents))
        chunks = self.text_splitter.split_documents(documents)
        
        # Enhance metadata with chunk information
        for i, chunk in enumerate(chunks):
            # Add chunk index and total chunks to metadata
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)
            
        logger.info("Created %d chunks from %d documents", len(chunks), len(documents))
        return chunks
    
    def split_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Split a text string into chunks and convert to Documents with metadata.
        
        Args:
            text: Text to split
            metadata: Optional metadata to include with each chunk
            
        Returns:
            List of Document objects with text chunks and metadata
        """
        if metadata is None:
            metadata = {}
            
        text_chunks = self.text_splitter.split_text(text)
        documents = []
        
        for i, chunk in enumerate(text_chunks):
            # Create a new metadata dict for each chunk
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = i
            chunk_metadata["total_chunks"] = len(text_chunks)
            
            documents.append(Document(page_content=chunk, metadata=chunk_metadata))
            
        logger.info("Split text into %d chunks", len(documents))
        return documents


# ZenML steps for text splitting
@step
def initialize_text_splitter(
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: Optional[List[str]] = None
) -> TextSplitter:
    """
    ZenML step to initialize the text splitter.
    
    Args:
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        separators: Optional custom separators
        
    Returns:
        Initialized TextSplitter
    """
    return TextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=separators
    )

@step
def split_documents(
    documents: List[Document],
    text_splitter: TextSplitter
) -> List[Document]:
    """
    ZenML step to split documents into chunks.
    
    Args:
        documents: List of Document objects
        text_splitter: TextSplitter instance
        
    Returns:
        List of chunked Document objects
    """
    return text_splitter.split_documents(documents)

@step
def split_texts(
    texts: List[str],
    text_splitter: TextSplitter,
    metadata: Optional[Dict[str, Any]] = None
) -> List[Document]:
    """
    ZenML step to split texts into chunks and convert to Documents.
    
    Args:
        texts: List of text strings
        text_splitter: TextSplitter instance
        metadata: Optional metadata to include with each chunk
        
    Returns:
        List of Document objects with text chunks and metadata
    """
    all_chunks = []
    
    for i, text in enumerate(texts):
        # Create metadata specific to this text if not provided
        text_metadata = metadata.copy() if metadata else {}
        text_metadata["text_index"] = i
        text_metadata["total_texts"] = len(texts)
        
        # Split the text and add to results
        chunks = text_splitter.split_text(text, text_metadata)
        all_chunks.extend(chunks)
        
    logger.info("Split %d texts into %d document chunks", len(texts), len(all_chunks))
    return all_chunks