"""
Text splitter module for chunking document text.
"""
from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from zenml.steps import step

class TextSplitter:
    """
    A class to split text documents into chunks suitable for embedding.
    """
    
    def __init__(
        self, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200
    ):
        """
        Initialize the text splitter.
        
        Args:
            chunk_size: The size of each text chunk
            chunk_overlap: The overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split a list of documents into chunks.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of chunked Document objects
        """
        return self.text_splitter.split_documents(documents)
    
    def split_text(self, text: str) -> List[str]:
        """
        Split a text string into chunks.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        return self.text_splitter.split_text(text)


# ZenML steps for text splitting
@step
def initialize_text_splitter(
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> TextSplitter:
    """
    ZenML step to initialize the text splitter.
    
    Args:
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        Initialized TextSplitter
    """
    return TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


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
    text_splitter: TextSplitter
) -> List[str]:
    """
    ZenML step to split texts into chunks.
    
    Args:
        texts: List of text strings
        text_splitter: TextSplitter instance
        
    Returns:
        List of text chunks
    """
    chunks = []
    for text in texts:
        chunks.extend(text_splitter.split_text(text))
    return chunks