"""
Text splitter module for chunking document text.
"""
from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

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