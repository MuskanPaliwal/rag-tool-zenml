"""
Document processor module for extracting text from various document formats.
"""
import os
from typing import List, Dict, Any, Optional
import pypdf
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.schema import Document

class DocumentProcessor:
    """
    A class to process documents of various formats and extract text.
    Currently supports PDF files and directories containing PDFs.
    """
    
    @staticmethod
    def process_pdf(file_path: str) -> List[Document]:
        """
        Process a PDF file and extract text.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of Document objects containing text and metadata
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not file_path.lower().endswith('.pdf'):
            raise ValueError(f"File is not a PDF: {file_path}")
        
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Add source metadata
        for doc in documents:
            doc.metadata["source"] = file_path
            doc.metadata["file_type"] = "pdf"
        
        return documents
    
    @staticmethod
    def process_directory(dir_path: str, glob_pattern: str = "**/*.pdf") -> List[Document]:
        """
        Process all PDF files in a directory.
        
        Args:
            dir_path: Path to the directory
            glob_pattern: Pattern to match files
            
        Returns:
            List of Document objects containing text and metadata
        """
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        
        if not os.path.isdir(dir_path):
            raise ValueError(f"Path is not a directory: {dir_path}")
        
        loader = DirectoryLoader(
            dir_path, 
            glob=glob_pattern,
            loader_cls=PyPDFLoader
        )
        
        documents = loader.load()
        
        # Add source metadata
        for doc in documents:
            doc.metadata["source_dir"] = dir_path
            doc.metadata["file_type"] = "pdf"
        
        return documents
    
    @staticmethod
    def get_document_metadata(file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a PDF document.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary containing document metadata
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not file_path.lower().endswith('.pdf'):
            raise ValueError(f"File is not a PDF: {file_path}")
        
        with open(file_path, 'rb') as f:
            pdf = pypdf.PdfReader(f)
            info = pdf.metadata
            num_pages = len(pdf.pages)
        
        metadata = {
            "title": info.title if info.title else os.path.basename(file_path),
            "author": info.author if info.author else "Unknown",
            "num_pages": num_pages,
            "file_path": file_path,
            "file_type": "pdf"
        }
        
        return metadata