"""
Enhanced document processor module for extracting text from various document formats.
"""
import os
from typing import List, Dict, Any
import re
import pypdf
from langchain.document_loaders import (
    PyPDFLoader, 
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader
)
from langchain.schema import Document
from zenml.steps import step
from zenml.logger import get_logger

logger = get_logger(__name__)

class DocumentProcessor:
    """
    A class to process documents of various formats and extract text.
    Supports multiple file formats including PDF, DOCX, TXT, HTML, MD, and PPT.
    """
    
    # Map file extensions to appropriate loaders
    LOADER_MAPPING = {
        ".pdf": PyPDFLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".doc": UnstructuredWordDocumentLoader,
        ".txt": TextLoader,
        ".html": UnstructuredHTMLLoader,
        ".htm": UnstructuredHTMLLoader,
        ".md": UnstructuredMarkdownLoader,
        ".pptx": UnstructuredPowerPointLoader,
        ".ppt": UnstructuredPowerPointLoader,
    }
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and preprocess extracted text.
        
        Args:
            text: Raw text extracted from documents
            
        Returns:
            Cleaned text
        """
        # Replace multiple newlines with a single one
        text = re.sub(r'\n+', '\n', text)
        # Replace multiple spaces with a single one
        text = re.sub(r'\s+', ' ', text)
        # Remove any non-printable characters
        text = re.sub(r'[^\x20-\x7E\n]', '', text)
        return text.strip()
    
    @classmethod
    def get_loader_for_file(cls, file_path: str):
        """
        Get the appropriate loader for a file based on its extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Loader class for the file type
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in cls.LOADER_MAPPING:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        return cls.LOADER_MAPPING[file_ext]
    
    @classmethod
    def process_file(cls, file_path: str) -> List[Document]:
        """
        Process a file and extract text.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of Document objects containing text and metadata
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in cls.LOADER_MAPPING:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        try:
            loader_cls = cls.get_loader_for_file(file_path)
            loader = loader_cls(file_path)
            documents = loader.load()
            
            # Add source metadata and clean text
            for doc in documents:
                doc.metadata["source"] = file_path
                doc.metadata["file_type"] = file_ext[1:]  # Remove the dot
                # Clean the text
                doc.page_content = cls.clean_text(doc.page_content)
            
            logger.info("Successfully processed %s, extracted %d document chunks", file_path, len(documents))
            return documents
        except Exception as e:
            logger.error("Error processing %s: %s", file_path, str(e))
            raise
    
    @classmethod
    def process_directory(cls, dir_path: str, glob_pattern: str = "**/*.*") -> List[Document]:
        """
        Process all supported files in a directory.
        
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
        
        all_documents = []
        supported_extensions = list(cls.LOADER_MAPPING.keys())
        
        # Walk through directory and process each supported file
        for root, _, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                
                if file_ext in supported_extensions:
                    try:
                        documents = cls.process_file(file_path)
                        all_documents.extend(documents)
                    except Exception as e:
                        logger.warning("Skipping %s: %s", file_path, str(e))
        
        logger.info("Processed directory %s, found %d document chunks", dir_path, len(all_documents))
        return all_documents
    
    @staticmethod
    def get_document_metadata(file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a document.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Dictionary containing document metadata
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        metadata = {
            "file_path": file_path,
            "file_type": file_ext[1:],  # Remove the dot
            "file_size": os.path.getsize(file_path),
            "modified_date": os.path.getmtime(file_path),
        }
        
        # Additional metadata for PDFs
        if file_ext == ".pdf":
            try:
                with open(file_path, 'rb') as f:
                    pdf = pypdf.PdfReader(f)
                    info = pdf.metadata
                    num_pages = len(pdf.pages)
                
                metadata.update({
                    "title": info.title if hasattr(info, 'title') and info.title else os.path.basename(file_path),
                    "author": info.author if hasattr(info, 'author') and info.author else "Unknown",
                    "num_pages": num_pages,
                })
            except Exception as e:
                logger.warning("Error extracting PDF metadata: %s", str(e))
        
        return metadata


# ZenML steps for document processing
@step
def process_document(file_path: str) -> List[Document]:
    """
    ZenML step to process a document or directory of documents.
    
    Args:
        file_path: Path to the document or directory
        
    Returns:
        List of Document objects
    """
    if os.path.isdir(file_path):
        return DocumentProcessor.process_directory(file_path)
    else:
        return DocumentProcessor.process_file(file_path)


@step
def extract_texts_from_documents(documents: List[Document]) -> List[str]:
    """
    ZenML step to extract text from documents.
    
    Args:
        documents: List of Document objects
        
    Returns:
        List of text strings
    """
    return [doc.page_content for doc in documents]


@step
def get_document_metadata(file_path: str) -> Dict[str, Any]:
    """
    ZenML step to extract metadata from a document.
    
    Args:
        file_path: Path to the document
        
    Returns:
        Dictionary containing document metadata
    """
    return DocumentProcessor.get_document_metadata(file_path)