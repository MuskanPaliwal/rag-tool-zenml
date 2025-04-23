# pipelines/__init__.py
from .document_pipeline import document_processing_pipeline
from .query_pipeline import query_pipeline

__all__ = [
    'document_processing_pipeline',
    'query_pipeline'
]