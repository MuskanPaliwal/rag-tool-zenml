# RAG Tool with ZenML

A Retrieval-Augmented Generation (RAG) tool built with ZenML that generates embeddings from documents and answers questions about them. This tool can be integrated with Composio as a local tool.

## Features

- **Document Processing**: Extract text from PDFs and other document formats
- **Text Chunking**: Split documents into manageable segments for embedding
- **Embedding Generation**: Generate vector representations of text using sentence-transformers
- **Vector Storage**: Store and retrieve embeddings efficiently using FAISS
- **Question Answering**: Answer questions about documents using RAG approach
- **ZenML Integration**: Track experiments, manage artifacts, and ensure reproducibility
- **Composio Integration**: Expose RAG functionality as a local tool for Composio

## Architecture

The project is structured as follows:

```
rag-tool-zenml/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── composio_tool/
│   │   ├── __init__.py
│   │   └── rag_tool.py           # Composio tool implementation
│   ├── data/
│   │   ├── __init__.py
│   │   └── vector_store.py       # Vector database management
│   ├── models/
│   │   ├── __init__.py
│   │   └── embeddings.py         # Embedding model management
│   └── utils/
│       ├── __init__.py
│       ├── document_processor.py # Document processing utilities
│       └── text_splitter.py      # Text chunking utilities
├── pipelines/
│   ├── __init__.py
│   ├── document_pipeline.py      # Document ingestion pipeline
│   └── query_pipeline.py         # Query processing pipeline
└── run.py                        # Main entry point
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/MuskanPaliwal/rag-tool-zenml.git
cd rag-tool-zenml
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your OpenAI API key and other configurations
```

## Usage

### Document Ingestion

To ingest a document and generate embeddings:

```bash
python run.py ingest /path/to/document.pdf
```

Options:
- `--chunk-size`: Size of each text chunk (default: 1000)
- `--chunk-overlap`: Overlap between chunks (default: 200)
- `--embedding-model`: Name of the embedding model (default: all-MiniLM-L6-v2)
- `--vector-store-dir`: Directory to save the vector store (default: vector_store)

### Question Answering

To ask a question about an ingested document:

```bash
python run.py query "What is the main topic of the document?"
```

Options:
- `--vector-store-dir`: Directory of the vector store (default: vector_store)
- `--embedding-model`: Name of the embedding model (default: all-MiniLM-L6-v2)
- `--k`: Number of results to return (default: 5)

### Using as a Composio Tool

1. Install Composio:
```bash
pip install composio_core
```

2. Set up your Composio API key:
```bash
export COMPOSIO_API_KEY=your_api_key
```

3. Import and use the RAG tool in your Composio application:
```python
from src.composio_tool.rag_tool import rag_tool

# Register the tool with your Composio application
```

## Why ZenML?

ZenML provides several advantages for this RAG tool:

1. **Pipeline Management**: ZenML allows us to create modular, reusable pipelines for document processing and question answering.

2. **Experiment Tracking**: Track different embedding models and RAG configurations to optimize performance.

3. **Artifact Management**: Store and version document embeddings efficiently.

4. **Reproducibility**: Make the RAG tool reproducible across different environments.

5. **Integration Capabilities**: ZenML integrates with various embedding models, vector databases, and LLMs.

6. **Deployment Options**: Easily deploy the RAG tool as a service.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.