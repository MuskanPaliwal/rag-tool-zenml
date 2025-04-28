# ZenML RAG System

A Retrieval-Augmented Generation (RAG) system built with ZenML pipelines for document question-answering.

## Overview

This RAG system allows you to:

1. Process documents (PDFs, DOCx, TXT, HTML, etc.) into a vector database
2. Query the processed documents using natural language
3. Retrieve the most relevant document chunks for your queries

## Features

- **Multi-format Document Support**: Process PDFs, Word documents, text files, HTML, and more
- **Smart Text Chunking**: Split documents intelligently with customizable chunk sizes
- **Efficient Embedding**: Generate embeddings using SentenceTransformers models
- **Fast Vector Search**: Use FAISS for efficient similarity search
- **Hybrid Search**: Combine semantic search with keyword matching for better results
- **ZenML Integration**: Leverage ZenML for pipeline orchestration and reproducibility
- **CLI Interface**: Simple command-line interface for document processing and querying

## Project Structure

```
rag_system/
├── src/
│   ├── utils/
│   │   ├── documents_processor.py   # Document loading and processing
│   │   ├── text_splitter.py         # Text chunking
│   │   └── vector_utils.py          # Vector operations utilities
│   ├── models/
│   │   └── embeddings.py            # Embedding models
│   ├── data/
│   │   └── vector_store.py          # Vector storage and retrieval
│   └── pipelines/
│       ├── document_pipeline.py     # Document processing pipeline
│       └── query_pipeline.py        # Query pipeline
├── rag_system.py                    # Main RAG system interface
├── main.py                          # Command-line entry point
└── README.md                        # Documentation
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/rag-system.git
cd rag-system
```

2. Install the required dependencies:

```bash
pip install zenml langchain sentence-transformers faiss-cpu pypdf
```

3. For additional document format support:

```bash
pip install unstructured
```

## Usage

### Command Line Interface

The `main.py` script provides a simple command-line interface with three modes of operation:

```bash
# Process documents
python main.py process --document-path path/to/documents/ --storage-path ./vector_db

# Query documents
python main.py query --storage-path ./vector_db --query "What is the main topic of these documents?"

# Interactive mode (ask multiple questions)
python main.py interactive --storage-path ./vector_db
```

### Options

- `--document-path`, `-d`: Path to document or directory to process
- `--storage-path`, `-s`: Path to store the vector database (default: temporary directory)
- `--query`, `-q`: Query string for searching documents
- `--chunk-size`: Size of document chunks (default: 1000)
- `--chunk-overlap`: Overlap between chunks (default: 200)
- `--top-k`, `-k`: Number of results to return for queries (default: 3)
- `--embedding-model`, `-m`: Name of embedding model to use (default: "all-MiniLM-L6-v2")
- `--hybrid-search`: Use hybrid search combining semantic and keyword matching

### Programmatic Usage

You can also use the RAG system programmatically in your Python code:

```python
from rag_system import RAGSystem

# Initialize RAG system
rag = RAGSystem(storage_path="./vector_db")

# Process a document or directory
result = rag.process_documents(
    document_path="path/to/documents/",
    chunk_size=1000,
    chunk_overlap=200
)
print(f"Processed {result['num_chunks']} document chunks")

# Query the processed documents
answer = rag.query(
    query="What is the main topic discussed in these documents?",
    top_k=3,
    hybrid_search=True
)

# Print results
for result in answer['results']:
    print(f"Rank {result['rank']} (Score: {result['score']:.4f})")
    print(f"Content: {result['content']}")
    print(f"Source: {result['source']}")
```

## Customization

### Embedding Models

You can use different SentenceTransformers models by changing the `embedding_model` parameter:

- `all-MiniLM-L6-v2` (default): Fast and balanced
- `all-mpnet-base-v2`: Higher quality but slower
- `paraphrase-multilingual-MiniLM-L12-v2`: For multilingual support

### Vector Search

- Change `index_type` to "IP" (Inner Product) for cosine similarity instead of L2 distance
- Use `hybrid_search=True` to combine semantic search with keyword matching

### Document Chunking

- Modify `chunk_size` and `chunk_overlap` to optimize for your specific documents
- For longer documents, increase chunk size
- For technical documents, decrease chunk size and increase overlap

## Integration with LLMs

To create a complete RAG system, integrate with an LLM:

```python
from rag_system import RAGSystem
import openai  # or any other LLM API

# Initialize RAG system and process documents
rag = RAGSystem(storage_path="./vector_db")

# Process documents if needed
if not os.path.exists("./vector_db"):
    rag.process_documents("path/to/documents/")

# Query function with LLM integration
def answer_question(query, top_k=3):
    # Get relevant context from RAG system
    results = rag.query(query, top_k=top_k)
    
    # Prepare context for the LLM
    context = "\n\n".join([r["content"] for r in results["results"]])
    
    # Create prompt with context
    prompt = f"Answer the question based on the following context:\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    
    # Call LLM API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return {
        "answer": response.choices[0].message["content"],
        "sources": [r["source"] for r in results["results"]]
    }

# Example usage
result = answer_question("What are the key benefits described in the document?")
print(result["answer"])
print(f"Sources: {result['sources']}")
```

## Troubleshooting

### Common Issues

1. **FileNotFoundError**: Ensure the document path is correct and accessible.
2. **Memory Issues**: For large documents, reduce batch size or chunk size.
3. **CUDA Errors**: Set device to 'cpu' in the embeddings module if you encounter GPU-related errors.
4. **Unsupported File Types**: Ensure you have the necessary dependencies for all file types (e.g., `unstructured` for Word documents).

## License

This project is licensed under the MIT License.