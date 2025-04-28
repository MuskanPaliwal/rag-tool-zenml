"""
Main script for the RAG system.
Provides command-line interface for document processing and querying.
"""
import argparse
import os
import sys

from src.rag_system import RAGSystem

def main():
    """Main function for the RAG system."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="RAG System - Document Q&A")
    
    # Command argument (process or query)
    parser.add_argument("command", choices=["process", "query", "interactive"], 
                        help="Command to execute (process documents, query, or interactive mode)")
    
    # Document path (for processing)
    parser.add_argument("--document-path", "-d", type=str, default=None,
                        help="Path to document or directory for processing")
    
    # Query (for querying)
    parser.add_argument("--query", "-q", type=str, default=None,
                        help="Query string for searching documents")
    
    # Options
    parser.add_argument("--storage-path", "-s", type=str, default=None,
                        help="Path to store vector database")
    parser.add_argument("--chunk-size", type=int, default=1000,
                        help="Size of document chunks")
    parser.add_argument("--chunk-overlap", type=int, default=200,
                        help="Overlap between chunks")
    parser.add_argument("--top-k", "-k", type=int, default=3,
                        help="Number of results to return for queries")
    parser.add_argument("--embedding-model", "-m", type=str, default="all-MiniLM-L6-v2",
                        help="Name of embedding model to use")
    parser.add_argument("--hybrid-search", action="store_true",
                        help="Use hybrid search (combine semantic and keyword)")
    
    args = parser.parse_args()
    
    # Initialize RAG system
    rag = RAGSystem(storage_path=args.storage_path)
    
    try:
        # Process documents
        if args.command == "process":
            if not args.document_path:
                print("Error: --document-path is required for 'process' command")
                return
                
            print(f"Processing documents at {args.document_path}...")
            result = rag.process_documents(
                document_path=args.document_path,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                embedding_model=args.embedding_model
            )
            print(f"Processing complete: {result['message']}")
            print(f"Number of chunks: {result['num_chunks']}")
            print(f"Vector store path: {result['vector_store_path']}")
        
        # Query documents
        elif args.command == "query":
            if not args.query:
                print("Error: --query is required for 'query' command")
                return
                
            print(f"Querying: '{args.query}'")
            query_result = rag.query(
                query=args.query,
                top_k=args.top_k,
                hybrid_search=args.hybrid_search,
                embedding_model=args.embedding_model
            )
            
            # Print results
            print(f"Found {query_result['result_count']} results:\n")
            
            for result in query_result['results']:
                print(f"Rank {result['rank']} (Score: {result['score']:.4f})")
                print(f"Source: {result['source']}")
                if 'page' in result and result['page'] != 'N/A':
                    print(f"Page: {result['page']}")
                print(f"Content: {result['content'][:300]}...")
                print("-" * 80)
        
        # Interactive mode
        elif args.command == "interactive":
            if not args.storage_path:
                print("Error: --storage-path is required for interactive mode")
                return
                
            if not os.path.exists(args.storage_path):
                if not args.document_path:
                    print("Error: Vector store doesn't exist. Please provide --document-path to process documents first")
                    return
                    
                print(f"Processing documents at {args.document_path}...")
                result = rag.process_documents(
                    document_path=args.document_path,
                    chunk_size=args.chunk_size,
                    chunk_overlap=args.chunk_overlap,
                    embedding_model=args.embedding_model
                )
                print(f"Processing complete: {result['message']}")
            
            print("\n========== RAG Interactive Mode ==========")
            print("Type 'exit', 'quit', or 'q' to exit")
            print("==========================================\n")
            
            while True:
                query = input("\nEnter your question: ")
                
                if query.lower() in ['exit', 'quit', 'q']:
                    break
                    
                if not query.strip():
                    continue
                
                query_result = rag.query(
                    query=query,
                    top_k=args.top_k,
                    hybrid_search=args.hybrid_search,
                    embedding_model=args.embedding_model
                )
                
                # Print results
                print(f"\nFound {query_result['result_count']} results:\n")
                
                for result in query_result['results']:
                    print(f"Rank {result['rank']} (Score: {result['score']:.4f})")
                    print(f"Source: {result['source']}")
                    if 'page' in result and result['page'] != 'N/A':
                        print(f"Page: {result['page']}")
                    print(f"Content: {result['content'][:300]}...")
                    print("-" * 80)
                
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Error: {str(e)}")
        return 1
    finally:
        # Only clean up if using temporary storage
        if not args.storage_path:
            rag.cleanup()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
