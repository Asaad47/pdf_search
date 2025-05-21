#!/usr/bin/env python3
import os
import sys
import logging
import argparse
import yaml
from typing import List
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

def load_config() -> dict:
    """Load configuration from YAML file."""
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        logging.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
        
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Error loading configuration: {str(e)}")
        sys.exit(1)

# ==== Config ====
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
config = load_config()
CHROMA_DIR = config['chroma_dir']
DEFAULT_QUERY = config['default_query']
DEFAULT_K = config['default_k_results']

def format_search_result(doc: Document, verbose: bool = False, max_length: int = 300) -> str:
    """Format a single search result for display."""
    content = doc.page_content
    if len(content) > max_length:
        content = content[:max_length] + "..."
        
    if verbose:
        return f"[Slide {doc.metadata['page']} - {doc.metadata['file_path']}]\n{content}\n---"
    else:
        return f"[Slide {doc.metadata['page']} - {doc.metadata['file_path']}]"

def search_documents(query: str, k: int = DEFAULT_K) -> List[Document]:
    """Perform semantic search on the vector database."""
    try:
        # Check if database exists
        db_path = os.path.join(CHROMA_DIR, "chroma.sqlite3")
        if not os.path.exists(db_path):
            logging.error(f"Database not found at {CHROMA_DIR}. Please run create_db.py first.")
            sys.exit(1)

        # Initialize embedding model and vector store
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding_model)
        
        # Perform search
        logging.info(f"Searching for: {query}")
        results = vectorstore.similarity_search(query, k=k)
        
        if not results:
            logging.warning("No results found for the query.")
            return []
            
        return results
        
    except Exception as e:
        logging.error(f"Error during search: {str(e)}")
        sys.exit(1)

def main():
    """Main function to run the search."""
    
    parser = argparse.ArgumentParser(description='Search through PDF documents using semantic search.')
    parser.add_argument('query', nargs='?', default=DEFAULT_QUERY,
                      help=f'The search query (default: "{DEFAULT_QUERY}")')
    parser.add_argument('-k', type=int, default=DEFAULT_K, 
                      help=f'Number of results to return (default: {DEFAULT_K})')
    parser.add_argument('-v', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    try:
        results = search_documents(args.query, args.k)
        
        if results:
            print("\n=== 🔍 Top Relevant Results ===\n")
            for doc in results:
                print(format_search_result(doc, args.v))
            print(f"\nFound {len(results)} relevant results.")
        else:
            print("\nNo relevant results found. Try rephrasing your query.")
            
    except KeyboardInterrupt:
        logging.info("\nSearch interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

