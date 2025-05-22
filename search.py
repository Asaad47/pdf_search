#!/usr/bin/env python3
import logging
import argparse
import yaml
import shutil
import time
import readchar
import os
import sys
from typing import List
import logging
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from rich.console import Console, Group
from rich.markdown import Markdown
from io import StringIO
from rich.panel import Panel
from rich.live import Live
from create_db import find_pdf_files

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

def search_documents(query: str, k: int = DEFAULT_K, exclude_classes: List[str] = None) -> List[Document]:
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
        start_time = time.time()
        if exclude_classes:
            results = vectorstore.similarity_search(query, k=k, filter={"source": {"$nin": exclude_classes}})
        else:
            results = vectorstore.similarity_search(query, k=k)
        end_time = time.time()
        logging.info(f"⏰ Search completed in {end_time - start_time} seconds")
        
        if not results:
            logging.warning("No results found for the query.")
            return []
            
        return results
        
    except Exception as e:
        logging.error(f"Error during search: {str(e)}")
        sys.exit(1)
    

def get_terminal_width() -> int:
    """Get the width of the terminal window."""
    try:
        columns, _ = shutil.get_terminal_size()
        return columns
    except:
        return 80  # fallback to default width

def format_slide_content(content: str) -> str:
    """Convert markdown content to ANSI text using rich."""
    sio = StringIO()
    console = Console(file=sio, force_terminal=True, width=get_terminal_width())
    console.print(Markdown(content))
    return sio.getvalue()


def interactive_slide_viewer(results, query):
    console = Console()
    i = 0
    total = len(results)
    
    def render_slide(index):
        doc = results[index]
        metadata = doc.metadata
        title = f"[{index + 1}/{total}] [bold blue]Slide {metadata.get('page')} - {metadata.get('source')}[/bold blue]"
        slide_md = Markdown(doc.page_content)
        query_bar = f"[bold green]🔍 Search query:[/] {query}"
        seperator = "[cyan]" + "─" * (get_terminal_width() - 4) + "[/cyan]"
        footer = "[yellow]Press: \[n]ext, \[p]rev, \[o]pen PDF, \[q]uit[/yellow]"
        content = Group(query_bar, seperator, slide_md, "")
        return Panel(content, title=title, subtitle=footer, border_style="cyan")

    with Live(render_slide(i), refresh_per_second=10, console=console) as live:
        while True:
            key = readchar.readkey()
            if key == 'n':
                i = (i + 1) % total
                live.update(render_slide(i))
            elif key == 'p':
                i = (i - 1) % total
                live.update(render_slide(i))
            elif key == 'o':
                doc = results[i]
                pdf_path = doc.metadata.get("source")
                if pdf_path:
                    os.system(f'open -a Preview "{pdf_path}"')  # macOS
            elif key == 'q':
                break
            else:
                console.print(f"[red]Unknown key:[/] {key}")


def main():
    """Main function to run the search."""
    
    parser = argparse.ArgumentParser(description='Search through PDF documents using semantic search.')
    parser.add_argument('query', nargs='?', default=DEFAULT_QUERY,
                      help=f'The search query (default: "{DEFAULT_QUERY}")')
    parser.add_argument('-k', type=int, default=DEFAULT_K, 
                      help=f'Number of results to return (default: {DEFAULT_K})')
    parser.add_argument('-v', action='store_true', help='Verbose output')
    parser.add_argument('-i', action='store_true', help='Interactive mode')
    parser.add_argument('-x', type=str, help='classes to exclude from search (comma separated)')
    args = parser.parse_args()
    
    try:
        if args.x:
            classes_to_exclude = args.x.split(',')
            exclude_patterns = []
            # get classes from config.yaml
            pdf_paths = config['pdf_paths']
            for class_to_exclude in classes_to_exclude:
                for path in pdf_paths:
                    if class_to_exclude in path:
                        exclude_patterns.append(path)
                        break
            print(exclude_patterns)
            
            exclude_classes = find_pdf_files(exclude_patterns)
            print(exclude_classes)
        else:
            exclude_classes = None
        results = search_documents(args.query, args.k, exclude_classes)
        
        if args.i:
            interactive_slide_viewer(results, args.query)
        else:
            print("\n=== 🔍 Top Relevant Results ===\n")
            for doc in results:
                print(format_search_result(doc, args.v))
            print(f"\nFound {len(results)} relevant results.")
            
    except KeyboardInterrupt:
        logging.info("\nSearch interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

