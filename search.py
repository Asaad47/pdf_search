#!/usr/bin/env python3
import os
import sys
import logging
import argparse
import yaml
import shutil
from typing import List
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import time
from prompt_toolkit.formatted_text import HTML, FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.application import Application
from prompt_toolkit.layout import Layout
from prompt_toolkit.widgets import TextArea, Frame, Label
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.styles import Style

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
        start_time = time.time()
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
    

def format_slide_content(content: str, max_width: int) -> str:
    """Format slide content with proper wrapping and indentation."""
    # Split into lines and process each line
    lines = content.split('\n')
    formatted_lines = []
    for line in lines:
        # Remove leading/trailing whitespace
        line = line.strip()
        if not line:
            formatted_lines.append('')
            continue
            
        # Handle bullet points
        if line.startswith('▶'):
            # Add proper indentation for bullet points
            formatted_lines.append('  • ' + line[1:].strip())
        else:
            formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)


def interactive_slide_viewer(results, query):
    i = 0
    terminal_width = get_terminal_width()

    def get_slide_text(index):
        doc = results[index]
        content = format_slide_content(doc.page_content, terminal_width)
        return content

    # Create styled components
    tooltip = TextArea(
        text="Press [n]ext, [p]rev, [o]pen PDF, or [q]uit",
        style="class:tooltip",
        wrap_lines=True,
        read_only=True,
        height=1
    )
    
    query_label = TextArea(
        text=f"🔍 Search query: {query}",
        style="class:query",
        wrap_lines=True,
        read_only=True,
        height=1
    )
    
    separator = TextArea(
        text="─" * (terminal_width - 4),
        style="class:separator",
        read_only=True,
        height=1
    )
    
    header = TextArea(
        text=f"[{i+1}/{len(results)}] Slide {results[i].metadata.get('page')} - {results[i].metadata.get('source')}",
        style="class:header",
        wrap_lines=True,
        read_only=True,
        height=1
    )
    
    # Main content area
    text_area = TextArea(
        text=get_slide_text(i),
        scrollbar=True,
        wrap_lines=True,
        read_only=True,
        line_numbers=True,
        style="class:content"
    )
    
    def update_content():
        nonlocal i
        text_area.text = get_slide_text(i)
        header.text = f"[{i+1}/{len(results)}] Slide {results[i].metadata.get('page')} - {results[i].metadata.get('source')}"
    
    # Create the layout with styled components
    layout = Layout(
        HSplit([
            tooltip,
            query_label,
            separator,
            header,
            separator,
            text_area
        ])
    )

    # Key bindings
    kb = KeyBindings()

    @kb.add('n')
    def next_slide(event):
        nonlocal i
        i = (i + 1) % len(results)
        update_content()

    @kb.add('p')
    def previous_slide(event):
        nonlocal i
        i = (i - 1) % len(results)
        update_content()

    @kb.add('q')
    def exit_viewer(event):
        event.app.exit()
        
    @kb.add('o')
    def open_pdf(event):
        doc = results[i]
        pdf_path = doc.metadata.get('source')
        page = doc.metadata.get('page')
        os.system(f"open -a Preview {pdf_path}")

    # Define styles
    style = Style([
        ('tooltip', 'fg:ansiyellow italic'),
        ('query', 'fg:ansigreen bold'),
        ('separator', 'fg:ansiblue'),
        ('header', 'fg:ansicyan bold'),
        ('content', 'fg:ansiwhite'),
    ])

    # Launch app with custom styles
    app = Application(
        layout=layout,
        key_bindings=kb,
        full_screen=True,
        style=style
    )
    app.run()



def main():
    """Main function to run the search."""
    
    parser = argparse.ArgumentParser(description='Search through PDF documents using semantic search.')
    parser.add_argument('query', nargs='?', default=DEFAULT_QUERY,
                      help=f'The search query (default: "{DEFAULT_QUERY}")')
    parser.add_argument('-k', type=int, default=DEFAULT_K, 
                      help=f'Number of results to return (default: {DEFAULT_K})')
    parser.add_argument('-v', action='store_true', help='Verbose output')
    parser.add_argument('-i', action='store_true', help='Interactive mode')
    args = parser.parse_args()
    
    try:
        results = search_documents(args.query, args.k)
        
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

