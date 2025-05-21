import os
import logging
import glob
import yaml
import shutil
import pymupdf
from pymupdf4llm import to_markdown
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import time

def load_config() -> dict:
    """Load configuration from YAML file."""
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        logging.error(f"Configuration file not found: {config_path}")
        exit(1)
        
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Error loading configuration: {str(e)}")
        exit(1)

# ==== Config ====
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
config = load_config()
PDF_PATHS = config['pdf_paths']
CHROMA_DIR = config['chroma_dir']

def find_pdf_files(patterns: list[str]) -> list[str]:
    """Find all PDF files matching the given glob patterns."""
    pdf_files = []
    for pattern in patterns:
        # Expand glob pattern
        matches = glob.glob(pattern, recursive=True)
        if not matches:
            logging.warning(f"No PDF files found matching pattern: {pattern}")
            continue
        pdf_files.extend(matches)
    
    # Remove duplicates while preserving order
    return list(dict.fromkeys(pdf_files))

def load_pdf_as_markdown(pdf_path: str) -> list[Document]:
    """Load a PDF file and convert it to markdown documents."""
    try:
        doc = pymupdf.open(pdf_path)
        # Get total number of pages
        total_pages = len(doc)
        
        # Convert each page to markdown
        pages = to_markdown(doc, page_chunks=True)
        documents = [Document(
            page_content=page['text'], 
            metadata={
                'source': pdf_path, 
                'page': page['metadata']['page'], 
                'total_pages': total_pages
            }
        ) for page in pages]
            
        return documents
    except Exception as e:
        logging.error(f"Error processing PDF {pdf_path}: {str(e)}")
        return []

# ==== Step 1: Load and parse all PDFs ====
documents = []
pdf_files = find_pdf_files(PDF_PATHS)

if not pdf_files:
    logging.error("No PDF files found. Please check your PDF_PATHS configuration.")
    exit(1)

logging.info(f"Found {len(pdf_files)} PDF files to process")
start_time = time.time()
for pdf_path in pdf_files:
    try:
        doc_slides = load_pdf_as_markdown(pdf_path)
        if doc_slides:
            documents.extend(doc_slides)
            logging.info(f"✅ Extracted {len(doc_slides)} pages from: {pdf_path}")
    except Exception as e:
        logging.error(f"❌ Error loading PDF {pdf_path}: {str(e)}")
        continue
end_time = time.time()
logging.info(f"⏰ Extracted {len(documents)} pages from {len(pdf_files)} PDFs in {end_time - start_time} seconds")

if not documents:
    logging.error("No documents were loaded. Please check if PDF files exist and are valid.")
    exit(1)

# ==== Step 2: Create vector database ====
try:
    # Create directory if it doesn't exist
    os.makedirs(CHROMA_DIR, exist_ok=True)
    
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if not os.path.exists(os.path.join(CHROMA_DIR, "chroma.sqlite3")):
        logging.info("Creating new database")
    else:
        logging.info("Overwriting existing database")
        # Delete existing database directory and all its contents
        logging.info(f"Deleting existing database at: {CHROMA_DIR}")
        shutil.rmtree(CHROMA_DIR)
        os.makedirs(CHROMA_DIR, exist_ok=True)

    start_time = time.time()
    vectorstore = Chroma.from_documents(documents, embedding_model, persist_directory=CHROMA_DIR)
    end_time = time.time()
    logging.info(f"⏰ Database created and persisted successfully in {end_time - start_time} seconds")
    
except Exception as e:
    logging.error(f"Error creating vector database: {str(e)}")
    exit(1)


