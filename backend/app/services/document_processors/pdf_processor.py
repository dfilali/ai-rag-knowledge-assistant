import os
from typing import List
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.app.core import config
from backend.app.core.logging import setup_logger

logger = setup_logger("pdf_processor")

def extract_text_from_pdf(file_path: str) -> List[dict]:
    """
    Extracts text from a PDF file page by page.
    Returns a list of dicts: [{"page": page_num, "text": page_text}]
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    pages_content = []
    try:
        reader = PdfReader(file_path)
        for idx, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                pages_content.append({
                    "page": idx + 1,
                    "text": text.strip()
                })
    except Exception as e:
        logger.error(f"Failed to parse PDF file {file_path}: {e}")
        raise RuntimeError(f"Failed to parse PDF file: {str(e)}")

    return pages_content

def process_document(file_path: str, filename: str) -> List[Document]:
    """
    Extracts text from a PDF file, splits it into chunks, and attaches metadata.
    Returns a list of LangChain Document objects.
    """
    pages_data = extract_text_from_pdf(file_path)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len
    )
    
    documents = []
    chunk_counter = 0
    
    for page_info in pages_data:
        page_num = page_info["page"]
        text = page_info["text"]
        
        # Split text of individual page to ensure page-level metadata stays accurate
        chunks = text_splitter.split_text(text)
        
        for chunk in chunks:
            if not chunk.strip():
                continue
                
            metadata = {
                "source": filename,
                "page": page_num,
                "chunk_idx": chunk_counter,
                "char_length": len(chunk)
            }
            
            documents.append(Document(page_content=chunk, metadata=metadata))
            chunk_counter += 1
            
    logger.info(f"Ingested {filename}: split into {chunk_counter} chunks")
    return documents
