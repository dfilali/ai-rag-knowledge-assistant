import os
from typing import List
from fastapi import APIRouter, UploadFile, File, Header, HTTPException, Depends
from backend.app.core import config
from backend.app.api.schemas.document import DocumentInfo, DocumentUploadResponse, DocumentDeleteResponse
from backend.app.api.routes.dependencies import resolve_credentials
from backend.app.services.document_processors import pdf_processor
from backend.app.services.vectorstores import faiss_store
from backend.app.core.logging import setup_logger

logger = setup_logger("routes.documents")
router = APIRouter()

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    creds: tuple[str, str] = Depends(resolve_credentials)
):
    """
    Uploads a PDF, processes text, chunks it, and adds it to the vector store.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    provider, api_key = creds
    filename = file.filename
    file_path = os.path.join(config.UPLOAD_DIR, filename)
    
    # Check if file with same name already exists to prevent duplicate indexes
    if os.path.exists(file_path):
         raise HTTPException(status_code=400, detail=f"A document named '{filename}' already exists.")

    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
            
        # Parse and chunk document
        chunks = pdf_processor.process_document(file_path, filename)
        
        # Index document in FAISS
        faiss_store.add_documents_to_store(chunks, provider, api_key)
        
        logger.info(f"Ingested and indexed '{filename}' successfully.")
        return DocumentUploadResponse(
            filename=filename,
            chunks_count=len(chunks),
            status="success",
            message=f"Successfully indexed '{filename}' ({len(chunks)} chunks)."
        )
    except Exception as e:
        # Cleanup uploaded file if index fails
        if os.path.exists(file_path):
            os.remove(file_path)
        logger.error(f"Failed to process PDF upload: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

@router.get("/documents", response_model=List[DocumentInfo])
def list_documents():
    """
    Lists all ingested PDF documents.
    """
    if not os.path.exists(config.UPLOAD_DIR):
        return []
        
    documents = []
    files = [f for f in os.listdir(config.UPLOAD_DIR) if f.lower().endswith(".pdf")]
    
    for filename in files:
        path = os.path.join(config.UPLOAD_DIR, filename)
        stats = os.stat(path)
        documents.append(DocumentInfo(
            filename=filename,
            size_bytes=stats.st_size,
            created_at=stats.st_mtime
        ))
        
    # Sort by creation date descending
    documents.sort(key=lambda x: x.created_at, reverse=True)
    return documents

@router.delete("/documents/{filename}", response_model=DocumentDeleteResponse)
def delete_document(
    filename: str,
    creds: tuple[str, str] = Depends(resolve_credentials)
):
    """
    Deletes an uploaded PDF and rebuilds the vector index.
    """
    file_path = os.path.join(config.UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")
        
    provider, api_key = creds
    
    try:
        # Delete file
        os.remove(file_path)
        
        # Rebuild vector store
        faiss_store.rebuild_index(provider, api_key)
        
        logger.info(f"Deleted '{filename}' and updated vector store index.")
        return DocumentDeleteResponse(
            status="success",
            message=f"Successfully deleted '{filename}' and updated vector store index."
        )
    except Exception as e:
        logger.error(f"Failed to delete document and rebuild index: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete and rebuild index: {str(e)}")
