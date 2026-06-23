import os
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Header, HTTPException, Body
from pydantic import BaseModel, Field
from backend.app import config
from backend.app.services import doc_processor, vector_store, rag_engine, evaluator

router = APIRouter()

# Request Pydantic Schemas
class ChatRequest(BaseModel):
    query: str = Field(..., description="User question")
    session_id: str = Field(..., description="Unique conversation session ID")
    provider: Optional[str] = Field(None, description="LLM provider: 'openai' or 'mistral'")
    model: Optional[str] = Field(None, description="Specific model name")

class EvaluateRequest(BaseModel):
    query: str
    contexts: List[str]
    answer: str
    provider: Optional[str] = None
    model: Optional[str] = None

class ClearHistoryRequest(BaseModel):
    session_id: str

# Helper to extract provider and relevant API key
def resolve_credentials(
    provider_header: Optional[str],
    openai_key_header: Optional[str],
    mistral_key_header: Optional[str],
    payload_provider: Optional[str] = None
):
    # Determine provider (payload takes precedence over header, then default config)
    prov = payload_provider or provider_header or config.DEFAULT_PROVIDER
    prov = prov.lower()
    
    if prov not in ["openai", "mistral"]:
        raise HTTPException(status_code=400, detail=f"Unsupported provider: {prov}")
        
    # Resolve key
    key = ""
    if prov == "openai":
        key = config.get_api_key("openai", openai_key_header)
    elif prov == "mistral":
        key = config.get_api_key("mistral", mistral_key_header)
        
    if not key:
        raise HTTPException(
            status_code=401, 
            detail=f"API Key for '{prov}' is missing. Please set it in UI Settings or backend .env."
        )
        
    return prov, key

@router.get("/status")
def get_status():
    """
    Checks if API keys are set in the environment.
    """
    return {
        "env_openai_configured": bool(config.OPENAI_API_KEY),
        "env_mistral_configured": bool(config.MISTRAL_API_KEY),
        "default_provider": config.DEFAULT_PROVIDER
    }

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    x_provider: Optional[str] = Header(None, alias="X-Provider"),
    x_openai_key: Optional[str] = Header(None, alias="X-OpenAI-Key"),
    x_mistral_key: Optional[str] = Header(None, alias="X-Mistral-Key")
):
    """
    Uploads a PDF, processes text, chunks it, and adds it to the vector store.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    # Check credentials before doing work
    prov, key = resolve_credentials(x_provider, x_openai_key, x_mistral_key)
    
    # Save the file locally
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
        chunks = doc_processor.process_document(file_path, filename)
        
        # Index document in FAISS
        vector_store.add_documents_to_store(chunks, prov, key)
        
        return {
            "filename": filename,
            "chunks_count": len(chunks),
            "status": "success",
            "message": f"Successfully indexed '{filename}' ({len(chunks)} chunks)."
        }
    except Exception as e:
        # Cleanup uploaded file if index fails
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

@router.post("/chat")
def chat_endpoint(
    req: ChatRequest,
    x_provider: Optional[str] = Header(None, alias="X-Provider"),
    x_openai_key: Optional[str] = Header(None, alias="X-OpenAI-Key"),
    x_mistral_key: Optional[str] = Header(None, alias="X-Mistral-Key")
):
    """
    Executes a RAG chat request based on uploaded files.
    """
    # Resolve credentials
    prov, key = resolve_credentials(x_provider, x_openai_key, x_mistral_key, req.provider)
    
    try:
        result = rag_engine.chat_with_docs(
            query=req.query,
            session_id=req.session_id,
            provider=prov,
            api_key=key,
            model_name=req.model
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/documents")
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
        documents.append({
            "filename": filename,
            "size_bytes": stats.st_size,
            "created_at": stats.st_mtime
        })
        
    # Sort by creation date descending
    documents.sort(key=lambda x: x["created_at"], reverse=True)
    return documents

@router.delete("/documents/{filename}")
def delete_document(
    filename: str,
    x_provider: Optional[str] = Header(None, alias="X-Provider"),
    x_openai_key: Optional[str] = Header(None, alias="X-OpenAI-Key"),
    x_mistral_key: Optional[str] = Header(None, alias="X-Mistral-Key")
):
    """
    Deletes an uploaded PDF and rebuilds the vector index.
    """
    file_path = os.path.join(config.UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")
        
    # Resolve credentials
    prov, key = resolve_credentials(x_provider, x_openai_key, x_mistral_key)
    
    try:
        # Delete file
        os.remove(file_path)
        
        # Rebuild vector store
        vector_store.rebuild_index(prov, key)
        
        return {
            "status": "success",
            "message": f"Successfully deleted '{filename}' and updated vector store index."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete and rebuild index: {str(e)}")

@router.post("/evaluate")
def evaluate_endpoint(
    req: EvaluateRequest,
    x_provider: Optional[str] = Header(None, alias="X-Provider"),
    x_openai_key: Optional[str] = Header(None, alias="X-OpenAI-Key"),
    x_mistral_key: Optional[str] = Header(None, alias="X-Mistral-Key")
):
    """
    Evaluates a generated RAG response.
    """
    # Resolve credentials
    prov, key = resolve_credentials(x_provider, x_openai_key, x_mistral_key, req.provider)
    
    try:
        evaluation = evaluator.evaluate_rag_response(
            question=req.query,
            contexts=req.contexts,
            answer=req.answer,
            provider=prov,
            api_key=key,
            model_name=req.model
        )
        return evaluation
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clear-history")
def clear_history(req: ClearHistoryRequest):
    """
    Resets the conversation history for a given session.
    """
    rag_engine.clear_session_history(req.session_id)
    return {"status": "success", "message": f"Session history for '{req.session_id}' cleared."}
