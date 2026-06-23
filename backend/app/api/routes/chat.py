from fastapi import APIRouter, Depends, HTTPException
from backend.app.api.schemas.chat import ChatRequest, ChatResponse, ClearHistoryRequest
from backend.app.api.routes.dependencies import resolve_credentials
from backend.app.services.agents import agent
from backend.app.core.logging import setup_logger

logger = setup_logger("routes.chat")
router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(
    req: ChatRequest,
    creds: tuple[str, str] = Depends(resolve_credentials)
):
    """
    Executes an Agentic RAG chat request.
    """
    provider, api_key = creds
    try:
        result = agent.chat_with_docs(
            query=req.query,
            session_id=req.session_id,
            provider=provider,
            api_key=api_key,
            model_name=req.model
        )
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            search_query=result["search_query"]
        )
    except Exception as e:
        logger.error(f"Chat endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clear-history")
def clear_history(req: ClearHistoryRequest):
    """
    Resets the conversation history for a given session.
    """
    agent.clear_session_history(req.session_id)
    logger.info(f"Conversation history for session '{req.session_id}' has been cleared.")
    return {"status": "success", "message": f"Session history for '{req.session_id}' cleared."}
