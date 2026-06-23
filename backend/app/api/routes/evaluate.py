from typing import Optional
from fastapi import APIRouter, Header, HTTPException, Depends
from backend.app.api.schemas.evaluate import EvaluateRequest, EvaluateResponse
from backend.app.api.routes.dependencies import resolve_credentials
from backend.app.services.evaluation import evaluator

router = APIRouter()

@router.post("/evaluate", response_model=EvaluateResponse)
def evaluate_endpoint(
    req: EvaluateRequest,
    creds: tuple[str, str] = Depends(resolve_credentials)
):
    """
    Evaluates a generated RAG response using the evaluation service.
    """
    provider, api_key = creds
    try:
        evaluation = evaluator.evaluate_rag_response(
            question=req.query,
            contexts=req.contexts,
            answer=req.answer,
            provider=provider,
            api_key=api_key,
            model_name=req.model
        )
        return evaluation
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")
