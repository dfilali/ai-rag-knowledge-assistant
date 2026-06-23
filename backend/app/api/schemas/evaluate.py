from typing import List, Optional
from pydantic import BaseModel, Field

class EvaluateRequest(BaseModel):
    query: str = Field(..., description="User query question.")
    contexts: List[str] = Field(..., description="Retrieved raw context chunks.")
    answer: str = Field(..., description="Generated answer to evaluate.")
    provider: Optional[str] = Field(None, description="LLM provider: 'openai' or 'mistral'.")
    model: Optional[str] = Field(None, description="Model to use for evaluation.")

class EvaluateResponse(BaseModel):
    faithfulness_score: int = Field(..., ge=-1, le=5, description="Faithfulness score (1-5), or -1 if failed.")
    faithfulness_reason: str = Field(..., description="Explaining why the faithfulness score was assigned.")
    relevance_score: int = Field(..., ge=-1, le=5, description="Relevance score (1-5), or -1 if failed.")
    relevance_reason: str = Field(..., description="Explaining why the relevance score was assigned.")
