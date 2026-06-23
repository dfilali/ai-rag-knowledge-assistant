from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    query: str = Field(..., description="User question or query.")
    session_id: str = Field(..., description="Unique conversation session ID.")
    provider: Optional[str] = Field(None, description="LLM provider: 'openai' or 'mistral'.")
    model: Optional[str] = Field(None, description="Specific model name to override default config.")

class SourceInfo(BaseModel):
    id: str = Field(..., description="Unique ID for the citation context chunk.")
    source: str = Field(..., description="Original file name source.")
    page: int = Field(..., description="Page number of the original document.")
    content: str = Field(..., description="Text content matching the citation.")

class ChatResponse(BaseModel):
    answer: str = Field(..., description="Generated answer from the Agentic RAG.")
    sources: List[SourceInfo] = Field(..., description="Citations used to formulate the answer.")
    search_query: str = Field(..., description="Final optimized query used to retrieve documents.")

class ClearHistoryRequest(BaseModel):
    session_id: str = Field(..., description="Conversation session ID to wipe out memory.")
