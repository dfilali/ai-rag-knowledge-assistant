from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from backend.app.core.logging import setup_logger

logger = setup_logger("exceptions")

class RAGBaseException(Exception):
    """Base exception for all system exceptions."""
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

class CredentialError(RAGBaseException):
    """Raised when an API key is missing or invalid."""
    def __init__(self, message: str):
        super().__init__(message, status_code=401)

class DocumentProcessingError(RAGBaseException):
    """Raised when a document cannot be parsed or indexed."""
    def __init__(self, message: str):
        super().__init__(message, status_code=422)

class RetrievalError(RAGBaseException):
    """Raised when query retrieval or vector operations fail."""
    def __init__(self, message: str):
        super().__init__(message, status_code=500)

class AgentExecutionError(RAGBaseException):
    """Raised when agent reasoning loop or tool calls fail."""
    def __init__(self, message: str):
        super().__init__(message, status_code=500)

async def rag_exception_handler(request: Request, exc: RAGBaseException):
    """Global handler for custom RAG exceptions."""
    logger.error(f"RAG Error: {exc.message} on path {request.url.path}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message}
    )

async def general_exception_handler(request: Request, exc: Exception):
    """Global handler for general unhandled exceptions."""
    logger.error(f"Unhandled system error: {str(exc)} on path {request.url.path}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal Server Error: {str(exc)}"}
    )
