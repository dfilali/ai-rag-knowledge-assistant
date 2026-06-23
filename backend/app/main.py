import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from backend.app.api.router import router
from backend.app.core.exceptions import RAGBaseException, rag_exception_handler, general_exception_handler

app = FastAPI(
    title="RAG AI Knowledge Assistant",
    description="A modern, high-performance FAANG-ready RAG agent chatbot for document analysis with built-in evaluation.",
    version="1.0.0"
)

# Register Global Exception Handlers
app.add_exception_handler(RAGBaseException, rag_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

# Configure CORS for local development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register RAG REST API router
app.include_router(router, prefix="/api")

# Resolve static directory path relative to project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
static_dir = os.path.join(ROOT_DIR, "static")

# Mount static files folder (style.css, app.js, images, etc.)
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def serve_spa():
    """
    Serves the main single page application layout at the root path.
    """
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "status": "online",
        "message": "FastAPI server running. Frontend files (static/index.html) were not found.",
        "api_docs": "/docs"
    }

# Ensure init files exist to register backend python packages properly
@app.on_event("startup")
def startup_event():
    # Make sure packages are recognized correctly
    pass

