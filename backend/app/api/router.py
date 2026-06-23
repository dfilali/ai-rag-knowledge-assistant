from fastapi import APIRouter
from backend.app.api.routes import chat, documents, evaluate, status

router = APIRouter()

# Include all sub-routers directly to maintain standard endpoint paths
router.include_router(status.router)
router.include_router(documents.router)
router.include_router(chat.router)
router.include_router(evaluate.router)
