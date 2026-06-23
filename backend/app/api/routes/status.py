from fastapi import APIRouter
from backend.app.core import config

router = APIRouter()

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
