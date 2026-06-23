from typing import Optional
from fastapi import Header, HTTPException
from backend.app.core import config

def resolve_credentials(
    provider_header: Optional[str] = Header(None, alias="X-Provider"),
    openai_key_header: Optional[str] = Header(None, alias="X-OpenAI-Key"),
    mistral_key_header: Optional[str] = Header(None, alias="X-Mistral-Key"),
    payload_provider: Optional[str] = None
) -> tuple[str, str]:
    """
    Dependency to resolve the provider and associated API key.
    Prioritizes explicit payload providers, then headers, then default backend environment keys.
    """
    prov = payload_provider or provider_header or config.DEFAULT_PROVIDER
    if not prov:
        raise HTTPException(status_code=400, detail="LLM provider could not be resolved.")
        
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
