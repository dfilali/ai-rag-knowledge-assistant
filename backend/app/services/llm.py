from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from backend.app.core import config

def get_chat_model(provider: str, api_key: str, model_name: Optional[str] = None):
    """
    Instantiates the Chat LLM based on provider and API key.
    """
    provider = provider.lower()
    if provider == "openai":
        if not api_key:
            raise ValueError("OpenAI API Key is required for the chat model.")
        model = model_name or config.DEFAULT_OPENAI_MODEL
        return ChatOpenAI(openai_api_key=api_key, model=model, temperature=0.2)
    elif provider == "mistral":
        if not api_key:
            raise ValueError("Mistral API Key is required for the chat model.")
        model = model_name or config.DEFAULT_MISTRAL_MODEL
        return ChatMistralAI(mistral_api_key=api_key, model=model, temperature=0.2)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
