import os
import shutil
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_mistralai import MistralAIEmbeddings
from backend.app import config
from backend.app.services import doc_processor

def get_embeddings_model(provider: str, api_key: str):
    """
    Instantiates the appropriate embedding model based on provider and API key.
    """
    provider = provider.lower()
    if provider == "openai":
        if not api_key:
            raise ValueError("OpenAI API Key is required for embeddings.")
        return OpenAIEmbeddings(openai_api_key=api_key)
    elif provider == "mistral":
        if not api_key:
            raise ValueError("Mistral API Key is required for embeddings.")
        return MistralAIEmbeddings(mistral_api_key=api_key)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def get_vectorstore_path(provider: str) -> str:
    """
    Returns the vectorstore path for a specific provider to isolate indices.
    """
    path = os.path.join(config.VECTOR_STORE_DIR, provider)
    os.makedirs(path, exist_ok=True)
    return path

def save_vectorstore(vectorstore: FAISS, provider: str) -> None:
    """
    Saves the FAISS index to the local filesystem.
    """
    path = get_vectorstore_path(provider)
    vectorstore.save_local(path)

def load_vectorstore(provider: str, api_key: str) -> Optional[FAISS]:
    """
    Loads the FAISS index from the local filesystem.
    Returns None if no index exists.
    """
    path = get_vectorstore_path(provider)
    index_file = os.path.join(path, "index.faiss")
    if not os.path.exists(index_file):
        return None
        
    try:
        embeddings = get_embeddings_model(provider, api_key)
        return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return None

def add_documents_to_store(documents: List[Document], provider: str, api_key: str) -> None:
    """
    Adds chunked documents to the FAISS index. If an index already exists,
    it loads it, merges the new documents, and saves it back.
    """
    if not documents:
        return
        
    embeddings = get_embeddings_model(provider, api_key)
    vectorstore = load_vectorstore(provider, api_key)
    
    if vectorstore is None:
        vectorstore = FAISS.from_documents(documents, embeddings)
    else:
        vectorstore.add_documents(documents)
        
    save_vectorstore(vectorstore, provider)

def similarity_search(query: str, provider: str, api_key: str, k: int = 5) -> List[Document]:
    """
    Performs similarity search against the vector index.
    """
    vectorstore = load_vectorstore(provider, api_key)
    if vectorstore is None:
        return []
    return vectorstore.similarity_search(query, k=k)

def rebuild_index(provider: str, api_key: str) -> None:
    """
    Rebuilds the FAISS index from scratch using all files in the upload directory.
    If no files exist, clears the index directory.
    """
    path = get_vectorstore_path(provider)
    
    # Check if there are files in upload dir
    files = [f for f in os.listdir(config.UPLOAD_DIR) if f.lower().endswith(".pdf")]
    
    if not files:
        # Clear the index directory
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
        return

    # Ingest all files
    all_documents = []
    for filename in files:
        file_path = os.path.join(config.UPLOAD_DIR, filename)
        try:
            chunks = doc_processor.process_document(file_path, filename)
            all_documents.extend(chunks)
        except Exception as e:
            print(f"Error reprocessing document {filename}: {e}")

    if all_documents:
        embeddings = get_embeddings_model(provider, api_key)
        vectorstore = FAISS.from_documents(all_documents, embeddings)
        save_vectorstore(vectorstore, provider)
    else:
        # Clear index directory if no documents could be parsed
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
