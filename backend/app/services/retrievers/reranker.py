import json
import re
import urllib.request
import urllib.error
from typing import List, Optional
from langchain_core.documents import Document
from backend.app.core import config
from backend.app.core.logging import setup_logger
from backend.app.services.llm import get_chat_model

logger = setup_logger("retrievers.reranker")

def rerank_documents(
    query: str,
    documents: List[Document],
    provider: str,
    api_key: str,
    model_name: Optional[str] = None,
    top_k: int = 5
) -> List[Document]:
    """
    Reranks document candidates using either Mistral's native Rerank API 
    or a fallback LLM-based prompt reranker.
    """
    if not documents:
        return []
        
    provider = provider.lower()
    
    if provider == "mistral":
        try:
            logger.info("Attempting native Mistral Rerank API call...")
            return _rerank_mistral_api(query, documents, api_key, top_k)
        except Exception as e:
            logger.warning(f"Mistral Rerank API failed, falling back to LLM reranker: {e}")
            
    # Default fallback: LLM-based prompt reranker (OpenAI or Mistral)
    try:
        logger.info(f"Using LLM-based prompt reranking with {provider}...")
        return _rerank_llm_prompt(query, documents, provider, api_key, model_name, top_k)
    except Exception as e:
        logger.error(f"LLM-based reranking failed. Returning original documents: {e}")
        return documents[:top_k]

def _rerank_mistral_api(
    query: str,
    documents: List[Document],
    api_key: str,
    top_k: int
) -> List[Document]:
    """
    Calls the official Mistral Rerank API endpoint.
    """
    url = "https://api.mistral.ai/v1/rerank"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Format documents for Mistral payload
    docs_payload = [{"text": doc.page_content} for doc in documents]
    
    payload = {
        "model": "mistral-rerank-latest",
        "query": query,
        "documents": docs_payload,
        "top_n": top_k
    }
    
    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=req_data, headers=headers, method="POST")
    
    with urllib.request.urlopen(req, timeout=10) as response:
        res_data = json.loads(response.read().decode("utf-8"))
        
    results = res_data.get("results", [])
    reranked_docs = []
    
    for item in results:
        idx = item.get("index")
        if idx is not None and idx < len(documents):
            reranked_docs.append(documents[idx])
            
    return reranked_docs[:top_k]

def _rerank_llm_prompt(
    query: str,
    documents: List[Document],
    provider: str,
    api_key: str,
    model_name: Optional[str],
    top_k: int
) -> List[Document]:
    """
    Uses a standard chat model to grade documents and returns the top k.
    """
    llm = get_chat_model(provider, api_key, model_name)
    
    # Construct prompt showing all documents and asking for relevance scores
    docs_text = ""
    for idx, doc in enumerate(documents):
        docs_text += f"--- DOCUMENT {idx} ---\n{doc.page_content}\n\n"
        
    system_prompt = (
        "You are an expert AI search reranker. Your task is to rank the documents based on their relevance "
        "to the user query. Evaluate how helpful each document is to answer the query.\n"
        "Assign a relevance score from 0 (completely irrelevant) to 10 (perfect match) for each document.\n"
        "Output ONLY a raw JSON dictionary mapping document indices to scores, like this:\n"
        '{\n  "0": 8.5,\n  "1": 4.0\n}\n'
        "Do not include explanation, markdown formatting, or thoughts. Only the JSON dictionary."
    )
    
    user_prompt = f"Query: {query}\n\n{docs_text}JSON Output:"
    
    from langchain_core.messages import SystemMessage, HumanMessage
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    content = response.content.strip()
    
    # Clean JSON wrappers if LLM returned them
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        raise ValueError(f"Could not parse JSON from LLM response: {content}")
        
    scores_dict = json.loads(match.group(0))
    
    # Parse scores and match to documents
    doc_scores = []
    for str_idx, score in scores_dict.items():
        try:
            idx = int(str_idx)
            if 0 <= idx < len(documents):
                doc_scores.append((documents[idx], float(score)))
        except ValueError:
            continue
            
    # Sort by score descending
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    
    # If LLM missed some documents, append them to the end with score 0
    scored_docs = [doc for doc, _ in doc_scores]
    for doc in documents:
        if doc not in scored_docs:
            doc_scores.append((doc, 0.0))
            
    # Return top_k
    return [doc for doc, _ in doc_scores][:top_k]
