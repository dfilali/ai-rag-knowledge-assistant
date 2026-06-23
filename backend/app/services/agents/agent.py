import os
import json
import re
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document

from backend.app.core import config
from backend.app.core.logging import setup_logger
from backend.app.services.retrievers import hybrid, reranker

logger = setup_logger("agents.agent")

# Global in-memory chat session history
# Format: { session_id: [ {"role": "user"|"assistant", "content": "..."} ] }
sessions_db: Dict[str, List[Dict[str, str]]] = {}

def get_session_history(session_id: str) -> List[Dict[str, str]]:
    """
    Retrieves the chat history for a session. Initializes if it doesn't exist.
    """
    if session_id not in sessions_db:
        sessions_db[session_id] = []
    return sessions_db[session_id]

def clear_session_history(session_id: str) -> None:
    """
    Clears the history for a given session.
    """
    if session_id in sessions_db:
        sessions_db[session_id] = []

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

def rephrase_query(query: str, history: List[Dict[str, str]], llm) -> str:
    """
    Rephrases a follow-up query to make it standalone based on the conversation history.
    """
    if not history:
        return query

    # Construct history text representation
    history_str = ""
    for msg in history[-5:]:  # Limit history context to last 5 messages to avoid overflow
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    rephrase_system_prompt = (
        "Given the following conversation history and a follow-up question, "
        "rephrase the follow-up question to be a standalone question that can be "
        "understood without the history. Do NOT answer the question. Just output "
        "the rephrased standalone question and nothing else."
    )

    rephrase_user_prompt = f"History:\n{history_str}\nFollow-up Question: {query}\nStandalone Question:"

    messages = [
        SystemMessage(content=rephrase_system_prompt),
        HumanMessage(content=rephrase_user_prompt)
    ]

    try:
        response = llm.invoke(messages)
        standalone_query = response.content.strip()
        # Return rephrased query or fallback to original if LLM failed
        return standalone_query if standalone_query else query
    except Exception as e:
        logger.error(f"Error rephrasing query: {e}")
        return query

def evaluate_evidence_sufficiency(query: str, docs: List[Document], llm) -> tuple[bool, str]:
    """
    Asks the LLM if the retrieved documents contain sufficient information to answer the query.
    Returns (is_sufficient, reason).
    """
    if not docs:
        return False, "No documents were retrieved."

    contexts = ""
    for idx, doc in enumerate(docs):
        contexts += f"--- Document {idx+1} ---\n{doc.page_content}\n\n"

    system_prompt = (
        "You are an evidence grading agent. Evaluate whether the provided documents contain enough "
        "relevant details to answer the user's query.\n"
        "Be strict: if the documents do not have facts to answer the question, state that it is insufficient.\n"
        "You MUST respond ONLY in raw JSON matching this schema:\n"
        "{\n"
        '  "sufficient": true/false,\n'
        '  "reason": "brief explanation of what is present or missing"\n'
        "}"
    )

    user_prompt = f"Query: {query}\n\n{contexts}JSON Grading Output:"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    try:
        response = llm.invoke(messages)
        content = response.content.strip()
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            return bool(data.get("sufficient", False)), str(data.get("reason", ""))
        return False, "Could not parse JSON grader output."
    except Exception as e:
        logger.error(f"Sufficiency check error: {e}")
        return True, "Sufficiency evaluation skipped due to exception." # Fallback to true to avoid locking RAG

def generate_alternative_query(query: str, previous_search: str, reason: str, llm) -> str:
    """
    Generates a reformulated query to broaden or redirect search.
    """
    system_prompt = (
        "You are a search query reformulation agent. Your goal is to improve document search.\n"
        "The previous query did not return sufficient details.\n"
        "Generate a single new search query to retrieve the missing information. Use synonyms or different phrasing.\n"
        "Output ONLY the new search query string, nothing else."
    )
    user_prompt = (
        f"Original User Question: {query}\n"
        f"Previous Search Query: {previous_search}\n"
        f"Why it failed: {reason}\n"
        f"New Search Query:"
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    try:
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        logger.error(f"Error generating alternative query: {e}")
        return query

def chat_with_docs(
    query: str,
    session_id: str,
    provider: str,
    api_key: str,
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes the Agentic RAG logic:
    1. Standalone query generation.
    2. Hybrid retrieval (Dense + Sparse).
    3. Reranking.
    4. Evidence sufficiency check (Self-Correction loop).
    5. Alternative query expansion (if needed).
    6. Formulate final response citing sources.
    """
    llm = get_chat_model(provider, api_key, model_name)
    history = get_session_history(session_id)

    # Resolve if document base is completely empty
    upload_dir = config.UPLOAD_DIR
    has_files = False
    if os.path.exists(upload_dir):
        files = [f for f in os.listdir(upload_dir) if f.lower().endswith(".pdf")]
        if files:
            has_files = True

    if not has_files:
        answer = (
            "Bonjour ! Je suis votre assistant de documents. Actuellement, aucun document PDF "
            "n'est indexé. Veuillez charger des fichiers PDF dans la barre latérale pour me donner "
            "une base de connaissances."
        )
        return {
            "answer": answer,
            "sources": [],
            "search_query": query
        }

    # 1. Rephrase follow-up query based on history
    search_query = rephrase_query(query, history, llm)
    logger.info(f"Original query: {query} -> Search query: {search_query}")

    # 2. Hybrid Retrieval + Reranking (Iteration 1)
    docs = hybrid.hybrid_retrieve(search_query, provider, api_key, k=config.RETRIEVAL_K)
    reranked_docs = reranker.rerank_documents(search_query, docs, provider, api_key, model_name, top_k=config.RERANK_TOP_K)

    # 3. Grade sufficiency (Self-Correction)
    sufficient, reason = evaluate_evidence_sufficiency(query, reranked_docs, llm)
    logger.info(f"Evidence sufficiency check 1: {sufficient} (Reason: {reason})")

    # 4. Agent self-correction loop: try query reformulation if details are lacking
    if not sufficient:
        alt_query = generate_alternative_query(query, search_query, reason, llm)
        logger.info(f"Reformulating search to alternative query: {alt_query}")
        
        # Second Retrieval
        alt_docs = hybrid.hybrid_retrieve(alt_query, provider, api_key, k=config.RETRIEVAL_K)
        alt_reranked = reranker.rerank_documents(alt_query, alt_docs, provider, api_key, model_name, top_k=config.RERANK_TOP_K)
        
        # Merge candidate documents ensuring uniqueness
        seen_keys = set()
        combined_docs = []
        
        for d in reranked_docs + alt_reranked:
            key = hybrid.get_doc_key(d)
            if key not in seen_keys:
                seen_keys.add(key)
                combined_docs.append(d)
                
        # Final rerank to keep the best top_k chunks
        reranked_docs = reranker.rerank_documents(query, combined_docs, provider, api_key, model_name, top_k=config.RERANK_TOP_K)
        
        # Re-evaluate sufficiency
        sufficient, final_reason = evaluate_evidence_sufficiency(query, reranked_docs, llm)
        logger.info(f"Evidence sufficiency check 2: {sufficient} (Reason: {final_reason})")
        search_query = f"{search_query} | {alt_query}"

    # 5. Format sources
    context_chunks = []
    sources = []
    
    for idx, doc in enumerate(reranked_docs):
        meta = doc.metadata
        filename = meta.get("source", "Unknown Document")
        page = meta.get("page", 1)
        chunk_idx = meta.get("chunk_idx", 0)
        
        context_chunks.append(
            f"[Document {idx+1}: {filename} (Page {page})]\n{doc.page_content}"
        )
        
        sources.append({
            "id": f"{filename}_p{page}_c{chunk_idx}",
            "source": filename,
            "page": page,
            "content": doc.page_content
        })

    context_str = "\n\n".join(context_chunks)

    # 6. Build system prompt instructions based on evaluation sufficiency
    if sufficient:
        system_instructions = (
            "You are a helpful AI assistant specialized in document analysis.\n"
            "Answer the user's question using ONLY the retrieved document chunks below. "
            "State your facts clearly and cite documents where applicable.\n\n"
            f"--- START RETRIEVED CONTEXT ---\n{context_str}\n--- END RETRIEVED CONTEXT ---"
        )
    else:
        system_instructions = (
            "You are a helpful AI assistant specialized in document analysis.\n"
            "You searched the knowledge base, but did NOT find enough information in the provided "
            "documents to fully answer the query. Respond by explaining what information was missing "
            "in the documents, then proceed to answer the query based on your general knowledge. "
            "WARNING: You MUST explicitly state that your answer is based on general knowledge and is "
            "NOT validated by the uploaded documents.\n\n"
            f"--- START RETRIEVED CONTEXT (INSUFFICIENT) ---\n{context_str}\n--- END RETRIEVED CONTEXT ---"
        )

    # Setup messages history
    messages = [SystemMessage(content=system_instructions)]
    
    for msg in history[-10:]:  # Keep last 10 messages for memory context
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
            
    # Add current question
    messages.append(HumanMessage(content=query))

    try:
        response = llm.invoke(messages)
        answer = response.content
        
        # Save exchange to chat history
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})
        
        return {
            "answer": answer,
            "sources": sources,
            "search_query": search_query
        }
    except Exception as e:
        logger.error(f"LLM execution error: {e}")
        raise RuntimeError(f"LLM execution error: {str(e)}")
