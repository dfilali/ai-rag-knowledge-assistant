import os
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from backend.app import config
from backend.app.services import vector_store

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
        print(f"Error rephrasing query: {e}")
        return query

def chat_with_docs(
    query: str,
    session_id: str,
    provider: str,
    api_key: str,
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes the full RAG process:
    1. Rephrases the query based on conversation history.
    2. Performs similarity search in FAISS.
    3. Construct RAG prompt context.
    4. Calls the LLM to generate the answer.
    5. Stores chat history.
    """
    # 1. Instantiate the LLM
    llm = get_chat_model(provider, api_key, model_name)
    history = get_session_history(session_id)

    # 2. Rephrase follow-up query to ensure good retrieval search
    search_query = rephrase_query(query, history, llm)

    # 3. Retrieve relevant chunks
    retrieved_docs = vector_store.similarity_search(search_query, provider, api_key, k=5)
    
    # 4. Construct context and source maps
    context_chunks = []
    sources = []
    
    for idx, doc in enumerate(retrieved_docs):
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

    # 5. Build system prompt instructions
    if context_str:
        system_instructions = (
            "You are a helpful AI assistant specialized in document analysis.\n"
            "Answer the user's question using ONLY the retrieved document chunks below. "
            "If the text does not contain enough information to answer the question, state clearly that "
            "you cannot find the answer in the provided documents. Do not make up facts or use external knowledge.\n\n"
            f"--- START RETRIEVED CONTEXT ---\n{context_str}\n--- END RETRIEVED CONTEXT ---"
        )
    else:
        system_instructions = (
            "You are a helpful AI assistant. Currently, no documents are uploaded or indexed. "
            "Please instruct the user to upload PDF files in the sidebar first to search their knowledge base. "
            "If they ask general questions, respond politely but remind them to upload files."
        )

    # 6. Setup full messages list including system prompt, chat history and current query
    messages = [SystemMessage(content=system_instructions)]
    
    # Append message history (convert simple dict history to LangChain message formats)
    for msg in history[-10:]:  # Keep last 10 messages for memory context
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
            
    # Add current question
    messages.append(HumanMessage(content=query))

    # 7. Generate RAG response
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
        raise RuntimeError(f"LLM execution error: {str(e)}")
