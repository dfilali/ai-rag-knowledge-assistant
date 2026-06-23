import json
import re
from typing import List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from backend.app.services.rag_engine import get_chat_model

def evaluate_rag_response(
    question: str,
    contexts: List[str],
    answer: str,
    provider: str,
    api_key: str,
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluates the quality of a RAG response based on:
    1. Faithfulness (Is the answer derived purely from the retrieved context?)
    2. Answer Relevance (Does the answer address the question directy?)
    Returns scores out of 5 and descriptions.
    """
    if not contexts:
        return {
            "faithfulness_score": 0,
            "faithfulness_reason": "No contexts retrieved, cannot evaluate faithfulness.",
            "relevance_score": 0,
            "relevance_reason": "No contexts retrieved, cannot evaluate relevance."
        }

    # Format retrieved contexts
    context_str = "\n\n".join([f"Context {i+1}:\n{text}" for i, text in enumerate(contexts)])
    
    evaluation_system_prompt = (
        "You are an expert AI evaluator for RAG (Retrieval-Augmented Generation) systems. "
        "Your task is to grade the generated answer based on two metrics:\n\n"
        "1. **Faithfulness (Honnêteté)**: Grade whether the generated answer is grounded in and fully supported "
        "by the provided context. Look for hallucinations or external facts. Give a score from 1 to 5. "
        "1 means completely ungrounded/hallucinated. 5 means every fact in the answer is perfectly supported by the context.\n\n"
        "2. **Answer Relevance (Pertinence)**: Grade whether the generated answer directly addresses the question. "
        "Give a score from 1 to 5. 1 means the answer is completely off-topic. 5 means the answer directly, clearly, "
        "and completely addresses the user's question.\n\n"
        "You MUST respond ONLY in raw JSON format matching this schema:\n"
        "{\n"
        "  \"faithfulness_score\": int (1-5),\n"
        "  \"faithfulness_reason\": \"short explanation of the score\",\n"
        "  \"relevance_score\": int (1-5),\n"
        "  \"relevance_reason\": \"short explanation of the score\"\n"
        "}"
    )

    evaluation_user_prompt = (
        f"Question: {question}\n\n"
        f"--- CONTEXTS ---\n{context_str}\n\n"
        f"--- GENERATED ANSWER ---\n{answer}\n\n"
        f"JSON Evaluation Output:"
    )

    try:
        llm = get_chat_model(provider, api_key, model_name)
        messages = [
            SystemMessage(content=evaluation_system_prompt),
            HumanMessage(content=evaluation_user_prompt)
        ]
        
        response = llm.invoke(messages)
        content = response.content.strip()
        
        # Clean response in case LLM adds ```json Markdown wrappers
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            json_str = match.group(0)
            eval_data = json.loads(json_str)
            return eval_data
        else:
            raise ValueError(f"Could not find JSON in LLM response: {content}")
            
    except Exception as e:
        print(f"Evaluation error: {e}")
        return {
            "faithfulness_score": -1,
            "faithfulness_reason": f"Evaluation failed: {str(e)}",
            "relevance_score": -1,
            "relevance_reason": f"Evaluation failed: {str(e)}"
        }
