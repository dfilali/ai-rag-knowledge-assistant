import re
from typing import List
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from backend.app.services.vectorstores import faiss_store
from backend.app.core.logging import setup_logger

logger = setup_logger("retrievers.hybrid")

def tokenize(text: str) -> List[str]:
    """
    Splits text into lowercased alphanumeric tokens for BM25.
    """
    return re.findall(r"\w+", text.lower())

def get_doc_key(doc: Document) -> tuple:
    """
    Creates a unique hashable key for identifying identical documents in RRF.
    """
    meta = doc.metadata
    return (
        meta.get("source", ""),
        meta.get("page", 0),
        meta.get("chunk_idx", 0),
        doc.page_content
    )

def hybrid_retrieve(
    query: str,
    provider: str,
    api_key: str,
    k: int = 10
) -> List[Document]:
    """
    Performs hybrid retrieval:
    1. Dense semantic search from FAISS.
    2. Sparse keyword search using rank_bm25 BM25.
    3. Fuses ranks using Reciprocal Rank Fusion (RRF).
    """
    vectorstore = faiss_store.load_vectorstore(provider, api_key)
    if not vectorstore:
        logger.warning(f"No FAISS vectorstore found for provider {provider}. Returning empty list.")
        return []

    # 1. Fetch dense candidates (retrieve 2 * k to allow better overlap filtering)
    try:
        dense_candidates = vectorstore.similarity_search(query, k=k * 2)
    except Exception as e:
        logger.error(f"Error during dense vector retrieval: {e}")
        dense_candidates = []

    # Get all indexed documents for sparse search
    try:
        all_docs = list(vectorstore.docstore._dict.values())
    except Exception as e:
        logger.error(f"Failed to access docstore dictionary: {e}")
        all_docs = []

    if not all_docs:
        logger.warning("Docstore is empty. Falling back to dense candidates.")
        return dense_candidates[:k]

    # 2. Fetch sparse candidates
    try:
        corpus = [tokenize(doc.page_content) for doc in all_docs]
        bm25 = BM25Okapi(corpus)
        tokenized_query = tokenize(query)
        
        # Calculate BM25 scores
        scores = bm25.get_scores(tokenized_query)
        doc_scores = list(zip(all_docs, scores))
        # Filter documents with positive matching scores and sort
        doc_scores = [item for item in doc_scores if item[1] > 0.0]
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        sparse_candidates = [doc for doc, score in doc_scores][:k * 2]
    except Exception as e:
        logger.error(f"Error during sparse BM25 retrieval: {e}")
        sparse_candidates = []

    # 3. Reciprocal Rank Fusion (RRF)
    # RRF score = sum(1 / (60 + rank))
    rrf_scores = {}
    
    for rank, doc in enumerate(dense_candidates):
        key = get_doc_key(doc)
        if key not in rrf_scores:
            rrf_scores[key] = {"doc": doc, "score": 0.0}
        rrf_scores[key]["score"] += 1.0 / (60.0 + rank + 1)

    for rank, doc in enumerate(sparse_candidates):
        key = get_doc_key(doc)
        if key not in rrf_scores:
            rrf_scores[key] = {"doc": doc, "score": 0.0}
        rrf_scores[key]["score"] += 1.0 / (60.0 + rank + 1)

    # Sort candidates by combined score
    fused_results = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
    fused_documents = [item["doc"] for item in fused_results][:k]
    
    logger.info(
        f"Hybrid retrieval finished: dense count={len(dense_candidates)}, "
        f"sparse count={len(sparse_candidates)}, fused count={len(fused_documents)}"
    )
    return fused_documents
