"""
retriever.py - Per-persona filtered retriever (modular, independent).
"""
import os
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
from langchain_chroma import Chroma
from src.providers import get_embeddings

CHROMA_DIR = Path(os.getenv("CHROMA_DIR", "./chroma_db"))

def get_retriever(persona_id: str, k: int = 4):
    embeddings = get_embeddings()
    store = Chroma(
        collection_name=f"persona_{persona_id}",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    return store.as_retriever(search_kwargs={"k": k})

def retrieve(persona_id: str, query: str, k: int = 4) -> str:
    """Simple helper for non-LCEL usage."""
    r = get_retriever(persona_id, k=k)
    docs = r.invoke(query)
    return "\n\n---\n\n".join(d.page_content for d in docs)


def search_with_scores(persona_id: str, query: str, k: int = 4):
    """Scored retrieval for the Evidence panel.

    Returns (results, elapsed_ms) where results is a list of
    (Document, distance) tuples, best match first. Lower distance = more relevant.
    """
    import time
    embeddings = get_embeddings()
    store = Chroma(
        collection_name=f"persona_{persona_id}",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    start = time.perf_counter()
    results = store.similarity_search_with_score(query, k=k)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return results, elapsed_ms
