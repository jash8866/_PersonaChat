"""
providers.py - Single abstraction for embeddings + LLM.
Embeddings: always HuggingFace (local, free, matches _learn_RAG pattern).
LLM: fixed single model — OpenRouter gpt-oss-120b (no switching).
"""
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# Pin HF cache to project-local ./models so weights live in repo (gitignored),
# survive restarts, and can run offline. Must be set before HF imports.
_MODELS_DIR = os.getenv("HF_MODELS_DIR", str((__import__("pathlib").Path(__file__).resolve().parent.parent / "models")))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", _MODELS_DIR)
os.environ.setdefault("HF_HUB_CACHE", _MODELS_DIR)
os.environ.setdefault("HF_HOME", _MODELS_DIR)
# Single fixed model — no switching.
MODEL_ID = "openrouter:openai/gpt-oss-120b"
MODEL_NAME = "openai/gpt-oss-120b"


_EMB_CACHE = None

def get_embeddings():
    """HuggingFace local embeddings - singleton (avoids reloading 103 weights per chain)."""
    global _EMB_CACHE
    if _EMB_CACHE is None:
        from src.telemetry import get_logger, timer
        log = get_logger("personachat.providers")
        from langchain_huggingface import HuggingFaceEmbeddings
        with timer(log, "embeddings_load", model=EMBED_MODEL):
            _EMB_CACHE = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return _EMB_CACHE


LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "300"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1500"))

def get_llm(temperature: float = 0.7):
    """Fixed single model: OpenRouter gpt-oss-120b. LLM_TIMEOUT is seconds
    (reasoning models need minutes); ChatOpenRouter takes milliseconds.
    """
    from langchain_openrouter import ChatOpenRouter
    return ChatOpenRouter(
        model_name=MODEL_NAME,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=temperature,
        # NOTE: ChatOpenRouter.timeout is MILLISECONDS (maps to SDK timeout_ms),
        # so convert from LLM_TIMEOUT seconds.
        timeout=LLM_TIMEOUT * 1000,
        max_retries=1,
        max_tokens=LLM_MAX_TOKENS,
    )
