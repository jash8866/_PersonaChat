"""
providers.py - Single abstraction for embeddings + LLM.
Embeddings: always HuggingFace (local, free, matches _learn_RAG pattern).
LLM: switchable via LLM_PROVIDER=ollama|openrouter (modular, independent).
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
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# Support .env with OPENROUTER_MODEL1/2/3 plus legacy OPENROUTER_MODEL
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-20b:free")


def _openrouter_models_from_env():
    """Collect OPENROUTER_MODEL1/2/3 (+ legacy OPENROUTER_MODEL) in order, deduped."""
    models = []
    for key in ["OPENROUTER_MODEL1", "OPENROUTER_MODEL2", "OPENROUTER_MODEL3", "OPENROUTER_MODEL"]:
        v = os.getenv(key)
        if v and v not in models:
            models.append(v)
    return models


def list_models():
    """All selectable models for UI DDL. Independent of LLM code - pure config.

    Returns: [{"id": "ollama:qwen2.5:3b", "label": "qwen2.5:3b (ollama local)", ...}, ...]
    """
    options = [
        {
            "id": f"ollama:{OLLAMA_MODEL}",
            "label": f"{OLLAMA_MODEL} (ollama local)",
            "provider": "ollama",
            "model": OLLAMA_MODEL,
        }
    ]
    for m in _openrouter_models_from_env():
        options.append({
            "id": f"openrouter:{m}",
            "label": f"{m} (openrouter)",
            "provider": "openrouter",
            "model": m,
        })
    return options


def get_default_model_id():
    """Default from LLM_PROVIDER env, falls back to first listed option."""
    if LLM_PROVIDER == "openrouter":
        ms = _openrouter_models_from_env()
        if ms:
            return f"openrouter:{ms[0]}"
    return f"ollama:{OLLAMA_MODEL}"


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


LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "90"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "800"))

def get_llm(temperature: float = 0.7, model_id: str | None = None):
    """Factory: runtime-selectable. model_id like 'ollama:qwen2.5:3b' or 'openrouter:...'.
    Falls back to env default. Fail-fast timeouts so free-tier hangs surface in ~90s, not 347s.
    """
    mid = model_id or get_default_model_id()
    provider, _, model = mid.partition(":")
    if provider == "openrouter":
        from langchain_openrouter import ChatOpenRouter
        return ChatOpenRouter(
            model_name=model,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            temperature=temperature,
            timeout=LLM_TIMEOUT,
            max_retries=1,
            max_tokens=LLM_MAX_TOKENS,
        )
    # default: ollama (local, usually seconds)
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=model or OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
        timeout=LLM_TIMEOUT,
        num_predict=LLM_MAX_TOKENS,
    )
