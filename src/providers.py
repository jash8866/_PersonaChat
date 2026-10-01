import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_MODELS_DIR = os.getenv("HF_MODELS_DIR", str((__import__("pathlib").Path(__file__).resolve().parent.parent / "models")))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", _MODELS_DIR)
os.environ.setdefault("HF_HUB_CACHE", _MODELS_DIR)
os.environ.setdefault("HF_HOME", _MODELS_DIR)
MODEL_ID = "openrouter:openai/gpt-oss-120b"
MODEL_NAME = "openai/gpt-oss-120b"


_EMB_CACHE = None

def get_embeddings():
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
    from langchain_openrouter import ChatOpenRouter
    return ChatOpenRouter(
        model_name=MODEL_NAME,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=temperature,
        timeout=LLM_TIMEOUT * 1000,
        max_retries=1,
        max_tokens=LLM_MAX_TOKENS,
    )
