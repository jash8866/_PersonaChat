import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE / "models"
MODELS_DIR.mkdir(exist_ok=True)

os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(MODELS_DIR)
os.environ["HF_HUB_CACHE"] = str(MODELS_DIR)
os.environ["HF_HOME"] = str(MODELS_DIR)

try:
    from dotenv import load_dotenv
    load_dotenv(BASE / ".env")
except ImportError:
    pass

MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
print(f"Downloading {MODEL} -> {MODELS_DIR}")

from sentence_transformers import SentenceTransformer
m = SentenceTransformer(MODEL, cache_folder=str(MODELS_DIR))
print(f"OK: {MODEL} cached. Test embed dim={len(m.encode('hello'))}")
print("Next: set HF_HUB_OFFLINE=1 in .env to force offline loads.")
