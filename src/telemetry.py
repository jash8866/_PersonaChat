"""
telemetry.py - Single logging abstraction (modular, independent).
Logs to console + logs/personachat.log. Use get_logger(__name__) anywhere.
"""
import logging
import time
from contextlib import contextmanager
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "personachat.log"

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_configured = False

def get_logger(name: str = "personachat") -> logging.Logger:
    global _configured
    logger = logging.getLogger(name)
    if not _configured:
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
        fh.setFormatter(logging.Formatter(_FORMAT))
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(fh)
        logger.addHandler(ch)
        _configured = True
    return logger

@contextmanager
def timer(logger: logging.Logger, label: str, **fields):
    """Usage: with timer(log, 'llm', persona=..., model=...): ..."""
    start = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - start
        extra = " ".join(f"{k}={v}" for k, v in fields.items())
        logger.info(f"{label} took {dt:.2f}s {extra}".strip())
