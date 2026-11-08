import os
import time
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


def space_id():
    return os.environ.get("HF_SPACE_URL", "").strip()


def space_page_url():
    s = space_id()
    return s if not s or s.startswith("http") else f"https://huggingface.co/spaces/{s}"


@lru_cache(maxsize=1)
def _client(sid):
    from gradio_client import Client
    return Client(sid, verbose=False)


def predict(text):
    """Call the DistilBERT Hugging Face Space; returns trainer.predict-shaped dict or None."""
    sid = space_id()
    if not sid:
        return None
    t0 = time.perf_counter()
    try:
        out = _client(sid).predict(text, api_name="/classify")
    except Exception as e:
        logger.warning("HF Space call failed: %s", e)
        _client.cache_clear()
        return None
    raw = out[-1] if isinstance(out, (list, tuple)) else out
    if not isinstance(raw, dict) or not raw.get("label"):
        return None
    return {**raw, "latency_ms": round((time.perf_counter() - t0) * 1000), "source": "hf_space"}
