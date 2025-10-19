# Path: utils/embed_cache.py
"""
Simple disk-based embedding cache.
Keyed by SHA256(text). Stores embeddings as lists of floats in emb_cache.json.
Atomic writes and basic in-memory caching for speed.

Usage:
    from utils.embed_cache import get_cached_embedding
    emb = get_cached_embedding(text, lambda s: model.encode(s))
"""
import hashlib
import json
import os
import threading
from typing import Callable, List

CACHE_FILE = "emb_cache.json"
_lock = threading.Lock()
_cache = {}

# load existing cache on import (if present)
if os.path.exists(CACHE_FILE):
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            _cache = json.load(f)
    except Exception:
        _cache = {}

def _save_cache():
    """Write cache atomically."""
    tmp = CACHE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_cache, f)
    os.replace(tmp, CACHE_FILE)

def _text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def get_cached_embedding(text: str, compute_fn: Callable[[str], List[float]]) -> List[float]:
    """
    Return cached embedding if present, otherwise compute with compute_fn(text),
    store in cache and return. compute_fn should return a list-like object.
    """
    key = _text_key(text)
    with _lock:
        if key in _cache:
            return _cache[key]

    # compute embedding outside lock (may be slow)
    emb = compute_fn(text)
    # normalize to python list of floats
    if hasattr(emb, "tolist"):
        emb = emb.tolist()
    emb = [float(x) for x in emb]

    with _lock:
        _cache[key] = emb
        try:
            _save_cache()
        except Exception:
            # Do not fail on cache persist error
            pass
    return emb
