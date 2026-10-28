"""
Client for the GPU-hosted next-entity router (classifier_service/). The
service makes the routing decision itself (keyword rules + LLM yes/no
scoring); this module just calls it, caches results, and fails open.
"""
import logging

import httpx

from . import config

logger = logging.getLogger("classifier_client")

ALL_TYPES = list(config.ENTITY_TYPES)

# Keyed on the full input text; absorbs duplicate/retried requests and
# re-typing after backspace without another ~125ms LLM call.
_CACHE: dict[str, "ClassifierResult"] = {}
_CACHE_MAX_SIZE = 2000


class ClassifierResult:
    def __init__(self, entity_types: list[str], scores: dict[str, float], source: str):
        self.entity_types = entity_types
        self.scores = scores
        self.source = source  # "rule", "llm", or "fallback"


def classify(text: str) -> ClassifierResult:
    """Empty text is treated as fully ambiguous -> show all types."""
    if not text.strip():
        return ClassifierResult(ALL_TYPES, {t: 0.0 for t in ALL_TYPES}, source="fallback")

    cache_key = text.strip().lower()
    cached = _CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        resp = httpx.post(config.CLASSIFIER_URL, json={"text": text}, timeout=config.CLASSIFIER_TIMEOUT_S)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("Classifier service unreachable (%s); failing open to ALL types.", e)
        return ClassifierResult(ALL_TYPES, {t: 0.0 for t in ALL_TYPES}, source="fallback")

    result = ClassifierResult(data["entity_types"], data["scores"], source=data["source"])
    if len(_CACHE) >= _CACHE_MAX_SIZE:
        _CACHE.clear()  # crude eviction; fine for a POC
    _CACHE[cache_key] = result
    return result
