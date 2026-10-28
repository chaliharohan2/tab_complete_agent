"""
Client for the GPU-hosted zero-shot NLI classifier microservice, plus the
threshold logic that turns raw per-label scores into a routing decision
(which entity type(s), if any, should be searched right now).
"""
import logging

import httpx

from . import config

logger = logging.getLogger("classifier_client")

ALL_TYPES = list(config.ENTITY_TYPES.keys())

# Simple in-memory cache keyed on the full input text, mainly to absorb
# duplicate/retried requests cheaply. GPU inference is ~5-10ms, so unlike a
# CPU-hosted model we don't need to avoid calling it on every keystroke.
_CACHE: dict[str, "ClassifierResult"] = {}
_CACHE_MAX_SIZE = 2000


class ClassifierResult:
    def __init__(self, entity_types: list[str], scores: dict[str, float], source: str):
        self.entity_types = entity_types
        self.scores = scores
        self.source = source  # "model" or "fallback"

    def to_dict(self):
        return {
            "entity_types": self.entity_types,
            "scores": self.scores,
            "source": self.source,
        }


def decide_entity_types(scores: dict[str, float]) -> list[str]:
    """Pure function: raw scores -> list of entity types to search (possibly all, possibly none)."""
    off_topic = scores.get("off_topic", 0.0)
    entity_scores = {k: v for k, v in scores.items() if k in config.ENTITY_TYPES}
    if not entity_scores:
        return []

    top_type, top_score = max(entity_scores.items(), key=lambda kv: kv[1])

    # Case 1: confidently off-topic (e.g. "show me total revenue") -> no entity expected
    if off_topic > top_score + config.OFF_TOPIC_MARGIN:
        return []

    # Case 2: nothing stands out -> ambiguous, could be any type -> show all
    if top_score < config.AMBIGUOUS_FLOOR:
        return ALL_TYPES

    # Case 3: one or more types clearly dominate -> keep those within DOMINANCE_RATIO of the top
    threshold = top_score * config.DOMINANCE_RATIO
    return [t for t, s in entity_scores.items() if s >= threshold]


def classify(text: str) -> ClassifierResult:
    """
    Classify the user's full in-progress text. Empty text (nothing typed yet)
    is treated as fully ambiguous -> show all types, matching "show suggestions
    for everything until narrowed down".
    """
    if not text.strip():
        return ClassifierResult(ALL_TYPES, {t: 0.0 for t in ALL_TYPES}, source="fallback")

    cache_key = text.strip().lower()
    cached = _CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        resp = httpx.post(
            config.CLASSIFIER_URL,
            json={"text": text},
            timeout=config.CLASSIFIER_TIMEOUT_S,
        )
        resp.raise_for_status()
        scores = resp.json()["scores"]
    except Exception as e:
        logger.warning("Classifier service unreachable (%s); failing open to ALL types.", e)
        return ClassifierResult(ALL_TYPES, {t: 0.0 for t in ALL_TYPES}, source="fallback")

    entity_types = decide_entity_types(scores)
    result = ClassifierResult(entity_types, scores, source="model")

    if len(_CACHE) >= _CACHE_MAX_SIZE:
        _CACHE.clear()  # crude eviction; fine for a POC
    _CACHE[cache_key] = result

    return result
