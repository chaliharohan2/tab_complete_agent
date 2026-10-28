"""
Backend config. The routing logic itself (keyword rules, LLM prompt,
threshold) lives in classifier_service/ on the GPU box.
"""
import os

ENTITY_TYPES = ["person", "location", "product", "client"]

# --- Router microservice (hosted on the GPU box) ----------------------------
CLASSIFIER_URL = os.environ.get("CLASSIFIER_URL", "http://spark-19de.local:8800/classify")
CLASSIFIER_TIMEOUT_S = float(os.environ.get("CLASSIFIER_TIMEOUT_S", "3.0"))

# --- Fuzzy search ------------------------------------------------------------
DB_PATH = os.environ.get("DB_PATH", os.path.join(os.path.dirname(__file__), "..", "data", "app.db"))
MIN_TRIGRAM_QUERY_LEN = 3  # FTS5 trigram needs >=3 chars to use the index well
RESULTS_PER_TYPE = 5
