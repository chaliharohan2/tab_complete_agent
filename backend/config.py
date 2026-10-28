"""
Central config for the POC: entity type definitions, classifier service location,
and the thresholds used to turn raw classifier scores into a routing decision.

Nothing here is trained -- these are hand-written zero-shot label descriptions.
Tune them freely; that's the expected way to improve this kind of classifier.
"""
import os

# --- Entity types & their zero-shot "hypothesis" descriptions -------------
# Each value is plugged into the NLI hypothesis template:
#   "This example is about {description}."
ENTITY_TYPES: dict[str, str] = {
    "person": "a sales person or salesperson's name",
    "location": "a city, place, or geographic location",
    "product": "a product or item name",
    "client": "a client, customer, or company name",
}

OFF_TOPIC_DESC = (
    "a general question that is not asking to search for a specific "
    "person, location, product, or client"
)

# --- Classifier microservice (hosted on the GPU box) -----------------------
CLASSIFIER_URL = os.environ.get("CLASSIFIER_URL", "http://spark-19de.local:8800/classify")
CLASSIFIER_TIMEOUT_S = float(os.environ.get("CLASSIFIER_TIMEOUT_S", "3.0"))

# --- Routing thresholds (tune empirically, no training involved) -----------
# If off_topic beats every entity score by this much, assume no entity is expected.
OFF_TOPIC_MARGIN = 0.15
# If the best entity score is below this, no type dominates -> treat as ambiguous
# (show suggestions from ALL entity types).
AMBIGUOUS_FLOOR = 0.20
# Among entity types, include any type whose score is within this fraction of the
# top score (lets 2+ plausible types show up together, e.g. person vs client).
DOMINANCE_RATIO = 0.6

# --- Fuzzy search ------------------------------------------------------------
DB_PATH = os.environ.get("DB_PATH", os.path.join(os.path.dirname(__file__), "..", "data", "app.db"))
MIN_TRIGRAM_QUERY_LEN = 3  # FTS5 trigram needs >=3 chars to use the index well
RESULTS_PER_TYPE = 5
