import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from . import classifier_client, config, db, text_utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

app = FastAPI(title="Entity-Aware Autocomplete POC")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    db.init_db()


@app.get("/api/suggest")
def suggest(text: str = ""):
    context, query = text_utils.extract_context_and_query(text)
    # Classify on the FULL raw text (not the context with the in-progress word
    # stripped off): a word like "for" or "person" is a strong signal the
    # instant it's typed, even before the user hits the next space. GPU
    # inference is cheap enough (~5-10ms) that we don't need to economize by
    # only classifying at word boundaries.
    result = classifier_client.classify(text)

    suggestions = []
    if query and result.entity_types:
        suggestions = db.fuzzy_search(query, result.entity_types)

    return {
        "input": text,
        "context": context,
        "query": query,
        "entity_types": result.entity_types,
        "classifier_source": result.source,
        "scores": result.scores,
        "suggestions": suggestions,
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}


_frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(_frontend_dir):
    app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")
