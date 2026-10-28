"""
Zero-shot entity-type classifier microservice.

Wraps MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33 (an NLI cross-encoder,
~70M params) behind a tiny FastAPI endpoint. Meant to run on a GPU box for
~5-10ms inference latency (CPU inference with this architecture is ~1-2s,
which is why this lives on its own service rather than in-process).

No training/fine-tuning involved -- this is pure zero-shot classification via
hand-written label hypotheses (see ENTITY_TYPES / OFF_TOPIC_DESC below, kept in
sync with backend/config.py).

Run with:
    uvicorn service:app --host 0.0.0.0 --port 8800
"""
import logging

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("classifier_service")

MODEL_NAME = "MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33"
HYP_TEMPLATE = "This example is about {}."

# Keep these descriptions in sync with backend/config.py (ENTITY_TYPES / OFF_TOPIC_DESC).
ENTITY_TYPES = {
    "person": "a sales person or salesperson's name",
    "location": "a city, place, or geographic location",
    "product": "a product or item name",
    "client": "a client, customer, or company name",
}
OFF_TOPIC_DESC = (
    "a general question that is not asking to search for a specific "
    "person, location, product, or client"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Loading %s on %s", MODEL_NAME, device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
model.eval()

_id2label = model.config.id2label
ENTAIL_IDX = [i for i, label in _id2label.items() if "entail" in label.lower()][0]

LABELS = list(ENTITY_TYPES.keys()) + ["off_topic"]
DESCRIPTIONS = list(ENTITY_TYPES.values()) + [OFF_TOPIC_DESC]
HYPOTHESES = [HYP_TEMPLATE.format(d) for d in DESCRIPTIONS]

# Warmup so the first real request isn't slow.
with torch.no_grad():
    _warm = tokenizer(["warmup"] * len(HYPOTHESES), HYPOTHESES, return_tensors="pt", padding=True, truncation=True).to(device)
    model(**_warm)
    if device == "cuda":
        torch.cuda.synchronize()
logger.info("Model warmed up and ready.")

app = FastAPI(title="Entity-Type Zero-Shot Classifier")


class ClassifyRequest(BaseModel):
    text: str
    labels: dict[str, str] | None = None  # optional override: name -> description


@app.post("/classify")
def classify(req: ClassifyRequest):
    if req.labels:
        labels = list(req.labels.keys()) + ["off_topic"]
        hypotheses = [HYP_TEMPLATE.format(d) for d in req.labels.values()] + [
            HYP_TEMPLATE.format(OFF_TOPIC_DESC)
        ]
    else:
        labels, hypotheses = LABELS, HYPOTHESES

    premises = [req.text] * len(hypotheses)
    inputs = tokenizer(premises, hypotheses, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    if device == "cuda":
        torch.cuda.synchronize()
    scores = torch.softmax(logits, dim=-1)[:, ENTAIL_IDX].tolist()

    return {"scores": dict(zip(labels, scores))}


@app.get("/health")
def health():
    return {"status": "ok", "device": device, "model": MODEL_NAME}
