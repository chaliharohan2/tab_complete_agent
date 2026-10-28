"""
Next-entity router microservice (rules + Qwen3-8B yes/no scoring, see
router.py). Runs on the GPU box; ~125ms per LLM-routed request, ~0ms when the
keyword rules fire.

Run with:
    uvicorn service:app --host 0.0.0.0 --port 8800
"""
import logging
import os
import time

from fastapi import FastAPI
from pydantic import BaseModel

from router import DEFAULT_MODEL, DEFAULT_THRESHOLD, Router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("classifier_service")

MODEL_NAME = os.environ.get("ROUTER_MODEL", DEFAULT_MODEL)
THRESHOLD = float(os.environ.get("ROUTER_THRESHOLD", DEFAULT_THRESHOLD))

logger.info("Loading router model %s", MODEL_NAME)
router = Router(MODEL_NAME, threshold=THRESHOLD)
logger.info("Router ready.")

app = FastAPI(title="Next-Entity Router")


class ClassifyRequest(BaseModel):
    text: str


@app.post("/classify")
def classify(req: ClassifyRequest):
    t0 = time.time()
    result = router.route(req.text)
    result["latency_ms"] = round((time.time() - t0) * 1000, 1)
    return result


@app.get("/health")
def health():
    return {"status": "ok", "device": router.lm.device, "model": MODEL_NAME, "threshold": THRESHOLD}
