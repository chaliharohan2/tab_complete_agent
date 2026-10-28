# tab_complete_agent

Entity-aware autocomplete POC. As the user types a search-style sentence
(e.g. `"Show me the sales for del"`), the app figures out (a) whether an
entity value is about to be typed, (b) which entity type(s) are relevant
(person / location / product / client), and (c) fuzzy-searches a demo DB for
matching values.

See [`docs/approach.md`](docs/approach.md) for the model selection writeup
and benchmark results that led to this design.

## Architecture

- **Frontend** (`frontend/`): plain HTML/JS, debounced input box, dropdown of
  tagged suggestions. Served as static files by the backend.
- **Backend** (`backend/`): FastAPI app.
  - `text_utils.py` -- rule-based split of in-progress text into the
    "query" (partial word currently being typed) used for search.
  - `classifier_client.py` -- calls the GPU-hosted classifier and turns raw
    scores into a routing decision (which entity types to search, if any).
  - `db.py` / `seed_data.py` -- SQLite stand-in for Elasticsearch: a plain
    table + an FTS5 trigram virtual table for fuzzy substring search.
- **Classifier microservice** (`classifier_service/`): wraps a small
  zero-shot NLI model (`MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33`,
  ~70M params). Runs on a GPU (~5-10ms/call); the same model takes ~1-2s/call
  on CPU, which is why it's a separate service rather than in-process.
- **`ops/`**: SSH helpers to deploy/restart the classifier service on the
  GPU box. Credentials are read from env vars (`GPU_HOST`, `GPU_USER`,
  `GPU_PASSWORD`), never hardcoded.

## Running locally

```bash
source venv/bin/activate
pip install -r backend/requirements.txt

# point at wherever the classifier service is running
export CLASSIFIER_URL=http://<gpu-host>:8800/classify

uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Then open http://localhost:8000/.

If the classifier service is unreachable, the backend fails open and treats
every query as ambiguous (shows all 4 entity types) rather than breaking.

## Deploying/restarting the classifier service on the GPU box

```bash
export GPU_HOST=... GPU_USER=... GPU_PASSWORD=...
python ops/deploy_classifier.py
```

This uploads `classifier_service/service.py` to `~/tab_complete_poc_bench`
on the GPU box (a venv with torch+CUDA already set up there) and
(re)starts it in the background on port 8800. For a longer-lived setup,
convert this into a systemd service instead of the nohup-based approach
used for this POC.

## Tuning the classifier

Nothing here is trained -- it's all zero-shot. The two places to tune:

- **Label descriptions** (`ENTITY_TYPES` / `OFF_TOPIC_DESC`, duplicated in
  `backend/config.py` and `classifier_service/service.py` -- keep them in
  sync): these are the natural-language descriptions plugged into the NLI
  hypothesis template. Wording changes here can meaningfully shift accuracy.
- **Routing thresholds** (`backend/config.py`): `OFF_TOPIC_MARGIN`,
  `AMBIGUOUS_FLOOR`, `DOMINANCE_RATIO` control how raw scores become a
  decision (no entity / all entities / a specific subset).

### Known limitation

The model sometimes over-associates generic sales phrasing (e.g. just
`"Show me the sales for"`) with the `client` label specifically, instead of
treating it as fully ambiguous across all 4 types. This is a zero-shot label
engineering problem, not a code bug -- worth iterating on the `client`
hypothesis wording or routing thresholds if it matters for your use case.

## Migrating to real Elasticsearch later

Only `backend/db.py` needs to change -- `fuzzy_search(query, types)` is the
entire interface the rest of the app depends on. Swap its internals for an
ES fuzzy/match query and nothing else needs to move.
