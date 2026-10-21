# Deploy your AI app: FastAPI + Docker + Cloud

A production-style API around a fine-tuned **Llama 3.2 1B** that labels vLLM GitHub issues
(bug, feature request, usage, installation, performance, new model, documentation).
The model comes from the fine-tuning video's LoRA adapter (`issue-triage-lora/`).

What's in it: FastAPI with Pydantic models, the model loaded once at startup (lifespan), dynamic batching,
timeouts and load shedding, an API key header, JSON logs, Prometheus metrics, pytest, a multi-stage
Dockerfile with a non-root user and a healthcheck, docker compose with Prometheus, GitHub Actions
(tests, then build, then push to GHCR) and a Google Cloud Run deploy script.

```
app/config.py       settings from environment variables
app/schemas.py      request and response models
app/prompt.py       the exact prompt the model was trained on
app/model.py        load the model, label a batch of issues
app/batcher.py      dynamic batching with a bounded queue
app/auth.py         X-API-Key check
app/logs.py         JSON logs
app/metrics.py      Prometheus counters and histograms
app/main.py         the FastAPI app: lifespan, middleware, /predict, /health, /metrics
tests/              pytest with TestClient and a fake model
scripts/            merge the adapter, load test, accuracy through the API
deploy/             Prometheus config, Cloud Run deploy script
```

## Quickstart (local, no Docker)
```bash
python -m venv .venv && source .venv/bin/activate
pip install torch peft -r requirements-dev.txt
cp .env.example .env                       # then put a real key in API_KEYS
python scripts/merge_adapter.py path/to/issue-triage-lora model
uvicorn app.main:app --port 8080 --no-access-log
pytest -q
```

## With Docker
```bash
docker compose up --build -d               # API on :8080, Prometheus on :9090
curl localhost:8080/health
curl localhost:8080/predict -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"title": "Engine crashes with CUDA OOM when loading Qwen3 with tensor parallel 4"}'
API_KEY=... python scripts/load_test.py -n 48 -c 1 4 16
API_KEY=... python scripts/eval_service.py
```

The weights (`model/`, 2.5 GB) are mounted into the container, not baked into the image, and are not in git.
Inside Docker on an ARM Mac (CPU only) float16 was 4.3x faster than bfloat16 and float32 did not fit in a 6 GB VM,
so compose sets `DTYPE=float16`; the Cloud Run script uses float32 on x86. Measure on your own hardware with
`python -m scripts.try_model`.

## CI and cloud
- `.github/workflows/ci.yml`: every push runs pytest; on `main` it then builds a linux/amd64 image and pushes it to
  `ghcr.io/<owner>/deploy-ai-app`.
- `deploy/cloudrun/deploy.sh`: Artifact Registry remote repo in front of GHCR, weights in Cloud Storage mounted as a
  volume, the API key in Secret Manager, then `gcloud run deploy`. It was written and explained in the video but not
  run, because the recording laptop has no cloud credentials.

`media/terminal/` has the exact output of every step from the run in the video (made with `capture.sh`).
