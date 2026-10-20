"""The HTTP service: load the model once at startup, then batch, time out, authenticate, log and count."""
import asyncio
import logging
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .auth import require_api_key
from .batcher import MicroBatcher, QueueFullError
from .config import settings
from .logs import setup_logging
from .metrics import LATENCY, PREDICTIONS, REQUESTS
from .schemas import Health, Issue, Prediction

log = logging.getLogger("triage")


def load_classifier():
    from .model import IssueClassifier          # imported here, so the tests never need torch
    return IssueClassifier(settings.weights_dir, settings.device, settings.dtype)


def add_routes(app: FastAPI) -> None:
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        if request.url.path not in ("/health", "/metrics"):      # health checks would drown the logs
            log.info("request", extra={"fields": {
                "request_id": request_id, "method": request.method, "path": request.url.path,
                "status": response.status_code, "ms": round((time.perf_counter() - started) * 1000, 1)}})
        return response

    @app.get("/health", response_model=Health)
    def health(request: Request) -> Health:
        return Health(status="ok", model_version=settings.model_version,
                      device=request.app.state.classifier.device)

    @app.post("/predict", response_model=Prediction, dependencies=[Depends(require_api_key)])
    async def predict(issue: Issue, request: Request) -> Prediction:
        started = time.perf_counter()
        try:
            label, confidence = await asyncio.wait_for(
                request.app.state.batcher.submit((issue.title, issue.body)), settings.request_timeout_s)
        except QueueFullError:
            REQUESTS.labels("overloaded").inc()
            raise HTTPException(503, "too many requests waiting, retry shortly", headers={"Retry-After": "1"})
        except TimeoutError:
            REQUESTS.labels("timeout").inc()
            raise HTTPException(504, f"no answer within {settings.request_timeout_s} s")
        seconds = time.perf_counter() - started
        REQUESTS.labels("ok").inc()
        LATENCY.observe(seconds)
        PREDICTIONS.labels(label).inc()
        return Prediction(label=label, confidence=round(confidence, 3),
                          latency_ms=round(seconds * 1000, 1), model_version=settings.model_version)

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def create_app(load: Callable = load_classifier) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging(settings.log_level)
        started = time.perf_counter()
        app.state.classifier = load()             # once per process, before the first request
        app.state.batcher = MicroBatcher(app.state.classifier.predict, settings.max_batch_size,
                                         settings.max_wait_ms, settings.max_queue)
        app.state.batcher.start()
        log.info("model loaded", extra={"fields": {
            "seconds": round(time.perf_counter() - started, 1), "device": app.state.classifier.device,
            "max_batch_size": settings.max_batch_size, "auth": bool(settings.keys)}})
        yield
        await app.state.batcher.stop()

    app = FastAPI(title="Issue triage API", version="1.0.0", lifespan=lifespan)
    add_routes(app)
    return app


app = create_app()
