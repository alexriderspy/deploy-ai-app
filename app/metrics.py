"""Prometheus metrics: counters only go up, histograms count how many values fell into each bucket."""
from prometheus_client import Counter, Histogram

REQUESTS = Counter("triage_requests_total", "Requests to /predict, by outcome", ["status"])
LATENCY = Histogram("triage_request_seconds", "Time to answer a /predict request",
                    buckets=(0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32))
BATCH_SIZE = Histogram("triage_batch_size", "Issues per model call", buckets=(1, 2, 4, 8, 16, 32))
PREDICTIONS = Counter("triage_predictions_total", "Labels we predicted", ["label"])
