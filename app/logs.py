"""One JSON object per log line, so Cloud Logging, Loki or Datadog can search and filter it."""
import json
import logging
import sys
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "time": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "severity": record.levelname,       # the field name Google Cloud Logging reads
            "message": record.getMessage(),
            **getattr(record, "fields", {}),
        }
        return json.dumps(entry)


def setup_logging(level: str) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=level, handlers=[handler], force=True)
