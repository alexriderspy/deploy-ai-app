"""API keys: the caller sends X-API-Key, and we compare it in constant time."""
import secrets

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from .config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: str | None = Security(api_key_header)) -> None:
    if not settings.keys:
        return                                   # no keys configured: local development
    if key is None or not any(secrets.compare_digest(key.encode(), k.encode()) for k in settings.keys):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing or invalid X-API-Key")
