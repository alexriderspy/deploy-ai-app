"""The shape of what comes in and what goes out. FastAPI checks every request against these."""
from typing import Literal

from pydantic import BaseModel, Field

Label = Literal["bug", "feature request", "usage", "installation",
                "performance", "new model", "documentation", "unknown"]


class Issue(BaseModel):
    title: str = Field(min_length=1, max_length=300,
                       examples=["Engine crashes with CUDA OOM when loading Qwen3 with tensor parallel 4"])
    body: str = Field(default="", max_length=20_000)


class Prediction(BaseModel):
    label: Label
    confidence: float = Field(ge=0, le=1, description="probability of the label's first token")
    latency_ms: float
    model_version: str


class Health(BaseModel):
    status: Literal["ok"]
    model_version: str
    device: str
