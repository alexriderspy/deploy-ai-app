"""Every setting comes from an environment variable, so the same image runs on a laptop and in the cloud."""
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    weights_dir: str = "model"            # the merged fine-tuned model, from scripts/merge_adapter.py
    model_version: str = "issue-triage-llama-3.2-1b-v1"
    device: str = "auto"                  # auto, cpu, mps or cuda
    dtype: str = "bfloat16"
    api_keys: SecretStr = SecretStr("")   # comma separated; empty switches auth off (local only!)
    max_batch_size: int = 8               # most issues we put through the model in one go
    max_wait_ms: int = 10                 # how long to wait for more requests to fill a batch
    max_queue: int = 64                   # more waiting than this and we answer 503 straight away
    request_timeout_s: float = 30.0
    log_level: str = "INFO"

    @property
    def keys(self) -> set[str]:
        return {k.strip() for k in self.api_keys.get_secret_value().split(",") if k.strip()}


settings = Settings()
