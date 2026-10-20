"""Print the settings the app would start with (the API key is a SecretStr, so it's never printed)."""
from app.config import settings

for name, value in settings.model_dump().items():
    print(f"{name:<18} {value!r}")
