"""What Pydantic does with good and bad requests, before FastAPI is even involved."""
from pydantic import ValidationError

from app.schemas import Issue

print(Issue(title="Engine crashes with CUDA OOM on 4 GPUs"))

for bad in [{"title": ""}, {"body": "no title at all"}, {"title": "x" * 301}]:
    try:
        Issue(**bad)
    except ValidationError as error:
        first = error.errors()[0]
        print(f"rejected: {first['loc'][0]!r}: {first['msg']}")
