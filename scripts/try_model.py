"""Load the classifier once and time it on one issue, then on eight issues in a single batch."""
import json
import time

from app.config import settings
from app.model import IssueClassifier

started = time.perf_counter()
clf = IssueClassifier(settings.weights_dir, settings.device, settings.dtype)
print(f"loaded on {clf.device} in {time.perf_counter() - started:.1f} s")

issues = [json.loads(line) for line in open("data/test_issues.jsonl")][:8]
pairs = [(row["title"], row["body"]) for row in issues]
clf.predict(pairs[:1])                                    # warm-up: the first call is always slower

started = time.perf_counter()
label, confidence = clf.predict(pairs[:1])[0]
one = time.perf_counter() - started
print(f"1 issue : {one * 1000:.0f} ms  -> {label} ({confidence:.2f})  title: {issues[0]['title'][:60]}")

started = time.perf_counter()
results = clf.predict(pairs)
eight = time.perf_counter() - started
print(f"8 issues: {eight * 1000:.0f} ms in one batch = {eight / 8 * 1000:.0f} ms per issue")
for row, (label, confidence) in zip(issues, results):
    print(f"  {label:<15} {confidence:.2f}  (true: {row['label']:<15}) {row['title'][:55]}")
