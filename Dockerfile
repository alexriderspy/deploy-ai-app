# ---------- stage 1: build the Python environment ----------
FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
# CPU-only PyTorch: the default Linux x86 wheel also pulls in gigabytes of CUDA libraries
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt
# C++ headers are only needed to compile extensions; deleting them here really shrinks the image
RUN rm -rf /opt/venv/lib/python3.12/site-packages/torch/include \
    && find /opt/venv -name "__pycache__" -prune -exec rm -rf {} +

# ---------- stage 2: the image we ship ----------
FROM python:3.12-slim
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    TRANSFORMERS_VERBOSITY=error HF_HUB_OFFLINE=1 \
    WEIGHTS_DIR=/models/issue-triage DEVICE=cpu PORT=8080
RUN useradd --create-home --uid 10001 app
COPY --from=builder /opt/venv /opt/venv
WORKDIR /srv
COPY app ./app
USER app
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=4)"
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port $PORT --no-access-log"]
