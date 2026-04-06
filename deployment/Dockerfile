# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: dependency builder
# ─────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3.11 -m pip install --upgrade pip setuptools wheel

WORKDIR /build
COPY requirements.txt .

# Install to a prefix so we can copy cleanly to runtime stage
RUN pip install --prefix=/install -r requirements.txt


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: runtime
# ─────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    # Prevent CUDA allocator fragmentation
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
    # Disable HuggingFace tokenizer parallelism warning
    TOKENIZERS_PARALLELISM=false \
    # vLLM host worker count
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    OMP_NUM_THREADS=8

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3-pip libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

COPY --from=builder /install /usr/local

WORKDIR /app
COPY . .

RUN mkdir -p /app/logs /app/offload_cache /app/artifacts

# Non-root user
RUN useradd -m -u 1000 inference && chown -R inference:inference /app
USER inference

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -sf http://localhost:${PORT:-8000}/health | python3 -c \
        "import sys,json; d=json.load(sys.stdin); sys.exit(0 if d['status']=='ok' else 1)"

CMD ["python", "run.py", "--config", "configs/qwen2_7b_awq.yaml", "--mode", "api"]
