# System Architecture

## Pipeline overview

```
┌─────────────────────────────────────────────────────────────┐
│  Offline (one-time)                                         │
│                                                             │
│  HuggingFace base model                                     │
│       │                                                     │
│       ▼  scripts/quantize.py awq                            │
│  4-bit AWQ artifact  ──────────────────►  S3 bucket         │
└─────────────────────────────────────────────────────────────┘
                                               │
                          ┌────────────────────┘
                          │  container init (k8s initContainer / ECS)
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Inference server (Docker / ECS / EKS)                      │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  FastAPI gateway  (serving/server.py)                │   │
│  │  • Admission controller  ──► route or reject early   │   │
│  │  • /generate  /stream  /batch                        │   │
│  │  • /v1/completions  /v1/chat/completions (OpenAI)   │   │
│  │  • /health  /metrics                                 │   │
│  └─────────────────────┬────────────────────────────────┘   │
│                        │                                    │
│           ┌────────────┴────────────┐                       │
│           ▼                         ▼                       │
│  ┌──────────────────┐    ┌────────────────────────┐         │
│  │  vLLM engine     │    │  HF + bitsandbytes     │         │
│  │  (primary)       │    │  (fallback)            │         │
│  │                  │    │                        │         │
│  │  PagedAttention  │    │  device_map="auto"     │         │
│  │  Tensor parallel │    │  CPU↔GPU offloading    │         │
│  │  AWQ/GPTQ native │    │  BnB 4-bit/8-bit       │         │
│  │  Speculative dec │    │  TextIteratorStreamer   │         │
│  └──────────────────┘    └────────────────────────┘         │
│           │                                                 │
│           └────────────────────────────────────────────►   │
│                     CUDA / cuBLAS / cuDNN                   │
│                     Tensor Cores (FP16/BF16)                │
│                     NVLink (multi-GPU)                      │
└─────────────────────────────────────────────────────────────┘
         │
         ▼  KEDA ScaledObject (Prometheus metrics)
  GPU memory, queue depth, p95 latency  →  autoscale replicas
```

## Module map

| Module | Responsibility |
|---|---|
| `inference/engine.py` | Two-backend dispatch: vLLM primary, HF fallback. Public interface: `load`, `generate`, `stream`, `batch_generate`, `apply_chat_template`, `memory_stats`, `shutdown` |
| `inference/quantization.py` | Offline quantization jobs (AWQ, GPTQ, BnB). Also `build_quantization_config()` for runtime BnB config |
| `serving/server.py` | FastAPI app with six endpoints. Calls admission controller before dispatching. Converts CUDA OOM → HTTP 503 |
| `serving/schemas.py` | All Pydantic request/response models. Single source of truth for every type imported in server.py |
| `serving/admission.py` | Memory-budget router. Estimates KV-cache footprint from model geometry and routes to fast/offload/queue/reject |
| `utils/memory.py` | CUDA allocator config, Tensor Core paths, OOM guard context manager |
| `utils/benchmarks.py` | Latency (p50/p95/p99), throughput at varying batch sizes |
| `utils/logging.py` | Rotating file + stdout logger |
| `run.py` | CLI entrypoint: api / prompt / chat / quantize / bench modes |
| `scripts/quantize.py` | Offline quantization CLI (AWQ, GPTQ, BnB) |
| `scripts/tensorrt_convert.py` | TensorRT-LLM conversion and Triton preparation (NVIDIA premium path) |
| `scripts/upload_s3.py` | Upload artifacts to S3 |
| `deployment/Dockerfile` | Multi-stage CUDA 12.1 image |
| `deployment/docker-compose.yaml` | Local development and quantization jobs |
| `deployment/k8s/deployment.yaml` | Kubernetes deployment with GPU resource limits and S3 init container |
| `deployment/k8s/keda_scaledobject.yaml` | GPU-aware autoscaling via KEDA + Prometheus |
| `deployment/terraform/main.tf` | AWS infra: S3, ECR, ECS, ALB, EFS, CloudWatch, IAM |

## Request lifecycle

```
Client POST /generate
    │
    ▼
FastAPI validates schema (Pydantic)
    │  400 if invalid
    ▼
Admission controller
    │  estimates KV-cache footprint from model geometry
    │  checks free GPU VRAM
    │  routes to: vllm | hf_offload | queue | reject(503)
    ▼
asyncio.get_event_loop().run_in_executor(None, engine.generate)
    │  runs in thread pool — does not block event loop
    ▼
engine.generate()
    │  vLLM: SamplingParams → LLM.generate()
    │  HF:   tokenize → model.generate() → decode
    ▼
OOMGuard context
    │  CUDA OutOfMemoryError → flush cache → MemoryError → HTTP 503
    ▼
_record(latency, tokens)   — updates sorted latency list for percentiles
    ▼
GenerateResponse JSON
```

## Autoscaling signal (why not CPU)

CPU utilisation is a poor signal for GPU inference:

- The Python process is mostly idle (GPU is doing the work)
- CPU spikes on tokenization, not on the actual generation bottleneck
- Scaling on CPU can add replicas that compete for the same GPU and make things worse

Better signals used in `deployment/k8s/keda_scaledobject.yaml`:

| Metric | Threshold | Meaning |
|---|---|---|
| Request queue depth | > 4 | Users are waiting — add capacity |
| GPU free VRAM | < 2 GB | OOM risk on next request |
| p95 latency | > 8 s | Requests are slow — need more replicas |

These are read from Prometheus (which scrapes `/metrics`) by KEDA, which then adjusts `Deployment.spec.replicas`.
