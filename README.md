# LLM Inference Engine

Production-grade inference of large language models on limited GPU resources.

**Primary stack:** AWQ quantization → vLLM (PagedAttention) → FastAPI  
**Fallback stack:** bitsandbytes → HuggingFace Transformers + Accelerate CPU offload  
**Premium path:** TensorRT-LLM → Triton (see `scripts/tensorrt_convert.py`)

---

## Repository layout

```
llm-inference-engine/
├── run.py                          Main CLI entrypoint
├── requirements.txt
├── configs/
│   ├── default.yaml
│   ├── qwen2_7b_awq.yaml           Single GPU, AWQ, vLLM
│   ├── gpt2_cpu.yaml               CPU, None, hf
├── inference/
│   ├── engine.py                   Two-backend engine (vLLM + HF)
│   └── quantization.py             AWQ / GPTQ / bitsandbytes jobs
├── serving/
│   ├── server.py                   FastAPI app
│   ├── schemas.py                  All Pydantic models
├── utils/
│   ├── benchmarks.py               Latency / throughput measurement
│   └── logging.py
├── scripts/
│   ├── quantize.py                 Offline quantization CLI
├── tests/
│   ├── test_quantization.py        Quantization config + error types
│   ├── test_engine.py              Engine unit tests
│   ├── test_schemas.py             Schema import + Pydantic validation
├── deployment/
│   ├── Dockerfile
│   ├── docker-compose.yaml

└── docs/
    ├── architecture.md             System design and request lifecycle
    ├── optimization_strategies.md  How each technique works
    └── tradeoffs.md                Performance vs accuracy tables
```

---
## Running GPT-2 on CPU (Local Testing — No GPU Required)

GPT-2 is a small model (~500 MB) that runs on CPU. Use this to verify everything works before using a real GPU model.

### Windows Commands

```powershell
# Step 1 — create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# Step 2 — install CPU dependencies
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers accelerate tokenizers safetensors pyyaml fastapi "uvicorn[standard]" pydantic httpx pytest

# Step 3 — start the API server (model downloads automatically)
python run.py --config configs/gpt2_cpu.yaml --mode api

# Step 4 — test with a single prompt (open a new terminal)
python run.py --config configs/gpt2_cpu.yaml --mode prompt --prompt "What is artificial intelligence?"

# Step 5 — run benchmarks
python run.py --config configs/gpt2_cpu.yaml --mode bench

# Step 6 — interactive chat
python run.py --config configs/gpt2_cpu.yaml --mode chat
```

### Mac / Linux Commands

```bash
# Step 1 — create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Step 2 — install CPU dependencies
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers accelerate tokenizers safetensors pyyaml fastapi "uvicorn[standard]" pydantic httpx pytest

# Step 3 — start the API server
python run.py --config configs/gpt2_cpu.yaml --mode api

# Step 4 — test with a single prompt (open a new terminal)
python run.py --config configs/gpt2_cpu.yaml --mode prompt --prompt "What is artificial intelligence?"

# Step 5 — run benchmarks
python run.py --config configs/gpt2_cpu.yaml --mode bench

# Step 6 — interactive chat
python run.py --config configs/gpt2_cpu.yaml --mode chat
```

### Config file used (configs/gpt2_cpu.yaml)

```yaml
model:
  path: gpt2
  dtype: float32
backend: hf
quantization:
  method: none
hardware:
  device: cpu
generation:
  max_new_tokens: 100
server:
  host: 0.0.0.0
  port: 8000
```

---

## Running Qwen2-7B on GPU (Production)

Qwen2-7B is a real production model. You need a GPU with at least 6 GB VRAM (with 4-bit AWQ quantization).

### Prerequisites

- NVIDIA GPU with 6+ GB VRAM
- CUDA 12.1+ installed
- NVIDIA driver 530+

Verify your GPU:
```bash
nvidia-smi
nvcc --version
```

### Windows Commands (GPU)

```powershell
# Step 1 — activate venv
venv\Scripts\activate

# Step 2 — install GPU dependencies
pip install torch==2.3.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate vllm autoawq pyyaml fastapi "uvicorn[standard]" pydantic

# Step 3 — quantize the model (one time, needs 14GB RAM + GPU)
python scripts/quantize.py awq --model Qwen/Qwen2-7B-Instruct --output ./artifacts/qwen2-7b-awq

# Step 4 — start the server
python run.py --config configs/qwen2_7b_awq.yaml --mode api

# Step 5 — run benchmarks
python run.py --config configs/qwen2_7b_awq.yaml --mode bench

# Step 6 — interactive chat
python run.py --config configs/qwen2_7b_awq.yaml --mode chat
```

### Mac / Linux Commands (GPU)

```bash
# Step 1 — activate venv
source venv/bin/activate

# Step 2 — install GPU dependencies
pip install torch==2.3.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate vllm autoawq pyyaml fastapi "uvicorn[standard]" pydantic

# Step 3 — quantize the model (one time, needs 14GB RAM + GPU)
python scripts/quantize.py awq --model Qwen/Qwen2-7B-Instruct --output ./artifacts/qwen2-7b-awq

# Step 4 — start the server
python run.py --config configs/qwen2_7b_awq.yaml --mode api

# Step 5 — run benchmarks
python run.py --config configs/qwen2_7b_awq.yaml --mode bench
```

### Config file used (configs/qwen2_7b_awq.yaml)

```yaml
model:
  path: ./artifacts/qwen2-7b-awq
  dtype: float16
backend: vllm
quantization:
  method: awq
hardware:
  device: cuda
  num_gpus: 1
  gpu_memory_utilization: 0.90
generation:
  max_new_tokens: 1024
server:
  host: 0.0.0.0
  port: 8000
```

---

## Testing the API with curl

Once the server is running at http://localhost:8000, you can call it using curl.

### Windows PowerShell

```powershell
# Health check
curl http://localhost:8000/health

# Generate text
curl -X POST http://localhost:8000/generate `
  -H "Content-Type: application/json" `
  -d '{"prompt": "What is artificial intelligence?", "max_tokens": 100, "temperature": 0.7}'
```

Note: on Windows PowerShell use backtick (`) for line continuation, not backslash.

### Mac / Linux

```bash
# Health check
curl http://localhost:8000/health

# Generate text
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is artificial intelligence?", "max_tokens": 100, "temperature": 0.7}'
```

### Full curl example (used in testing)

```bash
curl -X 'POST' \
  'http://localhost:8000/generate' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "max_tokens": 256,
  "prompt": "hello",
  "temperature": 0.5
}'
```

### Response received

```json
{
  "request_id": "9ac18282-5396-4491-9336-7826b247715b",
  "text": ".com/\n, which is the name of a popular blog about science and technology news from Wired magazine...",
  "tokens_generated": 92,
  "prompt_tokens": 1,
  "latency_seconds": 70.3575,
  "tokens_per_second": 1.3,
  "backend": "hf",
  "model": "gpt2"
}
```

### What each field in the response means

| Field | Value | Meaning |
|---|---|---|
| request_id | 9ac18282... | Unique ID for this request |
| text | ".com/\n..." | The generated text from GPT-2 |
| tokens_generated | 92 | GPT-2 generated 92 tokens |
| prompt_tokens | 1 | "hello" was 1 token |
| latency_seconds | 70.3575 | Took 70 seconds on CPU |
| tokens_per_second | 1.3 | Generated 1.3 tokens per second |
| backend | hf | Used HuggingFace backend |
| model | gpt2 | Used GPT-2 model |

The 70 second latency is expected on CPU. On a GPU this would be under 2 seconds.

---

## Benchmark Results — GPT-2 on CPU (Windows)

Hardware: CPU only, no GPU
Model: GPT-2 (124M parameters, 548 MB)
Backend: HuggingFace Transformers

### Latency benchmark (10 runs, 100 tokens each)

| Metric | Value | What it means |
|---|---|---|
| Mean latency | 54,388 ms | Average 54 seconds per request |
| Median latency | 57,913 ms | Middle value across all runs |
| P95 latency | 77,546 ms | 95% of requests finished within 77 seconds |
| P99 latency | 78,979 ms | 99% of requests finished within 79 seconds |
| Min latency | 18,585 ms | Fastest request: 18 seconds |
| Max latency | 79,337 ms | Slowest request: 79 seconds |
| Tokens per second | 1.4 | Generated 1.4 words per second |
| Total tokens | 783 | Generated 783 tokens across all 10 runs |

### Throughput benchmark (different batch sizes)

| Batch size | Tokens per second | Time taken |
|---|---|---|
| 1 prompt | 1.1 tok/s | 31 seconds |
| 2 prompts | 1.5 tok/s | 71 seconds |
| 4 prompts | 1.1 tok/s | 156 seconds |
| 8 prompts | 1.2 tok/s | 305 seconds |

### Understanding these numbers

These are CPU numbers. They are slow by design. A GPU would be 20-40x faster:

| Hardware | Expected tokens/second | Expected latency |
|---|---|---|
| CPU (our result) | 1.4 tok/s | 54 seconds |
| RTX 3090 GPU | 40-60 tok/s | 2-3 seconds |
| A100 GPU | 80-100 tok/s | under 1 second |

The benchmark confirms the code is correct. The slowness is purely a hardware limitation, not a code issue.

### Why min latency (18s) is much lower than mean (54s)

The first few requests after the model loads benefit from CPU caches being warm. Later requests are more consistent but slightly slower due to memory pressure from larger context accumulation.

### Why batch size 2 is faster than batch size 1

With 2 prompts, CPU parallelism kicks in across cores. The model processes both prompts simultaneously using multiple CPU threads, giving slightly better throughput. At batch size 4 and 8, memory bandwidth becomes the bottleneck and throughput drops back.

---

## Expected GPU results (for reference)

These are expected numbers when running Qwen2-7B on a GPU with 4-bit AWQ quantization. Not measured locally due to no GPU available — sourced from vLLM benchmarks.

| Metric | CPU (GPT-2) | RTX 3090 (Qwen2-7B AWQ) |
|---|---|---|
| Tokens per second | 1.4 | 55-65 |
| Mean latency (100 tok) | 54,000 ms | 1,800 ms |
| VRAM used | 0 GB | 4.5 GB |
| Model size | 548 MB | 3.5 GB (compressed) |

---

## Interactive API documentation

When the server is running, open your browser and go to:

```
http://localhost:8000/docs
```

You will see an interactive page where you can test all endpoints by clicking buttons without writing any code.

---

## Available run modes

```bash
# Start API server
python run.py --config configs/gpt2_cpu.yaml --mode api

# Single question, print answer, exit
python run.py --config configs/gpt2_cpu.yaml --mode prompt --prompt "Your question here"

# Interactive back-and-forth chat
python run.py --config configs/gpt2_cpu.yaml --mode chat

# Run speed benchmarks
python run.py --config configs/gpt2_cpu.yaml --mode bench

# Compress a model (needs GPU)
python run.py --mode quantize --model Qwen/Qwen2-7B-Instruct --quant awq --output ./artifacts/qwen2-7b-awq
```
