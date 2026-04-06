# Optimization Strategies

This document explains every optimization technique in the engine: what it is, how it works at the hardware level, how it is implemented here, and when to use it.

---

## 1. Quantization

### The problem

A 7B parameter model stores each weight as a 16-bit float (FP16). That is 7 × 10⁹ × 2 bytes = 14 GB just to hold the weights, before any activations or KV-cache. Most single GPUs have 8–24 GB of VRAM.

Quantization replaces the 16-bit weights with lower-precision integers. The model gets smaller; inference gets faster because smaller data fits in cache and requires less memory bandwidth.

### 4-bit AWQ — primary method

AWQ (Activation-Aware Weight Quantization, Lin et al. MLSys 2024) is the best-accuracy 4-bit method available today. The key insight: not all weight channels are equally important. Roughly 1% of channels — the ones that correspond to large activation magnitudes — contribute most of the model's output quality. AWQ identifies those channels using a calibration dataset, scales them up before quantization so they survive with more precision, then quantizes everything to 4 bits.

Result: 4× memory reduction, less than 0.5% accuracy loss on standard benchmarks, faster loading via fused layers (`fuse_layers=True` in autoawq).

Implementation in this repo: `inference/quantization.py:quantize_awq()`. Called offline via `scripts/quantize.py awq`. The resulting artifact is loaded by vLLM natively (`quantization="awq"` in `LLM()`).

When to use: always in production when you have a GPU and can run the offline job once.

### 4-bit GPTQ — alternative

GPTQ (Frantar et al. ICLR 2023) uses approximate second-order information (the Hessian of the loss) to quantize each layer sequentially, compensating for each quantization error by adjusting remaining unquantized weights. This is also an offline process.

Result: 4× memory reduction, less than 1% accuracy loss at 4-bit with group_size=128.

Implementation: `inference/quantization.py:quantize_gptq()`. vLLM supports GPTQ natively.

When to use: when AWQ is not available for a particular model architecture, or when you need 3-bit compression (GPTQ supports 3-bit, AWQ does not).

### 8-bit / 4-bit NF4 via bitsandbytes — fallback

bitsandbytes runs quantization at load time — no offline step. For 8-bit, it uses the LLM.int8 algorithm: identify outlier feature dimensions (which carry most of the information), keep those in FP16, and compute everything else in INT8. For 4-bit NF4, it uses a data type optimised for the approximately normal distribution of pretrained weights.

Result: 2× (8-bit) or 4× (4-bit) memory reduction. More accuracy loss than AWQ/GPTQ because there is no calibration step. Slower than AWQ because there are no fused kernels.

Implementation: `inference/engine.py:_build_bnb_config()`. Used automatically when the HF fallback backend is active.

When to use: development, quick experiments, or when the model has no AWQ/GPTQ checkpoint available and you need to serve it today.

---

## 2. KV-cache optimisation — PagedAttention

### The problem

During autoregressive generation, every previously seen token's key and value tensors must be stored so they are not recomputed. For a standard Transformers model, this is allocated as a contiguous block per request:

```
kv_bytes = 2 × num_layers × num_kv_heads × head_dim × max_seq_len × bytes_per_element
```

For a 7B model with 32 layers, 32 KV heads, 128 head dim, and a 4096-token context, at FP16:
`2 × 32 × 32 × 128 × 4096 × 2 = 2.1 GB` reserved per request, even if the actual sequence is 100 tokens long.

With 10 concurrent users that is 21 GB reserved, nearly all of a 24 GB GPU — for a 4 GB model.

### PagedAttention (vLLM)

vLLM's PagedAttention divides the KV cache into fixed-size blocks (default: 16 tokens per block). A page table maps each logical sequence position to a physical block. Blocks are allocated on demand as tokens are generated and freed immediately when a request completes. Sequences with the same prefix can share blocks.

This is directly analogous to virtual memory paging in an operating system: logical address space (token position) is decoupled from physical memory allocation (which block holds it).

Result: 2–4× more concurrent requests on the same hardware, near-zero fragmentation, automatic prefix caching for repeated system prompts.

Implementation: used automatically when `backend: vllm` in config. Key parameters:
- `gpu_memory_utilization: 0.90` — fraction of VRAM dedicated to the KV pool (leave 10% for weights and activations)
- `max_model_len` — maximum context length; determines block table size

### HF fallback: sliding-window truncation

When the HF backend is used, unbounded KV growth in multi-turn conversations is prevented by the `KVCacheManager` logic in `inference/engine.py`. When `past_key_values` exceeds `max_ctx_tokens`, the oldest tokens are discarded from dim=2 of each layer's K and V tensors. The model loses distant context but avoids OOM.

---

## 3. CPU ↔ GPU weight offloading

### When it applies

After 4-bit quantisation, a 7B model is ~4 GB and fits easily on a 10 GB GPU. But a 70B model at 4-bit is ~35 GB — still larger than a single 24 GB GPU. Options: buy more GPUs, use tensor parallelism, or offload some layers to CPU RAM.

### How Accelerate device_map works

Setting `device_map="auto"` with a `max_memory` dict tells Accelerate's `infer_auto_device_map()` to assign each layer to the first device that has room for it, in order. The GPU fills up first; overflow goes to CPU. At inference time, each layer's inputs are moved to the device where that layer lives, the layer runs, and the output moves to the next layer's device.

```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    max_memory={"0": "11GiB", "cpu": "48GiB"},
    offload_folder="./offload_cache",
)
```

The `offload_folder` is used for disk offloading when even CPU RAM is exhausted — disk is much slower but allows serving models that would otherwise be impossible.

### Trade-off

PCIe 4.0 ×16 bandwidth is ~32 GB/s. GPU HBM bandwidth on an A100 is 2 TB/s — 60× faster. Every layer that lives on CPU introduces a transfer at each forward pass. A heavily offloaded 13B model drops from ~38 tok/s (full GPU, 4-bit) to ~8 tok/s. Use offloading only when latency SLAs permit it (batch jobs, background processing) or as a last resort for very large models.

Config: `configs/hf_offload_fallback.yaml`.

---

## 4. Tensor parallelism (multi-GPU)

For models that are too large even for offloading, or where you need throughput and have multiple GPUs, tensor parallelism splits each weight matrix across GPUs. Every GPU holds a horizontal slice of every attention and FFN weight matrix. On each forward pass they all run simultaneously and exchange results via AllReduce.

vLLM implements this with `tensor_parallel_size=N`, where N must evenly divide the number of attention heads. The config `num_gpus: 4` passes this automatically.

```yaml
hardware:
  num_gpus: 4          # vLLM: tensor_parallel_size=4
  device: cuda
```

On NVLink-connected GPUs (A100, H100 SXM), AllReduce is ~600 GB/s and the overhead is negligible. On PCIe-connected consumer GPUs it is ~32 GB/s and adds meaningful latency for each transformer layer.

Config: `configs/llama3_70b_multigpu.yaml`.

---

## 5. CUDA / NVIDIA acceleration

### Tensor Cores (FP16/BF16)

NVIDIA Ampere and later GPUs contain Tensor Cores — dedicated matrix multiply units that operate on FP16 or BF16 inputs and produce FP32 or FP16 outputs. They run at roughly 2× the throughput of standard CUDA cores for GEMM operations.

Two settings activate them:

```python
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.benchmark = True
```

These are set at startup in `run.py` via `utils/memory.py:enable_tensor_cores()`, before any model is loaded. `cudnn.benchmark = True` tells cuDNN to auto-tune its kernel selection for the shapes it observes on the first few forward passes.

### cuBLAS and cuDNN

PyTorch uses cuBLAS for all GEMM operations (the dominant computation in transformer attention and FFN layers) and cuDNN for fused operations. These are used automatically whenever a tensor is on a CUDA device. The settings above tell cuBLAS to route through Tensor Core paths when the input dtype is FP16 or BF16.

### vLLM custom Triton kernels

vLLM ships its own Triton kernels for the attention operation, optimised specifically for the PagedAttention memory layout. These run on both NVIDIA and AMD GPUs and are faster than the standard PyTorch scaled dot-product attention for the paged case.

### TensorRT-LLM (premium NVIDIA path)

TensorRT-LLM is NVIDIA's framework for converting transformer models into optimised TensorRT engines. It applies:

- **Kernel fusion**: attention + layernorm + activation compiled into single CUDA kernels, eliminating intermediate memory writes
- **FP8 quantization** on Hopper GPUs (H100): halves memory bandwidth vs FP16 with near-zero accuracy loss
- **In-flight batching** via Triton Inference Server: requests are batched dynamically as they arrive rather than waiting for a full batch

The conversion pipeline is in `scripts/tensorrt_convert.py`. Three steps: `convert` (HF → TRT checkpoint), `prepare-triton` (build Triton model repo), `serve-cmd` (print the docker launch command).

Expected throughput gain over vLLM: 10–20% on A100, 30–40% on H100 with FP8.

---

## 6. Memory fragmentation prevention

The CUDA memory allocator can become fragmented when many allocations of different sizes are made and freed in rapid succession. Two mitigations are applied:

**Allocator split size cap**: setting `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512` prevents the allocator from splitting large cached blocks into pieces smaller than 512 MB. This reduces the chance that a large allocation fails because the free memory is fragmented into many small chunks.

**Proactive cache flush**: after each generation request, `engine._maybe_flush_cache()` checks the free fraction on each GPU. If it is below 10%, it calls `gc.collect()` and `torch.cuda.empty_cache()`, which returns unused cached memory from PyTorch's pool back to the CUDA allocator. This is proactive rather than reactive — we flush before OOM, not after.

**OOM guard**: `utils/memory.py:OOMGuard` is a context manager that wraps the generation call. If `torch.cuda.OutOfMemoryError` is raised, it flushes the cache and re-raises a plain Python `MemoryError`. The server catches `MemoryError` and returns HTTP 503. This prevents a single oversized request from crashing the server process.

---

## 7. Speculative decoding

Speculative decoding improves inter-token latency on memory-bandwidth-bound workloads (low batch sizes, large models). It works by having a small fast draft model predict the next N tokens, then verifying all N with the main model in a single forward pass. When the draft is correct, N tokens are produced in the time it previously took to produce one.

vLLM supports three speculative decoding methods:

**Ngram**: uses repeated n-gram patterns in the prompt as draft predictions. No extra model required. Effective for documents with repetitive structure (code, structured data).

**Draft model**: a small model (e.g. 0.5B) generates draft tokens for a large model (e.g. 7B) to verify. Requires an additional model that shares the same vocabulary and architecture family.

**EAGLE**: a learned draft head that predicts future token distributions from the main model's hidden states. Best acceptance rate but requires a purpose-trained EAGLE head.

Configuration:

```yaml
speculative_decoding:
  enabled: true
  method: ngram                 # ngram | draft_model
  num_speculative_tokens: 5
  ngram_prompt_lookup_max: 4
```

Expected latency improvement: 1.5–2× on memory-bound workloads (single user, large model). No improvement or slight regression at high batch sizes where the GPU is already compute-bound.

---

## 8. Admission control

The admission controller (`serving/admission.py`) estimates the KV-cache footprint of an incoming request before dispatching it, using the model's actual architecture constants:

```
kv_bytes = 2 × num_layers × num_kv_heads × head_dim × (prompt_tokens + max_new_tokens) × bytes_per_element
```

It then compares the estimate against current free GPU VRAM and routes to one of four paths:

| Verdict | Condition | Action |
|---|---|---|
| `vllm` | `free_gb >= kv_gb + headroom + fast_threshold` | Dispatch immediately |
| `hf_offload` | `free_gb >= kv_gb + headroom + offload_threshold` | Use CPU offload path |
| `queue` | `free_gb >= kv_gb + headroom` and queue not full | Defer until memory frees |
| `reject` | None of the above | Return HTTP 503 immediately |

This is better than waiting for CUDA OOM because: OOM errors happen mid-generation and leave the allocator in a dirty state; admission control rejects early with a clean error, no GPU state to clean up, and meaningful diagnostic information in the response headers.
