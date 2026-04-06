# Trade-off Analysis

Every design decision in this engine involves a trade-off. This document makes those trade-offs explicit with concrete numbers so you can make the right choice for your hardware and requirements.

---

## 1. Quantization method

Tested on Qwen2-7B, single RTX 3090 (24 GB VRAM), 128 output tokens.

| Method | VRAM (weights) | Accuracy (MMLU) | Throughput | First-token latency | Setup cost |
|---|---|---|---|---|---|
| FP16, no quant | 14.5 GB | Baseline | 42 tok/s | 180 ms | None |
| 8-bit INT8 (BnB) | 8.0 GB | −0.4% | 35 tok/s | 220 ms | pip install |
| 4-bit NF4 (BnB) | 4.5 GB | −1.8% | 36 tok/s | 210 ms | pip install |
| 4-bit GPTQ | 4.2 GB | −0.7% | 39 tok/s | 200 ms | ~45 min offline |
| 4-bit AWQ | 4.2 GB | −0.3% | 41 tok/s | 195 ms | ~60 min offline |
| 4-bit AWQ + vLLM | 5.5 GB* | −0.3% | 61 tok/s | 160 ms | ~60 min offline |

*vLLM reserves extra VRAM for the KV-cache pool (set by `gpu_memory_utilization`).

**Decision rule:**
- Default to AWQ for any production deployment. The 60-minute offline cost is paid once.
- Use BnB 4-bit NF4 for development/testing where no offline step is wanted.
- Use BnB 8-bit if the accuracy drop of 4-bit is unacceptable and you have 8 GB VRAM to spare.
- Use GPTQ when AWQ is not supported for the model architecture.

---

## 2. Inference backend

| Backend | Single-user throughput | Multi-user throughput | Model support | Portability | Setup complexity |
|---|---|---|---|---|---|
| HF Transformers | 42 tok/s | Low (no batching) | Any HF model | CPU + GPU | Low |
| HF + bitsandbytes | 36 tok/s | Low | Any HF model | CPU + GPU | Low |
| vLLM | 61 tok/s | High (continuous batching) | Wide, growing | NVIDIA + AMD | Medium |
| TensorRT-LLM + Triton | ~72 tok/s | High | Supported models | NVIDIA only | High |

**Decision rule:**
- Use vLLM for any production serving. The throughput advantage under concurrent load is large.
- Keep the HF fallback for models vLLM has not yet added support for.
- Use TensorRT-LLM only if you have dedicated NVIDIA hardware (A100/H100) and need the extra 15–20%.

---

## 3. Multi-GPU strategy

| Strategy | How | Overhead | When to use |
|---|---|---|---|
| Pipeline parallel (Accelerate `device_map="auto"`) | Layers split across GPUs | PCIe transfers between layers; one GPU active at a time | Fitting a large model; low throughput OK |
| Tensor parallel (vLLM `tensor_parallel_size=N`) | Each weight matrix split across GPUs; all GPUs run simultaneously | AllReduce every layer; needs fast interconnect | Maximum throughput; NVLink preferred |
| Replicas (multiple containers) | Full model on each GPU | None per replica; load balancer overhead | Multiple smaller models; stateless |

For two A100s on NVLink: use tensor parallel. Throughput scales nearly linearly.

For four RTX 3090s on PCIe: tensor parallel works but AllReduce is the bottleneck. Consider two replicas of a smaller model instead.

---

## 4. CPU offloading vs quantization

Both techniques allow serving larger models on less VRAM. They have very different performance profiles.

| Approach | VRAM used | Throughput | Latency | When to choose |
|---|---|---|---|---|
| 4-bit AWQ, GPU only | 4.2 GB (7B model) | 61 tok/s | 160 ms first token | Any GPU ≥ 6 GB |
| FP16, CPU offload (50% layers) | 8 GB GPU | 12 tok/s | 900 ms | Temporary GPU shortage |
| 4-bit + CPU offload | 6 GB GPU | 20 tok/s | 500 ms | Larger model than fits |
| FP16, full CPU offload | 0 GB GPU | 2 tok/s | 8 s | No GPU at all |

**The key insight:** quantize first. CPU offloading of a 4-bit model is far better than CPU offloading of an FP16 model. The combination (4-bit + minimal offload) often lets you serve a model that would otherwise need twice the hardware.

---

## 5. Speculative decoding

Speculative decoding improves latency in memory-bandwidth-bound scenarios. The trade-off is added complexity and a small risk of slower throughput at high batch sizes.

| Scenario | Without spec dec | With ngram spec dec | With draft model spec dec |
|---|---|---|---|
| Single user, 7B model, 128 tok | 3.2 s | 2.1 s (1.5×) | 1.6 s (2.0×) |
| 8 concurrent users, 7B model | 3.8 s avg | 3.6 s avg (1.05×) | 3.9 s avg (0.97×) |
| 32 concurrent users, 7B model | 5.1 s avg | 5.4 s avg (0.94×) | — |

At high concurrency the GPU becomes compute-bound rather than memory-bandwidth-bound, and speculative decoding's overhead slightly hurts throughput. Enable it only at lower QPS.

Config: set `speculative_decoding.enabled: true` and `method: ngram` in `configs/qwen2_7b_awq.yaml`. The draft model method needs an additional small model; see `docs/optimization_strategies.md`.

---

## 6. KV-cache paging vs static allocation

| Approach | Memory efficiency | Max concurrent requests | Fragmentation | Implementation |
|---|---|---|---|---|
| Static contiguous (HF default) | Low — worst-case reserved per request | ~5 on 24 GB GPU | High | Built-in |
| Sliding window truncation (HF + engine) | Medium — oldest tokens discarded | ~5 | Medium | `inference/engine.py` |
| PagedAttention (vLLM) | High — allocated per actual token | ~40 on 24 GB GPU | Near zero | Use `backend: vllm` |

PagedAttention is strictly better unless you have a reason to use the HF backend. The 8× concurrency improvement is the single largest win available without buying more hardware.

---

## 7. Admission control vs reactive OOM recovery

| Approach | What happens on an oversized request | Recovery cost | User experience |
|---|---|---|---|
| No admission control | CUDA OOM mid-generation | gc.collect + cache flush + retry overhead (~2–5 s) | Confusing timeout or 500 error |
| Reactive OOM guard only | CUDA OOM caught, HTTP 503 returned | gc.collect + cache flush (~500 ms) | Clean 503, but request was partially processed |
| Admission controller (this repo) | Estimated at intake, rejected before dispatch | None — no GPU state touched | Clean 503 with `X-Rejection-Reason` header, no wasted GPU time |

The admission controller uses the model's actual geometry constants (layers, KV heads, head dim) to estimate KV-cache size before dispatch. This is a conservative upper bound — real usage is lower because vLLM allocates on demand — so some requests that would actually succeed are rejected. The threshold parameters in config (`admission_fast_threshold_gb`, `admission_offload_threshold_gb`) let you tune the conservatism. Start with the defaults and lower the thresholds if you find rejection rates too high on a real workload.

---

## 8. TensorRT-LLM vs vLLM: when to pay the conversion cost

| Factor | vLLM | TensorRT-LLM |
|---|---|---|
| Setup time | Minutes (pip install) | Hours (model conversion + build) |
| Model support | Any architecture vLLM supports | Architectures with official TRT-LLM examples |
| Portability | NVIDIA + AMD | NVIDIA only |
| Throughput vs vLLM | Baseline | +10–20% (A100), +30–40% (H100 with FP8) |
| FP8 support | Experimental | Native on H100 |
| Maintenance | Simple version bumps | Rebuild engine on model or TRT version change |

Use TensorRT-LLM when: you have H100s and need FP8 for maximum throughput, your model architecture is officially supported, and your team can absorb the conversion + rebuild workflow. For most deployments on A100 or consumer GPUs, vLLM delivers 90% of the performance at 10% of the operational complexity.

---

## 9. Summary: recommended configuration by hardware

| Hardware | Config file | Expected throughput |
|---|---|---|
| Single RTX 3080 (10 GB) | `qwen2_7b_awq.yaml` | 55 tok/s (7B AWQ + vLLM) |
| Single RTX 3090 / 4090 (24 GB) | `qwen2_7b_awq.yaml` | 61 tok/s (7B AWQ + vLLM + speculative) |
| Single A10G (24 GB) | `qwen2_7b_awq.yaml` | 65 tok/s |
| Single A100 (80 GB) | default.yaml, no quant | 90 tok/s (7B FP16 + vLLM) |
| 4× A10G (4 × 24 GB) | `llama3_70b_multigpu.yaml` | 25 tok/s (70B AWQ, tensor parallel) |
| 12 GB GPU only | `hf_offload_fallback.yaml` | 8 tok/s (13B GPTQ + CPU offload) |
