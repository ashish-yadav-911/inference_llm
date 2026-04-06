
from __future__ import annotations

import argparse
import statistics
import time
from typing import List, Optional

SAMPLE_PROMPTS = [
    "What is the capital of France?",
    "Explain gradient descent in three sentences.",
    "Write a haiku about GPUs.",
    "What are the main differences between Python 2 and Python 3?",
    "Summarise the transformer architecture.",
]


def _percentile(data: List[float], p: int) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (k - lo) * (s[hi] - s[lo])


def benchmark_latency(
    engine,
    prompts: Optional[List[str]] = None,
    max_new_tokens: int = 128,
    warmup: int = 2,
    runs: int = 10,
) -> dict:
    prompts = prompts or SAMPLE_PROMPTS

    print(f"Warming up ({warmup} runs)…")
    for i in range(warmup):
        engine.generate(prompts[i % len(prompts)], max_new_tokens=32)

    print(f"Benchmarking latency ({runs} runs, max_tokens={max_new_tokens})…")
    lats: List[float] = []
    toks: List[int] = []

    for i in range(runs):
        p = prompts[i % len(prompts)]
        t0 = time.perf_counter()
        r = engine.generate(p, max_new_tokens=max_new_tokens)
        lats.append(time.perf_counter() - t0)
        toks.append(r["tokens_generated"])

    total_s = sum(lats)
    result = {
        "runs": runs,
        "max_new_tokens": max_new_tokens,
        "mean_ms": round(statistics.mean(lats) * 1000, 1),
        "median_ms": round(statistics.median(lats) * 1000, 1),
        "p95_ms": round(_percentile(lats, 95) * 1000, 1),
        "p99_ms": round(_percentile(lats, 99) * 1000, 1),
        "min_ms": round(min(lats) * 1000, 1),
        "max_ms": round(max(lats) * 1000, 1),
        "tokens_per_second": round(sum(toks) / total_s, 1),
        "total_tokens": sum(toks),
    }
    _print_table(result)
    return result


def benchmark_throughput(
    engine,
    batch_sizes: Optional[List[int]] = None,
    max_new_tokens: int = 64,
) -> List[dict]:
    batch_sizes = batch_sizes or [1, 2, 4, 8]
    results = []
    for bs in batch_sizes:
        prompts = (SAMPLE_PROMPTS * 8)[:bs]
        t0 = time.perf_counter()
        texts = engine.batch_generate(prompts, max_new_tokens=max_new_tokens)
        elapsed = time.perf_counter() - t0
        total_tok = sum(len(t.split()) for t in texts)
        r = {"batch_size": bs, "tokens_per_second": round(total_tok / elapsed, 1), "elapsed_s": round(elapsed, 2)}
        results.append(r)
        print(f"  batch={bs:2d} | {r['tokens_per_second']:6.1f} tok/s | {r['elapsed_s']:.2f}s")
    return results


def _print_table(r: dict) -> None:
    print("\n── Latency ──────────────────────────────────")
    for k, v in r.items():
        print(f"  {k:<22} {v}")
    print("─────────────────────────────────────────────\n")


def main():
    import yaml
    from inference.engine import InferenceEngine
    from utils.logging import setup_logging

    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--max-tokens", type=int, default=128)
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    setup_logging(config.get("logging", {}))
    engine = InferenceEngine(config)
    engine.load()

    print("\nMemory after load:")
    for g in engine.memory_stats():
        print(f"  {g['name']}: {g['allocated_gb']} GB allocated, {g['free_gb']} GB free")

    benchmark_latency(engine, max_new_tokens=args.max_tokens, runs=args.runs)
    benchmark_throughput(engine)


if __name__ == "__main__":
    main()
