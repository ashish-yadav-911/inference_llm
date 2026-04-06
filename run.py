# Inference Engine Entrypoint

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str | None, overrides: dict) -> dict:
    default = Path(__file__).parent / "configs" / "default.yaml"
    path = Path(config_path) if config_path else default

    if not path.exists():
        sys.exit(f"[ERROR] Config not found: {path}")

    with open(path) as f:
        cfg = yaml.safe_load(f)

    # Apply CLI overrides — None means "not set", so skip those
    if overrides.get("model"):
        cfg["model"]["path"] = overrides["model"]
    if overrides.get("quant"):
        cfg["quantization"]["method"] = overrides["quant"]
    if overrides.get("device"):
        cfg["hardware"]["device"] = overrides["device"]
    if overrides.get("num_gpus"):
        cfg["hardware"]["num_gpus"] = overrides["num_gpus"]
    if overrides.get("backend"):
        cfg["backend"] = overrides["backend"]
    if overrides.get("port"):
        cfg["server"]["port"] = overrides["port"]
    if overrides.get("max_tokens"):
        cfg["generation"]["max_new_tokens"] = overrides["max_tokens"]
    if overrides.get("log_level"):
        cfg["logging"]["level"] = overrides["log_level"]

    return cfg


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run.py",
        description="LLM Inference Engine — efficient serving on limited GPU resources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument("--config", "-c", help="YAML config file (default: configs/default.yaml)")
    p.add_argument("--mode", choices=["api", "chat", "prompt", "quantize", "bench"],
                   default="api", help="Run mode")

    # Model / hardware overrides
    p.add_argument("--model", "-m", help="HuggingFace model id or local path")
    p.add_argument("--quant", choices=["none", "4bit", "8bit", "awq", "gptq"],
                   help="Quantization method")
    p.add_argument("--device", choices=["cuda", "cpu", "auto"])
    p.add_argument("--num-gpus", type=int, dest="num_gpus")
    p.add_argument("--backend", choices=["vllm", "hf"])

    # Generation
    p.add_argument("--prompt", help="Prompt text (prompt mode)")
    p.add_argument("--max-tokens", type=int, dest="max_tokens")

    # Server
    p.add_argument("--port", type=int)
    p.add_argument("--host")

    # Quantize mode
    p.add_argument("--output", help="Output path for quantize mode")

    # Misc
    p.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   dest="log_level")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def mode_api(engine, cfg: dict, host: str | None, port: int | None) -> None:
    import uvicorn
    from serving.server import create_app

    app = create_app(engine)
    h = host or cfg["server"].get("host", "0.0.0.0")
    port = port or cfg["server"].get("port", 8000)

    print(f"\n  LLM Inference Engine v2")
    print(f"  API:    http://{h}:{port}/docs")
    print(f"  Health: http://{h}:{port}/health\n")

    uvicorn.run(
        app,
        host=h,
        port=port,
        log_level=cfg["logging"].get("level", "info").lower(),
        timeout_keep_alive=cfg["server"].get("timeout", 120),
    )


def mode_prompt(engine, prompt: str, cfg: dict) -> None:
    engine.load()
    result = engine.generate(
        prompt,
        max_new_tokens=cfg["generation"]["max_new_tokens"],
        temperature=cfg["generation"]["temperature"],
    )
    print(f"\n{'─' * 60}")
    print(f"Prompt: {prompt}")
    print(f"{'─' * 60}")
    print(result["text"])
    print(f"{'─' * 60}")
    print(f"Tokens: {result['tokens_generated']} generated | {result['prompt_tokens']} prompt")


def mode_chat(engine, cfg: dict) -> None:
    engine.load()
    print("Interactive chat — type 'quit' to exit\n")
    history = []

    while True:
        try:
            user = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
        if user.lower() in ("quit", "exit", "q"):
            break
        if not user:
            continue

        history.append({"role": "user", "content": user})

        class _Msg:
            def __init__(self, d):
                self.role = d["role"]
                self.content = d["content"]

        prompt = engine.apply_chat_template([_Msg(m) for m in history])
        print("Assistant: ", end="", flush=True)
        response_text = ""
        for token in engine.stream(prompt, max_new_tokens=cfg["generation"]["max_new_tokens"]):
            print(token, end="", flush=True)
            response_text += token
        print()
        history.append({"role": "assistant", "content": response_text})


def mode_quantize(model: str, method: str, output: str) -> None:
    from inference.quantization import quantize_awq, quantize_gptq, quantize_bitsandbytes

    if not model:
        sys.exit("[ERROR] --model is required for quantize mode")
    if not output:
        sys.exit("[ERROR] --output is required for quantize mode")

    dispatch = {"awq": quantize_awq, "gptq": quantize_gptq,
                "4bit": quantize_bitsandbytes, "8bit": quantize_bitsandbytes}
    fn = dispatch.get(method, quantize_awq)

    if method in ("4bit", "8bit"):
        fn(model, output, method=method)
    else:
        fn(model, output)


def mode_bench(engine, cfg: dict) -> None:
    from utils.benchmarks import benchmark_latency, benchmark_throughput

    engine.load()
    print("\nMemory after load:")
    for g in engine.memory_stats():
        print(f"  {g['name']}: {g['allocated_gb']:.2f} GB allocated / {g['free_gb']:.2f} GB free")

    benchmark_latency(engine, max_new_tokens=cfg["generation"]["max_new_tokens"])
    benchmark_throughput(engine)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    overrides = {
        "model": args.model,
        "quant": args.quant,
        "device": args.device,
        "num_gpus": args.num_gpus,
        "backend": args.backend,
        "port": args.port,
        "max_tokens": args.max_tokens,
        "log_level": args.log_level,
    }

    cfg = load_config(args.config, overrides)

    from utils.logging import setup_logging
    setup_logging(cfg.get("logging", {}))

    # Set CUDA allocator config and Tensor Core paths before any torch import
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512"
    )

    if args.mode == "quantize":
        mode_quantize(
            model=args.model or cfg["model"]["path"],
            method=args.quant or cfg["quantization"]["method"],
            output=args.output or "",
        )
        return

    from inference.engine import InferenceEngine
    engine = InferenceEngine(cfg)

    if args.mode == "api":
        mode_api(engine, cfg, args.host, args.port)
    elif args.mode == "prompt":
        if not args.prompt:
            sys.exit("[ERROR] --prompt is required in prompt mode")
        mode_prompt(engine, args.prompt, cfg)
    elif args.mode == "chat":
        mode_chat(engine, cfg)
    elif args.mode == "bench":
        mode_bench(engine, cfg)
    else:
        sys.exit(f"[ERROR] Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
