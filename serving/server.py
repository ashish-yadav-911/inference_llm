"""
FastAPI inference gateway.

  POST /generate              blocking completion
  GET  /health                liveness + GPU stats
  GET  /metrics               aggregate perf counters
  
  Also:
  
  POST /stream                SSE token-by-token
  POST /batch                 batch completions
  POST /v1/completions        OpenAI Completion API
  POST /v1/chat/completions   OpenAI Chat API

"""

from __future__ import annotations

import asyncio
import bisect
import statistics
import time
import threading
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

# All of these are defined in serving/schemas.py — checked line by line
from serving.schemas import (
    AdmissionVerdict,
    BatchRequest,
    BatchResponse,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatChoice,
    ChatMessage,
    CompletionChoice,
    CompletionUsage,
    GenerateRequest,
    GenerateResponse,
    GPUStats,
    HealthResponse,
    MetricsResponse,
    OpenAICompletionRequest,
    OpenAICompletionResponse,
    StreamChunk,
)

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Server state
# ---------------------------------------------------------------------------

_engine = None          # set by create_app()
_start_time: float = 0.0
_VERSION = "2.0.0"

# Metrics
_total_requests: int = 0
_total_tokens: int = 0
_total_prompt_tokens: int = 0
_rejected_oom: int = 0
_rejected_quota: int = 0
_latencies: list[float] = []          # kept sorted for percentile queries
_latency_lock = threading.Lock()


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(engine) -> FastAPI:
    """
    Wire the engine into a FastAPI application.

    Call this once at startup:
        app = create_app(engine)
        uvicorn.run(app, host="0.0.0.0", port=8000)
    """
    global _engine, _start_time
    _engine = engine
    _start_time = time.time()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Loading model…")
        _engine.load()
        logger.info("Model ready — serving requests")
        yield
        logger.info("Shutting down")
        _engine.shutdown()

    app = FastAPI(
        title="LLM Inference Engine",
        version=_VERSION,
        description=(
            "Production inference API. "
            "Primary backend: vLLM + AWQ. "
            "Fallback: HuggingFace Transformers + bitsandbytes."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["POST", "GET"],
        allow_headers=["*"],
    )

    # Routes
    app.post("/generate",            response_model=GenerateResponse)(generate)
    app.post("/stream")(stream_sse)
    app.post("/batch",               response_model=BatchResponse)(batch)
    app.post("/v1/completions",      response_model=OpenAICompletionResponse)(openai_completions)
    app.post("/v1/chat/completions", response_model=ChatCompletionResponse)(chat_completions)
    app.get("/health",               response_model=HealthResponse)(health)
    app.get("/metrics",              response_model=MetricsResponse)(metrics)

    return app


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------

async def generate(req: GenerateRequest) -> GenerateResponse:
    _require_engine()
    _check_admission(req)

    t0 = time.perf_counter()
    try:
        result = await _run_in_thread(
            _engine.generate,
            prompt=req.prompt,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
            stop=req.stop,
        )
    except MemoryError as exc:
        _bump_rejected_oom()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GPU out of memory. Reduce max_tokens or try again shortly.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    elapsed = time.perf_counter() - t0
    n_gen = result["tokens_generated"]
    n_prompt = result["prompt_tokens"]
    _record(elapsed, n_gen, n_prompt)

    return GenerateResponse(
        request_id=req.request_id,
        text=result["text"],
        tokens_generated=n_gen,
        prompt_tokens=n_prompt,
        latency_seconds=round(elapsed, 4),
        tokens_per_second=round(n_gen / elapsed, 1) if elapsed > 0 else 0.0,
        backend=_engine.backend_name,
        model=_engine.model_id,
    )


async def stream_sse(req: GenerateRequest) -> StreamingResponse:
    _require_engine()
    _check_admission(req)

    async def event_gen() -> AsyncGenerator[str, None]:
        loop = asyncio.get_event_loop()
        q: asyncio.Queue = asyncio.Queue(maxsize=128)

        def _producer():
            try:
                idx = 0
                for token in _engine.stream(
                    prompt=req.prompt,
                    max_new_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    top_k=req.top_k,
                ):
                    chunk = StreamChunk(token=token, index=idx).model_dump_json()
                    loop.call_soon_threadsafe(q.put_nowait, chunk)
                    idx += 1
                done = StreamChunk(token="", index=idx, done=True).model_dump_json()
                loop.call_soon_threadsafe(q.put_nowait, done)
            except Exception as exc:  # noqa: BLE001
                loop.call_soon_threadsafe(q.put_nowait, f"ERROR:{exc}")
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        threading.Thread(target=_producer, daemon=True).start()

        while True:
            item = await q.get()
            if item is None:
                break
            if isinstance(item, str) and item.startswith("ERROR:"):
                yield f"data: {item}\n\n"
                break
            yield f"data: {item}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def batch(req: BatchRequest) -> BatchResponse:
    _require_engine()
    t0 = time.perf_counter()
    try:
        texts = await _run_in_thread(
            _engine.batch_generate,
            prompts=req.prompts,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
        )
    except MemoryError as exc:
        _bump_rejected_oom()
        raise HTTPException(status_code=503, detail="GPU OOM during batch generation") from exc

    elapsed = time.perf_counter() - t0
    total_tok = sum(len(t.split()) for t in texts)
    return BatchResponse(
        texts=texts,
        count=len(texts),
        latency_seconds=round(elapsed, 4),
        tokens_per_second=round(total_tok / elapsed, 1) if elapsed > 0 else 0.0,
    )


async def openai_completions(req: OpenAICompletionRequest) -> OpenAICompletionResponse:
    """OpenAI /v1/completions — drop-in replacement."""
    _require_engine()
    gen_req = GenerateRequest(
        prompt=req.prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        stop=req.stop or [],
    )
    resp = await generate(gen_req)
    return OpenAICompletionResponse(
        model=req.model,
        choices=[CompletionChoice(text=resp.text, finish_reason="stop")],
        usage=CompletionUsage(
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.tokens_generated,
            total_tokens=resp.prompt_tokens + resp.tokens_generated,
        ),
    )


async def chat_completions(req: ChatCompletionRequest) -> ChatCompletionResponse:
    """OpenAI /v1/chat/completions — applies chat template then generates."""
    _require_engine()
    # Format messages into a prompt using the engine's chat template
    prompt = _engine.apply_chat_template(req.messages)
    gen_req = GenerateRequest(
        prompt=prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
    )
    resp = await generate(gen_req)
    return ChatCompletionResponse(
        model=req.model,
        choices=[ChatChoice(message=ChatMessage(role="assistant", content=resp.text))],
        usage=CompletionUsage(
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.tokens_generated,
            total_tokens=resp.prompt_tokens + resp.tokens_generated,
        ),
    )


async def health() -> HealthResponse:
    _require_engine()
    mem = _engine.memory_stats()
    return HealthResponse(
        status="ok" if _engine.is_ready else "starting",
        backend=_engine.backend_name,
        model=_engine.model_id,
        quantization=_engine.quantization,
        device=_engine.device,
        gpu_count=len(mem),
        gpu_allocated_gb=sum(g["allocated_gb"] for g in mem),
        gpu_reserved_gb=sum(g["reserved_gb"] for g in mem),
        gpu_free_gb=sum(g["free_gb"] for g in mem),
        uptime_seconds=round(time.time() - _start_time, 1),
    )


async def metrics() -> MetricsResponse:
    _require_engine()
    with _latency_lock:
        lats = list(_latencies)

    avg_lat = statistics.mean(lats) if lats else 0.0
    p50 = _percentile(lats, 50) if lats else 0.0
    p95 = _percentile(lats, 95) if lats else 0.0
    p99 = _percentile(lats, 99) if lats else 0.0
    tps = _total_tokens / sum(lats) if lats else 0.0

    mem = _engine.memory_stats()
    gpus = [
        GPUStats(
            index=i,
            name=g.get("name", f"GPU {i}"),
            allocated_gb=g["allocated_gb"],
            reserved_gb=g["reserved_gb"],
            total_gb=g["total_gb"],
            free_gb=g["free_gb"],
            utilization_pct=g.get("utilization_pct"),
        )
        for i, g in enumerate(mem)
    ]

    return MetricsResponse(
        total_requests=_total_requests,
        total_tokens_generated=_total_tokens,
        total_prompt_tokens=_total_prompt_tokens,
        avg_latency_seconds=round(avg_lat, 4),
        avg_tokens_per_second=round(tps, 1),
        p50_latency_seconds=round(p50, 4),
        p95_latency_seconds=round(p95, 4),
        p99_latency_seconds=round(p99, 4),
        requests_rejected_oom=_rejected_oom,
        requests_rejected_quota=_rejected_quota,
        uptime_seconds=round(time.time() - _start_time, 1),
        gpus=gpus,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_engine() -> None:
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialised")


def _check_admission(req: GenerateRequest) -> None:
    """
    Ask the admission controller before dispatching.
    Raises HTTP 503 immediately if the request would likely OOM.
    """
    controller = getattr(_engine, "admission_controller", None)
    if controller is None:
        return  # No controller configured — pass through

    free_gb = sum(g["free_gb"] for g in _engine.memory_stats())
    # Rough token count for admission estimate (real count done by tokenizer inside engine)
    approx_prompt_tokens = len(req.prompt.split()) * 4 // 3
    verdict = controller.evaluate(approx_prompt_tokens, req.max_tokens, free_gb)

    if not verdict["allowed"]:
        global _rejected_quota
        _rejected_quota += 1
        raise HTTPException(
            status_code=503,
            detail=f"Request rejected by admission controller: {verdict['reason']}",
            headers={"X-Rejection-Reason": verdict["reason"]},
        )

    logger.debug("Admission: backend=%s reason=%s", verdict["backend"], verdict["reason"])


async def _run_in_thread(fn, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: fn(**kwargs))


def _record(latency: float, n_gen: int, n_prompt: int) -> None:
    global _total_requests, _total_tokens, _total_prompt_tokens
    _total_requests += 1
    _total_tokens += n_gen
    _total_prompt_tokens += n_prompt
    with _latency_lock:
        bisect.insort(_latencies, latency)
        if len(_latencies) > 10_000:
            _latencies.pop(0)


def _bump_rejected_oom() -> None:
    global _rejected_oom
    _rejected_oom += 1


def _percentile(sorted_data: list[float], p: int) -> float:
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(sorted_data) - 1)
    return sorted_data[lo] + (k - lo) * (sorted_data[hi] - sorted_data[lo])
