

from __future__ import annotations

import gc
import logging
import multiprocessing as mp
import os
import queue
import threading
import time
from copy import deepcopy
from typing import Iterator, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Inter-process protocol
# ---------------------------------------------------------------------------

class _Request:
    # Sent from main process to worker process.
    __slots__ = ("request_id", "method", "args", "kwargs")

    def __init__(self, request_id: int, method: str, args: tuple, kwargs: dict) -> None:
        self.request_id = request_id
        self.method = method
        self.args = args
        self.kwargs = kwargs


class _Response:
    # Sent from worker process to main process.
    __slots__ = ("request_id", "payload", "error", "is_chunk", "is_done")

    def __init__(
        self,
        request_id: int,
        payload=None,
        error: Optional[str] = None,
        is_chunk: bool = False,
        is_done: bool = True,
    ) -> None:
        self.request_id = request_id
        self.payload = payload
        self.error = error
        self.is_chunk = is_chunk
        self.is_done = is_done


# ---------------------------------------------------------------------------
# Worker process entry-point
# ---------------------------------------------------------------------------

def _worker_process(config: dict, gpu_offset: int, req_q: mp.Queue, resp_q: mp.Queue) -> None:
    # Subprocess loop: owns a GPU slice, handles requests, and writes responses.
    # Must set before importing torch / vLLM
    tp_size = config["hardware"].get("tensor_parallel_size", config["hardware"].get("num_gpus", 1))
    device_ids = ",".join(str(gpu_offset + i) for i in range(tp_size))
    os.environ["CUDA_VISIBLE_DEVICES"] = device_ids

    # Local import — happens inside subprocess
    from inference.engine import InferenceEngine  # noqa: PLC0415

    worker_cfg = deepcopy(config)
    worker_cfg["hardware"]["gpu_offset"] = gpu_offset
    worker_cfg["hardware"]["tensor_parallel_size"] = tp_size
    # dp_size is a pool-level concept; worker doesn't need it
    worker_cfg["hardware"].pop("data_parallel_size", None)

    engine = InferenceEngine(worker_cfg, gpu_offset=gpu_offset)
    engine.load()
    logger.info("Worker ready | gpu_offset=%d | gpus=%s", gpu_offset, device_ids)

    while True:
        try:
            req: _Request = req_q.get(timeout=1.0)
        except Exception:
            continue

        if req is None:  # poison pill
            break

        try:
            if req.method == "generate":
                result = engine.generate(*req.args, **req.kwargs)
                resp_q.put(_Response(req.request_id, payload=result, is_done=True))

            elif req.method == "stream":
                for chunk in engine.stream(*req.args, **req.kwargs):
                    resp_q.put(_Response(req.request_id, payload=chunk, is_chunk=True, is_done=False))
                resp_q.put(_Response(req.request_id, payload=None, is_chunk=False, is_done=True))

            elif req.method == "batch_generate":
                result = engine.batch_generate(*req.args, **req.kwargs)
                resp_q.put(_Response(req.request_id, payload=result, is_done=True))

            elif req.method == "memory_stats":
                result = engine.memory_stats()
                resp_q.put(_Response(req.request_id, payload=result, is_done=True))

            elif req.method == "apply_chat_template":
                result = engine.apply_chat_template(*req.args)
                resp_q.put(_Response(req.request_id, payload=result, is_done=True))

        except Exception as exc:  # noqa: BLE001
            resp_q.put(_Response(req.request_id, error=str(exc), is_done=True))

    engine.shutdown()
    logger.info("Worker shut down | gpu_offset=%d", gpu_offset)


# ---------------------------------------------------------------------------
# WorkerHandle — main-process view of one worker subprocess
# ---------------------------------------------------------------------------

class _WorkerHandle:
    def __init__(self, worker_id: int, process: mp.Process, req_q: mp.Queue) -> None:
        self.worker_id = worker_id
        self.process = process
        self.req_q = req_q
        self._active_requests: int = 0
        self._lock = threading.Lock()

    @property
    def active_requests(self) -> int:
        with self._lock:
            return self._active_requests

    def increment(self) -> None:
        with self._lock:
            self._active_requests += 1

    def decrement(self) -> None:
        with self._lock:
            self._active_requests = max(0, self._active_requests - 1)

    def send(self, req: _Request) -> None:
        self.req_q.put(req)

    def is_alive(self) -> bool:
        return self.process.is_alive()


# ---------------------------------------------------------------------------
# WorkerPool
# ---------------------------------------------------------------------------

class WorkerPool:
    # Manages data-parallel worker subprocesses with thread-safe request routing.

    def __init__(self, config: dict) -> None:
        self._config = config
        hw = config["hardware"]

        self._tp_size: int = hw.get("tensor_parallel_size", hw.get("num_gpus", 1))
        num_gpus: int = hw.get("num_gpus", self._tp_size)

        # Derive dp_size if not given
        derived_dp = max(1, num_gpus // self._tp_size)
        self._dp_size: int = hw.get("data_parallel_size", derived_dp)

        total_needed = self._tp_size * self._dp_size
        if total_needed > num_gpus:
            raise ValueError(
                f"tensor_parallel_size ({self._tp_size}) × data_parallel_size ({self._dp_size}) "
                f"= {total_needed} GPUs required, but num_gpus={num_gpus}"
            )

        self._schedule: str = hw.get("schedule", "round_robin")
        self._workers: List[_WorkerHandle] = []

        # Single shared response queue — all workers send back here
        self._resp_q: mp.Queue = mp.Queue()

        # Maps request_id → threading.Event + result slot
        self._pending: dict = {}
        self._pending_lock = threading.Lock()

        # Round-robin state
        self._rr_index: int = 0
        self._rr_lock = threading.Lock()

        # Auto-incrementing request id
        self._req_counter: int = 0
        self._req_lock = threading.Lock()

        # Response dispatcher thread (started with pool)
        self._dispatcher: Optional[threading.Thread] = None
        self._running = False

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: dict) -> "WorkerPool":
        return cls(config)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        # Spawn all workers and start the response dispatcher thread.
        if self._workers:
            raise RuntimeError("WorkerPool already started")

        logger.info(
            "Starting WorkerPool | tp=%d | dp=%d | total_gpus=%d | schedule=%s",
            self._tp_size, self._dp_size, self._tp_size * self._dp_size, self._schedule,
        )

        ctx = mp.get_context("spawn")  # spawn is safer with CUDA

        for worker_id in range(self._dp_size):
            gpu_offset = worker_id * self._tp_size
            req_q: mp.Queue = ctx.Queue()

            proc = ctx.Process(
                target=_worker_process,
                args=(self._config, gpu_offset, req_q, self._resp_q),
                daemon=True,
                name=f"llm-worker-{worker_id}",
            )
            proc.start()
            self._workers.append(_WorkerHandle(worker_id, proc, req_q))
            logger.info(
                "Spawned worker %d | pid=%d | GPUs [%d, %d)",
                worker_id, proc.pid, gpu_offset, gpu_offset + self._tp_size,
            )

        self._running = True
        self._dispatcher = threading.Thread(target=self._dispatch_responses, daemon=True, name="resp-dispatcher")
        self._dispatcher.start()

    def shutdown(self) -> None:
        # Stop dispatcher, signal workers to exit, and join processes.
        self._running = False
        for w in self._workers:
            try:
                w.req_q.put(None)  # poison pill
            except Exception:
                pass
        for w in self._workers:
            w.process.join(timeout=30)
            if w.process.is_alive():
                logger.warning("Worker %d did not stop cleanly; killing.", w.worker_id)
                w.process.kill()
        if self._dispatcher:
            self._dispatcher.join(timeout=5)
        logger.info("WorkerPool shut down")

    # ------------------------------------------------------------------
    # Response dispatcher (runs in its own thread)
    # ------------------------------------------------------------------

    def _dispatch_responses(self) -> None:
        # Drain shared response queue and route results/chunks to pending callers.
        while self._running:
            try:
                resp: _Response = self._resp_q.get(timeout=0.5)
            except Exception:
                continue

            with self._pending_lock:
                slot = self._pending.get(resp.request_id)

            if slot is None:
                logger.warning("Received response for unknown request_id=%d", resp.request_id)
                continue

            if resp.is_chunk:
                # Streaming: push chunk into the caller's queue
                slot["chunk_q"].put(resp.payload)
            elif resp.is_done:
                if resp.error:
                    slot["error"] = resp.error
                else:
                    slot["result"] = resp.payload

                if slot.get("is_stream"):
                    slot["chunk_q"].put(None)  # sentinel — stream exhausted

                slot["event"].set()

                # Clean up worker busy counter
                worker_id = slot.get("worker_id")
                if worker_id is not None and worker_id < len(self._workers):
                    self._workers[worker_id].decrement()

    # ------------------------------------------------------------------
    # Request routing
    # ------------------------------------------------------------------

    def _next_request_id(self) -> int:
        with self._req_lock:
            self._req_counter += 1
            return self._req_counter

    def _pick_worker(self) -> _WorkerHandle:
        alive = [w for w in self._workers if w.is_alive()]
        if not alive:
            raise RuntimeError("No alive workers in pool")

        if self._schedule == "least_busy":
            return min(alive, key=lambda w: w.active_requests)

        # Default: round_robin
        with self._rr_lock:
            idx = self._rr_index % len(alive)
            self._rr_index += 1
        return alive[idx]

    def _send(self, method: str, args: tuple, kwargs: dict, is_stream: bool = False) -> tuple:
        # Pick a worker, register pending slot, and dispatch request.
        req_id = self._next_request_id()
        worker = self._pick_worker()
        worker.increment()

        slot = {
            "event": threading.Event(),
            "result": None,
            "error": None,
            "worker_id": worker.worker_id,
            "is_stream": is_stream,
        }
        if is_stream:
            slot["chunk_q"] = queue.Queue()

        with self._pending_lock:
            self._pending[req_id] = slot

        worker.send(_Request(req_id, method, args, kwargs))
        return req_id, slot, worker

    def _cleanup(self, req_id: int) -> None:
        with self._pending_lock:
            self._pending.pop(req_id, None)

    # ------------------------------------------------------------------
    # Public API — mirrors InferenceEngine
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
        timeout: float = 300.0,
    ) -> dict:
        req_id, slot, _ = self._send(
            "generate",
            (prompt,),
            dict(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                stop=stop,
            ),
        )
        try:
            if not slot["event"].wait(timeout=timeout):
                raise TimeoutError(f"generate() timed out after {timeout}s")
            if slot["error"]:
                raise RuntimeError(slot["error"])
            return slot["result"]
        finally:
            self._cleanup(req_id)

    def stream(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        timeout: float = 300.0,
    ) -> Iterator[str]:
        req_id, slot, _ = self._send(
            "stream",
            (prompt,),
            dict(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            ),
            is_stream=True,
        )
        try:
            chunk_q: queue.Queue = slot["chunk_q"]
            while True:
                try:
                    chunk = chunk_q.get(timeout=timeout)
                except queue.Empty:
                    raise TimeoutError(f"stream() timed out after {timeout}s with no new chunk")
                if chunk is None:  # stream sentinel
                    break
                yield chunk
            if slot["error"]:
                raise RuntimeError(slot["error"])
        finally:
            self._cleanup(req_id)

    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        timeout: float = 600.0,
    ) -> List[str]:
        req_id, slot, _ = self._send(
            "batch_generate",
            (prompts,),
            dict(max_new_tokens=max_new_tokens, temperature=temperature),
        )
        try:
            if not slot["event"].wait(timeout=timeout):
                raise TimeoutError(f"batch_generate() timed out after {timeout}s")
            if slot["error"]:
                raise RuntimeError(slot["error"])
            return slot["result"]
        finally:
            self._cleanup(req_id)

    def memory_stats(self) -> List[List[dict]]:
        # Collect memory stats from each worker.
        results = []
        for worker in self._workers:
            req_id = self._next_request_id()
            slot = {
                "event": threading.Event(),
                "result": None,
                "error": None,
                "worker_id": worker.worker_id,
                "is_stream": False,
            }
            with self._pending_lock:
                self._pending[req_id] = slot
            worker.send(_Request(req_id, "memory_stats", (), {}))
            slot["event"].wait(timeout=10.0)
            self._cleanup(req_id)
            results.append(slot.get("result") or [])
        return results

    def pool_stats(self) -> dict:
        # Return a quick pool snapshot without subprocess round-trips.
        return {
            "workers": [
                {
                    "worker_id": w.worker_id,
                    "pid": w.process.pid,
                    "alive": w.is_alive(),
                    "active_requests": w.active_requests,
                    "gpu_range": [
                        w.worker_id * self._tp_size,
                        w.worker_id * self._tp_size + self._tp_size - 1,
                    ],
                }
                for w in self._workers
            ],
            "tensor_parallel_size": self._tp_size,
            "data_parallel_size": self._dp_size,
            "schedule": self._schedule,
        }
