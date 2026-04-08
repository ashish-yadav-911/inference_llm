"""
  Primary  : vLLM  (PagedAttention, AWQ/GPTQ/bitsandbytes, streaming, batching)
  Fallback : HuggingFace Transformers + bitsandbytes  (broader compatibility)

  Parallelism support:
    - Tensor Parallelism  (tp_size)  : splits one model across N GPUs  → solves SPACE
    - Pipeline Parallelism (pp_size) : splits layers across N GPUs     → alternative to TP
    - Data Parallelism               : multiple workers, each TP-sharded → solves THROUGHPUT
      → handled by WorkerPool in inference/worker_pool.py
"""

from __future__ import annotations

import gc
import logging
import os
import threading
import time
from typing import Iterator, List, Optional

import torch

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Single engine instance.  Owns `tp_size` GPUs.

    For Data Parallelism spin up multiple InferenceEngine instances via
    WorkerPool, passing `gpu_offset` so each worker maps to the right devices.

    Config additions vs. the 7B version
    ------------------------------------
    hardware:
      num_gpus: 8                     # total GPUs on the machine
      tensor_parallel_size: 4         # GPUs per model replica  (TP)
      pipeline_parallel_size: 1       # layer-pipeline stages   (PP, vLLM only)
      data_parallel_size: 2           # num replicas = num_gpus / tp_size
      gpu_offset: 0                   # first GPU index this worker owns
                                      # (set automatically by WorkerPool)
    """

    def __init__(self, config: dict, gpu_offset: int = 0) -> None:
        self._config = config
        self._model = None
        self._tokenizer = None
        self._loaded = False

        self._backend_name: str = ""
        self._model_id: str = config["model"]["path"]
        self._quantization: str = config["quantization"]["method"]
        self._device: str = config["hardware"]["device"]

        hw = config["hardware"]
        # ── Parallelism geometry ──────────────────────────────────────────
        self._tp_size: int = hw.get("tensor_parallel_size", hw.get("num_gpus", 1))
        self._pp_size: int = hw.get("pipeline_parallel_size", 1)
        # gpu_offset lets WorkerPool assign non-overlapping GPU ranges
        self._gpu_offset: int = hw.get("gpu_offset", gpu_offset)

        # Admission controller — injected after load so we have model geometry
        self.admission_controller = None

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        return self._loaded

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def quantization(self) -> str:
        return self._quantization

    @property
    def device(self) -> str:
        return self._device

    @property
    def tp_size(self) -> int:
        return self._tp_size

    @property
    def gpu_offset(self) -> int:
        return self._gpu_offset

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load model onto GPU(s).  Tries vLLM first, falls back to HF."""
        # Pin this worker to its designated GPUs before importing anything
        self._pin_cuda_devices()

        preferred = self._config.get("backend", "vllm")

        if preferred == "vllm":
            try:
                self._load_vllm()
                self._backend_name = "vllm"
            except Exception as exc:  # noqa: BLE001
                logger.warning("vLLM load failed (%s) — falling back to HF", exc)
                self._load_hf()
                self._backend_name = "hf"
        else:
            self._load_hf()
            self._backend_name = "hf"

        self._loaded = True
        logger.info(
            "Engine ready | backend=%s | model=%s | quant=%s | tp=%d | pp=%d | gpu_offset=%d",
            self._backend_name, self._model_id, self._quantization,
            self._tp_size, self._pp_size, self._gpu_offset,
        )
        self._setup_admission_controller()

    def _pin_cuda_devices(self) -> None:
        """
        Restrict CUDA visibility to the GPUs this worker owns.

        Worker 0, tp=4 → CUDA_VISIBLE_DEVICES=0,1,2,3
        Worker 1, tp=4 → CUDA_VISIBLE_DEVICES=4,5,6,7

        Must happen before any CUDA / vLLM import so the driver sees only
        the assigned devices.  Has no effect on CPU-only configs.
        """
        if self._device == "cpu":
            return
        device_ids = ",".join(
            str(self._gpu_offset + i) for i in range(self._tp_size)
        )
        # Only set if not already pinned (single-engine / testing path)
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = device_ids
            logger.debug("Pinned CUDA_VISIBLE_DEVICES=%s", device_ids)

    # -- vLLM --------------------------------------------------------------

    def _load_vllm(self) -> None:
        from vllm import LLM  # type: ignore

        quant = self._quantization
        quantization_arg = None if quant == "none" else quant

        gpu_util = self._config["hardware"].get("gpu_memory_utilization", 0.90)
        max_model_len = self._config["generation"].get("max_model_len", 4096)
        dtype = self._config["model"].get("dtype", "auto")

        logger.info(
            "Loading via vLLM | quant=%s | tensor_parallel=%d | pipeline_parallel=%d | gpu_util=%.2f",
            quantization_arg, self._tp_size, self._pp_size, gpu_util,
        )

        self._model = LLM(
            model=self._model_id,
            quantization=quantization_arg,
            dtype=dtype,
            # ── Tensor Parallelism: slice weight matrices across GPUs ──
            tensor_parallel_size=self._tp_size,
            # ── Pipeline Parallelism: partition layers into stages ──────
            pipeline_parallel_size=self._pp_size,
            gpu_memory_utilization=gpu_util,
            max_model_len=max_model_len,
            trust_remote_code=self._config["model"].get("trust_remote_code", False),
            enforce_eager=self._config.get("enforce_eager", False),
            **self._speculative_decoding_kwargs(),
        )

        self._tokenizer = self._model.get_tokenizer()

    def _speculative_decoding_kwargs(self) -> dict:
        """
        Build speculative decoding kwargs if configured.
        NOTE: speculative decoding is incompatible with pipeline_parallel_size > 1.
        vLLM will raise if both are set; we guard here and warn.
        """
        spec = self._config.get("speculative_decoding", {})
        if not spec.get("enabled", False):
            return {}

        if self._pp_size > 1:
            logger.warning(
                "Speculative decoding is incompatible with pipeline_parallel_size > 1 "
                "(pp=%d). Disabling speculative decoding.", self._pp_size
            )
            return {}

        method = spec.get("method", "ngram")
        kwargs: dict = {}

        if method == "ngram":
            kwargs["speculative_model"] = "[ngram]"
            kwargs["num_speculative_tokens"] = spec.get("num_speculative_tokens", 5)
            kwargs["ngram_prompt_lookup_max"] = spec.get("ngram_prompt_lookup_max", 4)
        elif method == "draft_model":
            draft_model = spec.get("draft_model")
            if draft_model:
                kwargs["speculative_model"] = draft_model
                kwargs["num_speculative_tokens"] = spec.get("num_speculative_tokens", 5)

        if kwargs:
            logger.info("Speculative decoding enabled: method=%s kwargs=%s", method, kwargs)
        return kwargs

    # -- HuggingFace fallback ----------------------------------------------

    def _load_hf(self) -> None:
        """
        HF fallback.  device_map='auto' handles basic tensor-parallel-like
        sharding via Accelerate.  True AllReduce TP is vLLM-only.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        hw = self._config["hardware"]
        model_cfg = self._config["model"]
        quant_cfg = self._config["quantization"]
        dtype_str = model_cfg.get("dtype", "float16")
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(dtype_str, torch.float16)

        if self._tp_size > 1:
            logger.warning(
                "HF backend does not support true Tensor Parallelism. "
                "Using device_map='auto' (Accelerate naive sharding) across %d GPUs. "
                "For real TP use vLLM.", self._tp_size
            )

        logger.info("Loading via HuggingFace Transformers | quant=%s", quant_cfg["method"])

        bnb_config = self._build_bnb_config(quant_cfg)
        device_map = "auto"  # Accelerate spreads layers across visible GPUs
        max_memory = hw.get("max_memory", None)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_id,
            trust_remote_code=model_cfg.get("trust_remote_code", False),
            use_fast=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            quantization_config=bnb_config,
            device_map=device_map,
            max_memory=max_memory,
            offload_folder=hw.get("offload_folder", "./offload_cache"),
            torch_dtype=dtype if bnb_config is None else None,
            trust_remote_code=model_cfg.get("trust_remote_code", False),
        )
        self._model.eval()
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        torch.backends.cudnn.benchmark = True

    @staticmethod
    def _build_bnb_config(quant_cfg: dict):
        method = quant_cfg.get("method", "none").lower()
        if method == "none":
            return None
        if method in ("awq", "gptq"):
            return None
        try:
            from transformers import BitsAndBytesConfig  # type: ignore
        except ImportError:
            logger.warning("bitsandbytes not installed; loading without quantization")
            return None

        if method == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
        if method == "4bit":
            compute_dtype = getattr(torch, quant_cfg.get("compute_dtype", "float16"), torch.float16)
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type=quant_cfg.get("quant_type", "nf4"),
                bnb_4bit_use_double_quant=quant_cfg.get("double_quant", True),
            )
        raise ValueError(
            f"Unknown quantization method: '{method}'. "
            "Valid: none | 8bit | 4bit | awq | gptq"
        )

    # ------------------------------------------------------------------
    # Generate
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
    ) -> dict:
        self._assert_loaded()
        if self._backend_name == "vllm":
            return self._generate_vllm(prompt, max_new_tokens, temperature, top_p, top_k, repetition_penalty, stop or [])
        return self._generate_hf(prompt, max_new_tokens, temperature, top_p, top_k, repetition_penalty)

    def _generate_vllm(self, prompt, max_new_tokens, temperature, top_p, top_k, rep_pen, stop) -> dict:
        from vllm import SamplingParams  # type: ignore

        params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=rep_pen,
            stop=stop,
        )
        t0 = time.perf_counter()
        outputs = self._model.generate([prompt], params)
        elapsed = time.perf_counter() - t0

        out = outputs[0]
        text = out.outputs[0].text
        return {
            "text": text,
            "tokens_generated": len(out.outputs[0].token_ids),
            "prompt_tokens": len(out.prompt_token_ids),
            "latency_seconds": round(elapsed, 4),
        }

    def _generate_hf(self, prompt, max_new_tokens, temperature, top_p, top_k, rep_pen) -> dict:
        inputs = self._tokenizer(prompt, return_tensors="pt")
        input_len = inputs["input_ids"].shape[1]
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(self._device != "cpu")):
                out_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=rep_pen,
                    do_sample=(temperature > 0),
                    pad_token_id=self._tokenizer.eos_token_id,
                )

        new_ids = out_ids[0][input_len:]
        self._maybe_flush_cache()
        return {
            "text": self._tokenizer.decode(new_ids, skip_special_tokens=True),
            "tokens_generated": len(new_ids),
            "prompt_tokens": input_len,
        }

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def stream(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> Iterator[str]:
        self._assert_loaded()
        if self._backend_name == "vllm":
            yield from self._stream_vllm(prompt, max_new_tokens, temperature, top_p, top_k)
        else:
            yield from self._stream_hf(prompt, max_new_tokens, temperature, top_p, top_k)

    def _stream_vllm(self, prompt, max_new_tokens, temperature, top_p, top_k) -> Iterator[str]:
        from vllm import SamplingParams  # type: ignore

        params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        prev_text = ""
        for out in self._model.generate(prompt, params, request_id=str(id(prompt))):
            new_text = out.outputs[0].text
            delta = new_text[len(prev_text):]
            if delta:
                yield delta
            prev_text = new_text

    def _stream_hf(self, prompt, max_new_tokens, temperature, top_p, top_k) -> Iterator[str]:
        from transformers import TextIteratorStreamer  # type: ignore

        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=(temperature > 0),
            pad_token_id=self._tokenizer.eos_token_id,
            streamer=streamer,
        )

        def _gen():
            with torch.no_grad():
                self._model.generate(**gen_kwargs)

        t = threading.Thread(target=_gen, daemon=True)
        t.start()
        for chunk in streamer:
            yield chunk
        t.join()
        self._maybe_flush_cache()

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------

    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> List[str]:
        self._assert_loaded()
        if self._backend_name == "vllm":
            return self._batch_vllm(prompts, max_new_tokens, temperature)
        return self._batch_hf(prompts, max_new_tokens, temperature)

    def _batch_vllm(self, prompts, max_new_tokens, temperature) -> List[str]:
        from vllm import SamplingParams  # type: ignore

        params = SamplingParams(max_tokens=max_new_tokens, temperature=temperature)
        outputs = self._model.generate(prompts, params)
        return [o.outputs[0].text for o in outputs]

    def _batch_hf(self, prompts, max_new_tokens, temperature) -> List[str]:
        self._tokenizer.padding_side = "left"
        inputs = self._tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=(temperature > 0),
                pad_token_id=self._tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        results = [
            self._tokenizer.decode(ids[input_len:], skip_special_tokens=True)
            for ids in out_ids
        ]
        self._maybe_flush_cache()
        return results

    # ------------------------------------------------------------------
    # Chat template
    # ------------------------------------------------------------------

    def apply_chat_template(self, messages: list) -> str:
        msg_dicts = [{"role": m.role, "content": m.content} for m in messages]
        try:
            return self._tokenizer.apply_chat_template(
                msg_dicts, tokenize=False, add_generation_prompt=True
            )
        except Exception:  # noqa: BLE001
            parts = []
            for m in msg_dicts:
                prefix = {"system": "System", "user": "Human", "assistant": "Assistant"}.get(
                    m["role"], m["role"].capitalize()
                )
                parts.append(f"{prefix}: {m['content']}")
            parts.append("Assistant:")
            return "\n".join(parts)

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    def memory_stats(self) -> List[dict]:
        if not torch.cuda.is_available():
            return [{"allocated_gb": 0, "reserved_gb": 0, "total_gb": 0, "free_gb": 0, "name": "cpu"}]

        stats = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            alloc = torch.cuda.memory_allocated(i) / 1e9
            res = torch.cuda.memory_reserved(i) / 1e9
            total = props.total_memory / 1e9
            stats.append({
                "name": props.name,
                "index": self._gpu_offset + i,   # physical GPU index
                "allocated_gb": round(alloc, 2),
                "reserved_gb": round(res, 2),
                "total_gb": round(total, 2),
                "free_gb": round(total - res, 2),
            })
        return stats

    def _maybe_flush_cache(self, threshold: float = 0.10) -> None:
        if not torch.cuda.is_available():
            return
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total = props.total_memory
            free = total - torch.cuda.memory_reserved(i)
            if free / total < threshold:
                logger.warning("GPU %d: low memory (%.1f%% free) — flushing cache", i, free / total * 100)
                gc.collect()
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Admission controller wiring
    # ------------------------------------------------------------------

    def _setup_admission_controller(self) -> None:
        try:
            from serving.admission import AdmissionController, geometry_from_hf_config  # noqa: PLC0415

            if self._backend_name == "vllm":
                hf_cfg = self._model.llm_engine.model_config.hf_config
            else:
                hf_cfg = self._model.config

            geometry = geometry_from_hf_config(hf_cfg)
            hw = self._config["hardware"]
            self.admission_controller = AdmissionController(
                geometry=geometry,
                fast_threshold_gb=hw.get("admission_fast_threshold_gb", 4.0),
                offload_threshold_gb=hw.get("admission_offload_threshold_gb", 1.5),
            )
            logger.info(
                "Admission controller active | layers=%d kv_heads=%d head_dim=%d",
                geometry.num_layers, geometry.num_kv_heads, geometry.head_dim,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not set up admission controller: %s", exc)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        self._model = None
        self._tokenizer = None
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _assert_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("Engine.load() has not been called")