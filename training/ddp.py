"""
ddp.py — distributed setup supporting two backends:

  DDP (torch.distributed)
    - Every GPU holds the FULL model copy
    - Gradients are AllReduced after every backward pass
    - Memory per GPU: full params + full gradients + full optimizer states
    - Good for: models that comfortably fit on one GPU (≤7B in bf16/fp16)

  DeepSpeed ZeRO
    - ZeRO-1: shards optimizer states only             (~4x memory vs DDP)
    - ZeRO-2: shards optimizer states + gradients      (~8x memory vs DDP)
    - ZeRO-3: shards optimizer + gradients + params    (~N_gpu x reduction)
    - Good for: large models (13B, 30B, 70B+) that don't fit per-GPU with DDP

  Quick decision guide:
    ┌─────────────────────────────────┬────────────────┬──────────────────┐
    │ Scenario                        │ Backend        │ ZeRO stage       │
    ├─────────────────────────────────┼────────────────┼──────────────────┤
    │ ≤7B model, ≥24 GB VRAM/GPU      │ DDP            │ —                │
    │ 7B–30B model, 16–24 GB VRAM/GPU │ DeepSpeed      │ ZeRO-2           │
    │ 30B–70B model, any GPU size     │ DeepSpeed      │ ZeRO-3           │
    │ 70B+ model, <40 GB VRAM/GPU     │ DeepSpeed      │ ZeRO-3 + offload │
    └─────────────────────────────────┴────────────────┴──────────────────┘

  How to select backend:
    In your YAML config set:
      hardware:
        distributed_backend: ddp        # or: deepspeed
        zero_stage: 2                   # only used when backend = deepspeed
"""

import os
import torch
import torch.distributed as dist


# ─── DDP ──────────────────────────────────────────────────────────────────────

def setup_ddp():
    """
    Initialise vanilla PyTorch DDP via NCCL.
    Expects torchrun to have set LOCAL_RANK in the environment.
    Returns local_rank (int).
    """
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    """Tear down the NCCL process group after training."""
    dist.destroy_process_group()


# ─── DeepSpeed config builder ─────────────────────────────────────────────────

def build_deepspeed_config(train_cfg: dict, zero_stage: int = 2) -> dict:

    lr           = train_cfg.get("lr", 3e-5)
    weight_decay = train_cfg.get("weight_decay", 0.01)
    warmup_steps = train_cfg.get("warmup_steps", 100)
    grad_accum   = train_cfg.get("gradient_accumulation_steps", 1)
    grad_clip    = train_cfg.get("grad_clip", 1.0)

    ds_config = {
        # ── Batch sizing ──────────────────────────────────────────────
        # DeepSpeed needs micro batch size per GPU separately so it can
        # compute the true global batch size for logging/scheduling.
        "train_micro_batch_size_per_gpu": train_cfg.get("batch_size", 2),
        "gradient_accumulation_steps":    grad_accum,
        "gradient_clipping":              grad_clip,

        # ── Mixed precision ───────────────────────────────────────────
        # bf16 is strongly preferred for large models:
        #   - No loss scaling needed (wider dynamic range than fp16)
        #   - No NaN risk after ~100 steps (a common fp16 failure)
        # Only enable fp16 if your GPU doesn't support bf16 (pre-Ampere).
        "bf16": {"enabled": True},
        "fp16": {"enabled": False},

        # ── Optimizer ────────────────────────────────────────────────
        # DeepSpeed uses its own fused AdamW kernel — faster than
        # torch.optim.AdamW because it fuses the param update into one
        # CUDA kernel instead of one per tensor.
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr":           lr,
                "betas":        [0.9, 0.999],
                "eps":          1e-8,
                "weight_decay": weight_decay,
            },
        },

        # ── LR scheduler ─────────────────────────────────────────────
        # WarmupDecayLR = linear warmup then linear decay.
        # total_num_steps is patched at runtime in setup_deepspeed()
        # once we know the actual number of training steps.
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr":    0,
                "warmup_max_lr":    lr,
                "warmup_num_steps": warmup_steps,
                "total_num_steps":  1000,   # overwritten at runtime
            },
        },

        # ── ZeRO optimisation ─────────────────────────────────────────
        "zero_optimization": _zero_config(zero_stage),

        # ── Misc ──────────────────────────────────────────────────────
        "steps_per_print":      50,
        "wall_clock_breakdown": False,
    }

    return ds_config


def _zero_config(stage: int) -> dict:
    """
    Return the zero_optimization sub-dict for the requested stage.

    Memory breakdown by stage (example: 7B model, 4 GPUs, bf16):
      DDP      : ~28 GB params + ~28 GB grad + ~56 GB optimizer  = ~112 GB per GPU
      ZeRO-1   : ~28 GB params + ~28 GB grad + ~14 GB optimizer  = ~70 GB per GPU
      ZeRO-2   : ~28 GB params + ~7 GB grad  + ~14 GB optimizer  = ~49 GB per GPU
      ZeRO-3   : ~7 GB params  + ~7 GB grad  + ~14 GB optimizer  = ~28 GB per GPU
    """
    if stage == 1:
        # Only shard optimizer states (m + v buffers).
        # Simplest ZeRO mode — no extra communication patterns.
        # Good entry point if DDP almost fits but optimizer pushes you over.
        return {
            "stage": 1,
        }

    if stage == 2:
        # Shard optimizer states AND gradients.
        # During backward: gradients are reduce-scattered (each rank keeps
        # its own 1/N shard), then each rank updates its param shard.
        # Forward still reads full params from all ranks via all-gather.
        # overlap_comm=True hides comm latency behind compute.
        return {
            "stage": 2,
            "allgather_partitions":  True,
            "allgather_bucket_size": 2e8,   # 200 MB: larger = fewer round-trips
            "reduce_scatter":        True,
            "reduce_bucket_size":    2e8,
            "overlap_comm":          True,  # overlap grad comm with backward compute
            "contiguous_gradients":  True,  # pack grads in one contiguous buffer
        }

    if stage == 3:
        # Shard EVERYTHING: params + gradients + optimizer states.
        # Memory per GPU ≈ total_model / N_gpus.
        # Trade-off: every forward layer needs an all-gather of its params,
        # and every backward needs a reduce-scatter of its gradients.
        # For very large models the bandwidth cost is worth the memory saving.
        #
        # offload_optimizer / offload_param:
        #   Change "device": "none"  → "device": "cpu"  to push that
        #   data to CPU RAM. Saves VRAM at the cost of PCIe bandwidth.
        #   Use CPU offload when even ZeRO-3 doesn't fit on GPU.
        return {
            "stage": 3,
            "offload_optimizer": {
                "device":     "none",   # "cpu" to offload optimizer states to RAM
                "pin_memory": True,
            },
            "offload_param": {
                "device":     "none",   # "cpu" to offload parameters to RAM
                "pin_memory": True,
            },
            "overlap_comm":            True,
            "contiguous_gradients":    True,
            "sub_group_size":          1e9,
            "reduce_bucket_size":      "auto",
            "stage3_prefetch_bucket_size":        "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters":         1e9,
            "stage3_max_reuse_distance":          1e9,
            # Collects full fp16 weights on rank 0 before save_pretrained()
            "stage3_gather_16bit_weights_on_model_save": True,
        }

    raise ValueError(f"Invalid ZeRO stage: {stage!r}. Choose 1, 2, or 3.")


# ─── DeepSpeed init ───────────────────────────────────────────────────────────

def setup_deepspeed(model, optimizer, train_cfg: dict,
                    zero_stage: int = 2, total_steps: int = 1000):

    try:
        import deepspeed  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "DeepSpeed is not installed.\n"
            "Install it with:  pip install deepspeed\n"
            "CUDA 12.x build:  DS_BUILD_OPS=1 pip install deepspeed"
        ) from exc

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    ds_config = build_deepspeed_config(train_cfg, zero_stage=zero_stage)
    # Patch the real total_num_steps so the scheduler decays correctly
    ds_config["scheduler"]["params"]["total_num_steps"] = total_steps

    engine, ds_optimizer, _, ds_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=ds_config,
    )

    return engine, ds_optimizer, ds_scheduler, local_rank


def cleanup_deepspeed():
    """Tear down distributed state after DeepSpeed training."""
    if dist.is_initialized():
        dist.destroy_process_group()