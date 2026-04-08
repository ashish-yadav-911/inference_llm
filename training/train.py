
import argparse
import torch
import yaml
 
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
 
from dataset import JsonDataset
from collator import DataCollatorForCausalLM
from trainer import Trainer
from ddp import setup_ddp, cleanup_ddp, setup_deepspeed, cleanup_deepspeed
from save import save_checkpoint, save_final_model, save_deepspeed_checkpoint
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/config/qwen_train.yaml")
    args = parser.parse_args()
 
    config    = yaml.safe_load(open(args.config))
    hw_cfg    = config["hardware"]
    train_cfg = config["training"]
    out_cfg   = config["output"]
 
    # ── Decide backend ────────────────────────────────────────────────────────
    backend    = hw_cfg.get("distributed_backend", "ddp").lower()
    zero_stage = hw_cfg.get("zero_stage", 2)
 
    # ── Model + tokenizer ─────────────────────────────────────────────────────
    model_name = config["model"]["name"]
    dtype_str  = config["model"].get("dtype", "bfloat16")
    dtype      = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16
    use_fp16   = (dtype == torch.float16)
 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
 
    # ── Dataset + sampler + loader ────────────────────────────────────────────
    dataset = JsonDataset(
        config["data"]["dataset_path"],
        tokenizer,
        train_cfg["max_seq_len"],
    )
 
    # DataLoader setup shared by both backends
    grad_accum = train_cfg.get("gradient_accumulation_steps", 1)
 
    # ── DDP path ──────────────────────────────────────────────────────────────
    if backend == "ddp":
        local_rank = setup_ddp()
 
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
        ).to(local_rank)
 
        # Gradient checkpointing: recomputes activations during backward
        # instead of storing them — ~4x VRAM reduction, ~30% slower.
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()  # required when using GC + DDP
 
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
 
        sampler = DistributedSampler(dataset, shuffle=True)
        collator = DataCollatorForCausalLM(
            pad_token_id=tokenizer.pad_token_id or 0
        )
        loader = DataLoader(
            dataset,
            batch_size=train_cfg["batch_size"],
            sampler=sampler,
            collate_fn=collator,
            num_workers=2,
            pin_memory=True,
        )
 
        total_steps = (len(loader) // grad_accum) * train_cfg["epochs"]
 
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_cfg["lr"],
            weight_decay=train_cfg.get("weight_decay", 0.01),
        )
 
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=train_cfg.get("warmup_steps", 100),
            num_training_steps=total_steps,
        )
 
        # GradScaler only needed for fp16; bf16 has wide enough dynamic range
        scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
 
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            loader=loader,
            local_rank=local_rank,
            grad_accum_steps=grad_accum,
            save_steps=out_cfg.get("save_steps", 500),
            save_dir=out_cfg["save_dir"],
            backend="ddp",
        )
 
        for epoch in range(train_cfg["epochs"]):
            sampler.set_epoch(epoch)
            trainer.train_epoch(epoch)
 
        if local_rank == 0:
            # Fix: use model.module to unwrap DDP before saving
            save_final_model(model.module, tokenizer, out_cfg["save_dir"])
            print(f"Training complete. Model saved to {out_cfg['save_dir']}")
 
        cleanup_ddp()
 
    # ── DeepSpeed path ────────────────────────────────────────────────────────
    elif backend == "deepspeed":
        # DeepSpeed handles its own distributed init — do NOT call
        # dist.init_process_group before deepspeed.initialize().
        # LOCAL_RANK is still set by torchrun.
        local_rank = int(__import__("os").environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
 
        # Load model to CPU first — ZeRO-3 will partition it across GPUs
        # during deepspeed.initialize(). Loading directly to GPU would OOM
        # on models that don't fit on a single GPU.
        if zero_stage == 3:
            # ZeRO-3: load to CPU, DS partitions during init
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
            )
        else:
            # ZeRO-1/2: each rank loads to its own GPU (params still replicated)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
            ).to(local_rank)
 
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
 
        # Build a base optimizer — DS will wrap it internally with ZeRO hooks
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_cfg["lr"],
            weight_decay=train_cfg.get("weight_decay", 0.01),
        )
 
        sampler = DistributedSampler(dataset, shuffle=True)
        collator = DataCollatorForCausalLM(
            pad_token_id=tokenizer.pad_token_id or 0
        )
        loader = DataLoader(
            dataset,
            batch_size=train_cfg["batch_size"],
            sampler=sampler,
            collate_fn=collator,
            num_workers=2,
            pin_memory=True,
        )
 
        total_steps = (len(loader) // grad_accum) * train_cfg["epochs"]
 
        engine, optimizer, scheduler, local_rank = setup_deepspeed(
            model=model,
            optimizer=optimizer,
            train_cfg=train_cfg,
            zero_stage=zero_stage,
            total_steps=total_steps,
        )
 
        # engine replaces model — it exposes forward() and engine.backward()
        trainer = Trainer(
            model=engine,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,            # DeepSpeed manages mixed precision internally
            loader=loader,
            local_rank=local_rank,
            grad_accum_steps=grad_accum,
            save_steps=out_cfg.get("save_steps", 500),
            save_dir=out_cfg["save_dir"],
            backend="deepspeed",
        )
 
        for epoch in range(train_cfg["epochs"]):
            sampler.set_epoch(epoch)
            trainer.train_epoch(epoch)
 
        # ZeRO-3: weights are sharded across GPUs.
        # engine.save_checkpoint() gathers them before writing.
        # ZeRO-1/2: we can also use engine.save_checkpoint() or the
        # manual gather path — use the same code for consistency.
        save_deepspeed_checkpoint(
            engine=engine,
            tokenizer=tokenizer,
            save_dir=out_cfg["save_dir"],
            zero_stage=zero_stage,
            local_rank=local_rank,
        )
 
        cleanup_deepspeed()
 
    else:
        raise ValueError(
            f"Unknown distributed_backend: {backend!r}. "
            "Choose 'ddp' or 'deepspeed' in hardware.distributed_backend."
        )
 
 
if __name__ == "__main__":
    main()
 