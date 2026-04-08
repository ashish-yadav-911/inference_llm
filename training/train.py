"""
Main training entry point.

Fixes vs original:
  - argparse so --config CLI arg actually works (was silently ignored)
  - gradient_checkpointing_enable() to fit large models on small GPUs (~4x VRAM reduction)
  - torch.cuda.amp.GradScaler for fp16 (prevents NaN losses after ~100 steps)
  - sampler.set_epoch(epoch) so each epoch shuffles differently
  - weight_decay and warmup_steps wired to AdamW + linear scheduler
  - gradient_accumulation_steps actually implemented
  - step-level checkpoint saving at save_steps interval
  - loss logging only on rank 0 (no 4x duplicate prints)
"""

import argparse
import torch
import yaml

from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset import JsonDataset
from trainer import Trainer
from ddp import setup_ddp, cleanup_ddp
from save import save_checkpoint, save_final_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/config/qwen_train.yaml")
    args = parser.parse_args()

    local_rank = setup_ddp()

    config = yaml.safe_load(open(args.config))

    model_name = config["model"]["name"]
    dtype_str  = config["model"].get("dtype", "bfloat16")
    dtype      = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16
    use_fp16   = (dtype == torch.float16)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
    ).to(local_rank)

    # CRITICAL: reduces activation memory ~4x — essential for large models on small GPUs
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()  # required for gradient checkpointing + DDP

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    dataset = JsonDataset(
        config["data"]["dataset_path"],
        tokenizer,
        config["training"]["max_seq_len"],
    )

    sampler = DistributedSampler(dataset, shuffle=True)

    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
    )

    train_cfg = config["training"]
    grad_accum = train_cfg.get("gradient_accumulation_steps", 1)
    total_steps = (len(loader) // grad_accum) * train_cfg["epochs"]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 0.01),  # was ignored before
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=train_cfg.get("warmup_steps", 100),
        num_training_steps=total_steps,
    )

    # GradScaler for fp16: prevents gradient underflow / NaN after ~100 steps
    # bfloat16 doesn't need a scaler (wider dynamic range than fp16)
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        loader=loader,
        local_rank=local_rank,
        grad_accum_steps=grad_accum,
        save_steps=config["output"].get("save_steps", 500),
        save_dir=config["output"]["save_dir"],
    )

    for epoch in range(train_cfg["epochs"]):
        sampler.set_epoch(epoch)  # CRITICAL: different shuffle seed per epoch
        trainer.train_epoch(epoch)

    if local_rank == 0:
        save_final_model(model.module, tokenizer, config["output"]["save_dir"])
        print(f"Training complete. Model saved to {config['output']['save_dir']}")

    cleanup_ddp()


if __name__ == "__main__":
    main()