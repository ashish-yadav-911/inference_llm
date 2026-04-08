
import os
import torch
 
 
# ─── DDP ──────────────────────────────────────────────────────────────────────
 
def save_checkpoint(model, optimizer, step: int, save_dir: str):

    os.makedirs(save_dir, exist_ok=True)
 
    checkpoint = {
        "step":      step,
        "model":     model.state_dict(),        # plain model, no 'module.' prefix
        "optimizer": optimizer.state_dict(),
    }
 
    path = os.path.join(save_dir, f"checkpoint_{step}.pt")
    torch.save(checkpoint, path)
 
 
def save_final_model(model, tokenizer, save_dir: str):

    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
 
 
# ─── DeepSpeed ────────────────────────────────────────────────────────────────
 
def save_deepspeed_checkpoint(
    engine,
    tokenizer,
    save_dir: str,
    zero_stage: int,
    local_rank: int,
):

    os.makedirs(save_dir, exist_ok=True)
 
    if zero_stage in (1, 2):
        # Weights are full on every GPU — rank 0 can save directly
        if local_rank == 0:
            engine.module.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"[save] ZeRO-{zero_stage} model saved to {save_dir}")
 
    elif zero_stage == 3:
        # Collective gather — ALL ranks participate
        # stage3_gather_16bit_weights_on_model_save=True in the DS config
        # tells DS to assemble the full weights on rank 0 before writing.
        ckpt_dir = os.path.join(save_dir, "ds_checkpoint")
        engine.save_checkpoint(ckpt_dir, tag="final")
 
        if local_rank == 0:
            _convert_zero3_to_hf(
                ds_checkpoint_dir=ckpt_dir,
                hf_output_dir=save_dir,
                engine=engine,
                tokenizer=tokenizer,
            )
 
    else:
        raise ValueError(f"Unexpected zero_stage={zero_stage}")
 
 
def _convert_zero3_to_hf(ds_checkpoint_dir, hf_output_dir, engine, tokenizer):
    """
    Convert a ZeRO-3 DeepSpeed checkpoint to a HuggingFace model.
 
    Tries two methods in order:
      1. deepspeed.utils.zero_to_fp32  (preferred, in-process)
      2. subprocess call to zero_to_fp32.py  (fallback, DS ships this script)
    """
    import subprocess
    import sys
 
    fp32_path = os.path.join(hf_output_dir, "pytorch_model.bin")
 
    try:
        from deepspeed.utils.zero_to_fp32 import (  # noqa: PLC0415
            get_fp32_state_dict_from_zero_checkpoint,
        )
        state_dict = get_fp32_state_dict_from_zero_checkpoint(ds_checkpoint_dir)
        engine.module.load_state_dict(state_dict)
        engine.module.save_pretrained(hf_output_dir)
        tokenizer.save_pretrained(hf_output_dir)
        print(f"[save] ZeRO-3 model converted and saved to {hf_output_dir}")
 
    except Exception as e:
        print(f"[save] In-process ZeRO-3 conversion failed ({e}), trying subprocess...")
        result = subprocess.run(
            [
                sys.executable,
                "-m", "deepspeed.utils.zero_to_fp32",
                ds_checkpoint_dir,
                fp32_path,
            ],
            check=False,
        )
        if result.returncode == 0:
            print(f"[save] zero_to_fp32 script completed. Weights at {fp32_path}")
            print("[save] Load with: model.load_state_dict(torch.load(fp32_path))")
        else:
            print(
                f"[save] WARNING: ZeRO-3 conversion failed. "
                f"Raw checkpoint is at {ds_checkpoint_dir}. "
                f"Manually run: python -m deepspeed.utils.zero_to_fp32 "
                f"{ds_checkpoint_dir} {fp32_path}"
            )
 