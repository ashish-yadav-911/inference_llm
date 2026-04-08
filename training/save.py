import os
import torch


def save_checkpoint(model, optimizer, step, save_dir):

    os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }

    path = os.path.join(save_dir, f"checkpoint_{step}.pt")

    torch.save(checkpoint, path)


def save_final_model(model, tokenizer, save_dir):

    os.makedirs(save_dir, exist_ok=True)

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)