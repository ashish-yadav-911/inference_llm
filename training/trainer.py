"""
Trainer — fixed vs original:
  - gradient_accumulation_steps actually applied
  - torch.amp.autocast for mixed precision (works with both fp16 scaler and bfloat16)
  - LR scheduler step after each optimizer step
  - labels masking for padding (-100) so loss ignores pad tokens
  - step-level checkpoint saving
  - loss logging only on rank 0
"""

import torch


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        scaler,
        loader,
        local_rank: int,
        grad_accum_steps: int = 1,
        save_steps: int = 500,
        save_dir: str = "./artifacts/checkpoints",
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.loader = loader
        self.device = local_rank
        self.grad_accum_steps = grad_accum_steps
        self.save_steps = save_steps
        self.save_dir = save_dir
        self.global_step = 0

    def train_epoch(self, epoch: int = 0):
        self.model.train()
        self.optimizer.zero_grad()
        accum_loss = 0.0

        for step, batch in enumerate(self.loader):
            input_ids     = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels        = batch["labels"].to(self.device)

            # Mask padding positions so loss doesn't train on pad tokens
            labels = labels.masked_fill(labels == self.model.module.config.pad_token_id, -100)

            # Mixed precision forward pass
            with torch.amp.autocast(device_type="cuda"):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                # Scale loss by grad_accum so effective LR is independent of accum steps
                loss = outputs.loss / self.grad_accum_steps

            self.scaler.scale(loss).backward()
            accum_loss += loss.item()

            # Only step optimizer every grad_accum_steps batches
            if (step + 1) % self.grad_accum_steps == 0:
                # Unscale before gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                if self.device == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    print(f"epoch={epoch} step={self.global_step} loss={accum_loss:.4f} lr={lr:.2e}")

                accum_loss = 0.0

                # Save checkpoint at interval — rank 0 only
                if self.device == 0 and self.global_step % self.save_steps == 0:
                    from save import save_checkpoint
                    save_checkpoint(self.model, self.optimizer, self.global_step, self.save_dir)
                    print(f"  → checkpoint saved at step {self.global_step}")