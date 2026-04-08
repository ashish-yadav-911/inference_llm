
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
        backend: str = "ddp",           # "ddp" or "deepspeed"
    ):
        self.model            = model
        self.optimizer        = optimizer
        self.scheduler        = scheduler
        self.scaler           = scaler   # None for DeepSpeed
        self.loader           = loader
        self.device           = local_rank
        self.grad_accum_steps = grad_accum_steps
        self.save_steps       = save_steps
        self.save_dir         = save_dir
        self.backend          = backend
        self.global_step      = 0
        self.is_deepspeed     = (backend == "deepspeed")
 
    # ── helpers ───────────────────────────────────────────────────────────────
 
    def _get_pad_token_id(self):
        """
        Safely get pad_token_id from the model config.
        Works for both DDP (model.module.config) and DeepSpeed (model.module.config).
        Returns None if pad_token_id is not set — caller must handle this.
        """
        try:
            inner = self.model.module
        except AttributeError:
            inner = self.model
        return getattr(inner.config, "pad_token_id", None)
 
    def _current_lr(self) -> float:
        """Get current learning rate regardless of backend."""
        if self.is_deepspeed:
            # DS engine exposes get_lr() directly
            return self.model.get_lr()[0]
        return self.scheduler.get_last_lr()[0]
 
    # ── main training loop ────────────────────────────────────────────────────
 
    def train_epoch(self, epoch: int = 0):
        self.model.train()
 
        # DDP: zero_grad upfront, accumulate gradients over N steps
        # DeepSpeed: DS manages gradient zeroing internally
        if not self.is_deepspeed:
            self.optimizer.zero_grad()
 
        accum_loss = 0.0
        pad_token_id = self._get_pad_token_id()
 
        for step, batch in enumerate(self.loader):
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels         = batch["labels"].to(self.device)
 
            # Mask any remaining padding positions in labels with -100 so
            # CrossEntropyLoss ignores them.
            # The collator already pads labels with -100, but this guards
            # against edge cases where pad_token_id slips through.
            if pad_token_id is not None:
                labels = labels.masked_fill(labels == pad_token_id, -100)
 
            # ── Forward pass ──────────────────────────────────────────
            if self.is_deepspeed:
                # DeepSpeed manages autocast internally based on its bf16/fp16 config
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / self.grad_accum_steps
 
                # engine.backward() handles:
                #   - scaled backward (bf16/fp16)
                #   - ZeRO reduce-scatter of gradients across ranks
                self.model.backward(loss)
 
            else:
                # DDP: manual autocast + GradScaler
                with torch.amp.autocast(device_type="cuda"):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss / self.grad_accum_steps
 
                self.scaler.scale(loss).backward()
 
            accum_loss += loss.item()
 
            # ── Optimizer step (every grad_accum_steps micro-batches) ─
            if (step + 1) % self.grad_accum_steps == 0:
 
                if self.is_deepspeed:
                    # engine.step() handles:
                    #   - gradient unscaling
                    #   - gradient clipping (from DS config)
                    #   - ZeRO all-gather of parameters (ZeRO-3)
                    #   - optimizer update
                    #   - scheduler step
                    self.model.step()
 
                else:
                    # DDP: manual unscale → clip → optimizer step → scheduler
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
 
                self.global_step += 1
 
                # ── Logging (rank 0 only) ──────────────────────────────
                if self.device == 0:
                    lr = self._current_lr()
                    print(
                        f"epoch={epoch}  step={self.global_step}"
                        f"  loss={accum_loss:.4f}  lr={lr:.2e}"
                        f"  [{self.backend}]"
                    )
 
                accum_loss = 0.0
 
                # ── Step-level checkpoint (rank 0 only) ───────────────
                if self.device == 0 and self.global_step % self.save_steps == 0:
                    if self.is_deepspeed:
                        # DS checkpoint includes ZeRO state; all ranks must
                        # call save_checkpoint() together (collective op).
                        # We call it outside the rank-0 guard below.
                        pass  # handled in the block below
                    else:
                        from save import save_checkpoint  # noqa: PLC0415
                        save_checkpoint(
                            self.model.module,   # unwrap DDP
                            self.optimizer,
                            self.global_step,
                            self.save_dir,
                        )
                        print(f"  → DDP checkpoint saved at step {self.global_step}")
 
                # DeepSpeed checkpoint must be called by ALL ranks
                if self.is_deepspeed and self.global_step % self.save_steps == 0:
                    tag = f"step_{self.global_step}"
                    self.model.save_checkpoint(self.save_dir, tag=tag)
                    if self.device == 0:
                        print(f"  → DeepSpeed checkpoint saved at step {self.global_step}")
 