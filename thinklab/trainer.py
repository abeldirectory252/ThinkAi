"""Trainer — separate from model and inference."""
import logging
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

logger = logging.getLogger("thinklab.trainer")


class Trainer:
    """Fine-tuning trainer for any ThinkLab model."""

    def __init__(
        self,
        thinklab_model,
        lr: float = 2e-5,
        epochs: int = 3,
        batch_size: int = 4,
        gradient_accumulation: int = 4,
        warmup_steps: int = 100,
        max_grad_norm: float = 1.0,
        save_dir: str = "./checkpoints",
        freeze_vision: bool = True,
        lora_rank: Optional[int] = None,
    ):
        self.tm = thinklab_model
        self.model = thinklab_model.model
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.grad_accum = gradient_accumulation
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.freeze_vision = freeze_vision
        self.lora_rank = lora_rank

        if freeze_vision:
            for p in self.model.vision_tower.parameters():
                p.requires_grad = False
            logger.info("Vision encoder frozen")

    def train(self, dataset, eval_dataset=None):
        """
        Train on a dataset.

        dataset: iterable yielding (image, prompt, target_text) tuples
        """
        self.model.train()
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr, weight_decay=0.01,
        )

        step = 0
        for epoch in range(self.epochs):
            total_loss = 0
            for i, batch in enumerate(dataset):
                loss = self._train_step(batch)
                loss = loss / self.grad_accum
                loss.backward()

                if (i + 1) % self.grad_accum == 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    optimizer.step()
                    optimizer.zero_grad()
                    step += 1

                total_loss += loss.item()

                if step % 50 == 0:
                    logger.info("Epoch %d | Step %d | Loss: %.4f",
                                epoch, step, loss.item())

            avg = total_loss / max(i + 1, 1)
            logger.info("Epoch %d done | Avg loss: %.4f", epoch, avg)

            # Save checkpoint
            ckpt = self.save_dir / f"checkpoint_epoch{epoch}.pt"
            torch.save(self.model.state_dict(), ckpt)
            logger.info("Saved %s", ckpt)

        self.model.eval()
        return {"final_loss": avg, "total_steps": step}

    def _train_step(self, batch):
        """Single training step — override for custom losses."""
        image, prompt, target = batch
        dev = next(self.model.parameters()).device

        pv = self.tm.image_processor(image, dtype=self.tm.dtype).to(dev)
        input_ids = self.tm.tokenizer.build_paligemma_input(prompt)
        target_ids = self.tm.tokenizer.encode(target, add_bos=False, add_eos=True)
        all_ids = input_ids + target_ids

        ids_tensor = torch.tensor([all_ids], device=dev)
        out = self.model(pv, ids_tensor)
        logits = out["logits"]

        # Shift for causal LM loss (only on target tokens)
        n_prefix = len(input_ids)
        shift_logits = logits[:, n_prefix - 1:-1, :]
        shift_labels = ids_tensor[:, n_prefix:]

        loss = nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=0,
        )
        return loss
