"""Masked-denoising LoRA SFT for the block-diffusion students.

BD3LM/MDLM training differs from autoregressive SFT: instead of predicting the
next token, we corrupt the *response* by masking a random fraction t of its
tokens and train the model to recover them in one shot, under the same
staircase attention mask used at inference (causal across blocks, bidirectional
within a block).

Loss = (1/t) * mean cross-entropy over masked response positions. The 1/t
weighting is the standard MDLM importance weight; t is sampled away from 0 to
keep the gradient variance manageable at this batch size.

The prompt is never masked and never contributes to the loss, matching the
autoregressive `completion_only_loss=True` setting so the two tracks are
comparable.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
from rich.console import Console
from torch.utils.data import DataLoader, Dataset

from .config import RUNS_DIR, SYSTEM_PROMPT, StudentSpec
from .diffusion import DEFAULT_BLOCK_SIZE, _staircase_attention_mask, resolve_mask_id

console = Console()

T_MIN, T_MAX = 0.15, 0.95


class MaskedSFTDataset(Dataset):
    """Rows of (prompt_ids, response_ids) padded to a multiple of block_size."""

    def __init__(self, rows: list[dict], tok, block_size: int, max_len: int):
        self.examples: list[dict] = []
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        for row in rows:
            messages = row["messages"]
            prompt_msgs = [m for m in messages if m["role"] != "assistant"]
            answer = next(m["content"] for m in messages if m["role"] == "assistant")

            prompt_text = tok.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
            answer_ids = tok(answer, add_special_tokens=False)["input_ids"]
            if tok.eos_token_id is not None:
                answer_ids = answer_ids + [tok.eos_token_id]

            ids = (prompt_ids + answer_ids)[:max_len]
            n_prompt = min(len(prompt_ids), len(ids))
            # Pad up to a block boundary so the staircase mask is well formed.
            target = int(math.ceil(len(ids) / block_size) * block_size)
            attn_len = len(ids)
            ids = ids + [self.pad_id] * (target - len(ids))

            self.examples.append(
                {
                    "input_ids": torch.tensor(ids, dtype=torch.long),
                    "n_prompt": n_prompt,
                    "n_real": attn_len,
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> dict:
        return self.examples[i]


def collate(batch: list[dict], pad_id: int) -> dict:
    width = max(x["input_ids"].numel() for x in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    response_mask = torch.zeros((len(batch), width), dtype=torch.bool)
    for i, item in enumerate(batch):
        n = item["input_ids"].numel()
        ids[i, :n] = item["input_ids"]
        response_mask[i, item["n_prompt"] : item["n_real"]] = True
    return {"input_ids": ids, "response_mask": response_mask}


def train_one_diffusion(
    student: StudentSpec,
    fold: int,
    n_splits: int,
    epochs: int,
    block_size: int = DEFAULT_BLOCK_SIZE,
    max_len: int = 512,
    batch_size: int = 4,
    grad_accum: int = 4,
) -> dict:
    from peft import LoraConfig, get_peft_model

    from .diffusion import load_diffusion_model
    from .train import fold_families, load_sft

    folds = fold_families(n_splits)
    if fold >= len(folds):
        raise SystemExit(f"fold {fold} out of range (n_splits={len(folds)})")
    train_fams = set(folds[fold]["train_families"])
    rows = [r for r in load_sft() if r["template_family"] in train_fams]
    if not rows:
        raise SystemExit(f"no training rows for fold {fold}")

    out_dir = RUNS_DIR / f"{student.key}-fold{fold}"
    console.print(
        f"[bold]{student.key}[/] (block diffusion) fold {fold}: {len(rows)} rows"
    )

    model, tok = load_diffusion_model(student.hf_id)
    mask_id = resolve_mask_id(tok)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    peft_config = LoraConfig(
        r=student.lora_r,
        lora_alpha=student.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=(
            student.lora_targets
            if isinstance(student.lora_targets, str)
            else list(student.lora_targets)
        ),
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.train()
    model.config.use_cache = False

    ds = MaskedSFTDataset(rows, tok, block_size, max_len)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate(b, pad_id),
    )
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=student.learning_rate
    )

    device = model.device
    started = time.time()
    step = 0
    losses: list[float] = []
    generator = torch.Generator(device="cpu").manual_seed(17)

    for epoch in range(epochs):
        for batch in loader:
            ids = batch["input_ids"].to(device)
            resp = batch["response_mask"].to(device)

            t = torch.empty(ids.size(0), 1).uniform_(T_MIN, T_MAX, generator=generator)
            t = t.to(device)
            draw = torch.rand(ids.shape, device=device)
            masked = resp & (draw < t)
            # Guarantee at least one supervised position per row.
            for i in range(ids.size(0)):
                if resp[i].any() and not masked[i].any():
                    first = int(resp[i].nonzero()[0])
                    masked[i, first] = True

            noisy = torch.where(masked, torch.full_like(ids, mask_id), ids)
            attn, pos = _staircase_attention_mask(noisy, block_size, pad_id)
            logits = model(noisy, attention_mask=attn, position_ids=pos, use_cache=False).logits

            flat_mask = masked.view(-1)
            if not flat_mask.any():
                continue
            ce = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1))[flat_mask].float(),
                ids.view(-1)[flat_mask],
                reduction="none",
            )
            weight = (1.0 / t).expand_as(masked).reshape(-1)[flat_mask]
            loss = (ce * weight).mean() / grad_accum
            loss.backward()
            losses.append(float(loss) * grad_accum)

            step += 1
            if step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optim.step()
                optim.zero_grad(set_to_none=True)

        window = losses[-len(loader) :]
        console.print(
            f"  epoch {epoch + 1}/{epochs} loss={sum(window) / max(len(window), 1):.4f}"
        )

    elapsed = time.time() - started
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))

    meta = {
        "student": student.key,
        "hf_id": student.hf_id,
        "decoder": "block_diffusion",
        "fold": fold,
        "n_splits": n_splits,
        "n_train_rows": len(rows),
        "train_families": sorted(train_fams),
        "test_families": folds[fold]["test_families"],
        "epochs": epochs,
        "block_size": block_size,
        "lora_r": student.lora_r,
        "learning_rate": student.learning_rate,
        "train_seconds": round(elapsed, 1),
        "final_loss": round(sum(losses[-len(loader) :]) / max(len(loader), 1), 4),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    console.print(f"  saved -> {out_dir} ({elapsed:.0f}s)")

    del model
    torch.cuda.empty_cache()
    return meta
