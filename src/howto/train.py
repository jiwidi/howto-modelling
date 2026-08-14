"""LoRA SFT of the small student candidates on execution-verified data.

Stage 1 of the distillation recipe: verified sequence-level KD. The student is
trained only on complete teacher commands that the sandbox proved correct on
every replay seed, using the exact one-line output contract the deployment
runtime expects.

Training is per (candidate, fold) so that evaluation never scores a model on a
template family it was trained on.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from rich.console import Console

from .config import PROCESSED_DIR, RUNS_DIR, STUDENTS, SYSTEM_PROMPT, StudentSpec
from .splits import make_folds
from .spec import TaskSpec

console = Console()


def load_tasks() -> list[TaskSpec]:
    path = PROCESSED_DIR.parent / "raw" / "tasks.jsonl"
    if not path.exists():
        raise SystemExit("no tasks found -- run `make generate-dataset` first")
    return [TaskSpec.model_validate_json(l) for l in path.read_text().splitlines() if l.strip()]


def load_sft() -> list[dict]:
    path = PROCESSED_DIR / "sft.jsonl"
    if not path.exists():
        raise SystemExit("no SFT data -- run `make generate-dataset` first")
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def fold_families(n_splits: int) -> list[dict]:
    """Folds are defined over tasks, then applied to SFT rows by family."""
    tasks = load_tasks()
    folds = make_folds(tasks, n_splits=n_splits)
    out = []
    for fold in folds:
        out.append(
            {
                "train_families": sorted({tasks[i].provenance.template_family for i in fold["train"]}),
                "test_families": sorted({tasks[i].provenance.template_family for i in fold["test"]}),
            }
        )
    return out


def assert_gradients_flow(model, tok) -> None:
    """Fail loudly if the adapter is attached where no gradient reaches it.

    On multimodal checkpoints a plausible-looking target pattern can land on the
    vision tower, which never runs in a text-only forward pass. Training then
    "succeeds" with grad_norm 0 and saves a useless adapter, so we check for a
    real gradient path before spending the run.
    """
    device = next(model.parameters()).device
    batch = tok("echo hello world", return_tensors="pt").to(device)
    model.train()
    out = model(**batch, labels=batch["input_ids"])
    if not out.loss.requires_grad:
        raise SystemExit(
            "adapter has no gradient path: loss does not require grad. "
            "Check lora_targets -- it may be matching modules outside the "
            "language model."
        )
    out.loss.backward()

    trainable = [p for _, p in model.named_parameters() if p.requires_grad]
    live = sum(
        1 for p in trainable if p.grad is not None and p.grad.abs().sum().item() > 0
    )
    model.zero_grad(set_to_none=True)
    if live == 0:
        raise SystemExit(
            f"adapter has no gradient path: 0 of {len(trainable)} trainable "
            "tensors received a nonzero gradient. Check lora_targets."
        )
    console.print(f"  gradient check: {live}/{len(trainable)} adapter tensors live")


def train_one(
    student: StudentSpec,
    fold: int,
    n_splits: int,
    epochs: int,
    max_seq_len: int = 512,
    batch_size: int = 16,
    grad_accum: int = 1,
) -> dict:
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    if student.gated:
        console.print(
            f"[yellow]{student.hf_id} is license-gated: accept the terms on the hub "
            "and run `huggingface-cli login` first.[/]"
        )

    folds = fold_families(n_splits)
    if fold >= len(folds):
        raise SystemExit(f"fold {fold} out of range (n_splits={len(folds)})")
    train_fams = set(folds[fold]["train_families"])

    rows = [r for r in load_sft() if r["template_family"] in train_fams]
    if not rows:
        raise SystemExit(f"no training rows for fold {fold}")

    out_dir = RUNS_DIR / f"{student.key}-fold{fold}"
    console.print(
        f"[bold]{student.key}[/] fold {fold}: {len(rows)} SFT rows, "
        f"train families={sorted(train_fams)}"
    )

    tok = AutoTokenizer.from_pretrained(
        student.hf_id, trust_remote_code=student.remote_code
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = Dataset.from_list([{"messages": r["messages"]} for r in rows])

    model = AutoModelForCausalLM.from_pretrained(
        student.hf_id,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=student.remote_code,
    )
    model.config.use_cache = False

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

    cfg = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        # Commands are short, so a batch of 4 left the 5090 mostly idle. Larger
        # real batches with no accumulation keep the same effective batch while
        # cutting optimizer steps ~4x and raising GPU utilisation.
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        dataloader_num_workers=4,
        learning_rate=student.learning_rate,
        warmup_steps=2,  # trl 1.9 dropped warmup_ratio from SFTConfig
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_strategy="no",
        bf16=True,
        max_length=max_seq_len,
        report_to=[],
        seed=17,
        # Loss on the command only: prompt tokens must not train the student.
        completion_only_loss=True,
    )

    started = time.time()
    # Pass the tokenizer explicitly: otherwise TRL calls AutoProcessor, which for
    # multimodal checkpoints (Gemma 4) pulls in vision backends we do not need
    # for text-only command generation.
    trainer = SFTTrainer(  # noqa: F841 -- assert_gradients_flow runs first below
        model=model,
        args=cfg,
        train_dataset=ds,
        peft_config=peft_config,
        processing_class=tok,
    )
    assert_gradients_flow(trainer.model, tok)
    trainer.train()
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    elapsed = time.time() - started

    meta = {
        "student": student.key,
        "hf_id": student.hf_id,
        "fold": fold,
        "n_splits": n_splits,
        "n_train_rows": len(rows),
        "train_families": sorted(train_fams),
        "test_families": folds[fold]["test_families"],
        "epochs": epochs,
        "lora_r": student.lora_r,
        "learning_rate": student.learning_rate,
        "train_seconds": round(elapsed, 1),
        "final_loss": trainer.state.log_history[-1].get("train_loss")
        if trainer.state.log_history
        else None,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    console.print(f"  saved -> {out_dir} ({elapsed:.0f}s)")

    del trainer, model
    torch.cuda.empty_cache()
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="LoRA SFT for student candidates")
    ap.add_argument("--candidates", default=",".join(("qwen3-0.6b", "qwen25-coder-0.5b")))
    ap.add_argument("--folds", default="0", help="'all' or comma-separated fold indices")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    args = ap.parse_args()

    keys = [k.strip() for k in args.candidates.split(",") if k.strip()]
    unknown = [k for k in keys if k not in STUDENTS]
    if unknown:
        raise SystemExit(f"unknown candidates {unknown}; known: {sorted(STUDENTS)}")

    n_folds = len(fold_families(args.n_splits))
    folds = range(n_folds) if args.folds == "all" else [int(f) for f in args.folds.split(",")]

    results = []
    for key in keys:
        student = STUDENTS[key]
        for fold in folds:
            if student.decoder == "block_diffusion":
                from .train_diffusion import train_one_diffusion

                results.append(
                    train_one_diffusion(student, fold, args.n_splits, args.epochs)
                )
            else:
                results.append(
                    train_one(
                        student, fold, args.n_splits, args.epochs,
                        batch_size=args.batch_size, grad_accum=args.grad_accum,
                    )
                )
    # Name the summary after this invocation's candidates: several training
    # processes share the GPU in parallel and would otherwise clobber one file.
    tag = "-".join(keys)[:60] or "run"
    (RUNS_DIR / f"train_summary_{tag}.json").write_text(json.dumps(results, indent=2))
    console.print(f"[green]trained {len(results)} (candidate, fold) runs[/]")


if __name__ == "__main__":
    main()
