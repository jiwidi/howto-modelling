"""Sweep the block-diffusion decoding knobs: quality vs. latency.

The whole argument for a diffusion student is that latency is governed by the
number of denoising `steps` rather than the token count. That is only a win if
quality holds up, so this sweeps (steps, block_size, max_new_tokens) and reports
executable pass@1 next to mean latency on the same held-out tasks.

Usage:
    uv run python -m howto.sweep_diffusion --candidate qwen3-0.6b-bd3lm --fold 0
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
from rich.console import Console
from rich.table import Table

from .config import REPORTS_DIR, RUNS_DIR, STUDENTS, SYSTEM_PROMPT
from .diffusion import decode_new, diffusion_generate, load_diffusion_model
from .evaluate import load_tasks, score
from .splits import make_folds

console = Console()

# (steps, block_size, max_new_tokens)
GRID = [
    (8, 16, 32),
    (16, 16, 32),
    (32, 16, 32),
    (8, 32, 64),
    (16, 32, 64),
    (32, 32, 64),
    (64, 32, 64),
    (64, 64, 64),
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Diffusion decoding sweep")
    ap.add_argument("--candidate", default="qwen3-0.6b-bd3lm")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--variant", default="sft", choices=["sft", "base"])
    args = ap.parse_args()

    student = STUDENTS[args.candidate]
    if student.decoder != "block_diffusion":
        raise SystemExit(f"{args.candidate} is not a block-diffusion model")

    tasks = load_tasks()
    folds = make_folds(tasks, n_splits=args.n_splits)
    test_tasks = [tasks[i] for i in folds[args.fold]["test"]]

    adapter = RUNS_DIR / f"{student.key}-fold{args.fold}"
    use_adapter = str(adapter) if (args.variant == "sft" and adapter.exists()) else None
    model, tok = load_diffusion_model(student.hf_id, use_adapter)

    prompts = []
    for task in test_tasks:
        text = tok.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task.user_request},
            ],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        prompts.append(tok(text, return_tensors="pt")["input_ids"])

    rows = []
    for steps, block_size, max_new in GRID:
        raw, latencies = [], []
        for ids in prompts:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                full = diffusion_generate(
                    model, tok, ids, steps=steps, max_new_tokens=max_new,
                    block_size=block_size, temperature=0.0,
                )
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)
            raw.append(decode_new(tok, full, ids.shape[1])[0])

        metrics = score(test_tasks, raw)
        row = {
            "steps": steps,
            "block_size": block_size,
            "max_new_tokens": max_new,
            "pass@1": metrics["pass@1"],
            "syntax_valid_rate": metrics["syntax_valid_rate"],
            "one_line_format_rate": metrics["one_line_format_rate"],
            "mean_ms": round(statistics.mean(latencies) * 1000, 1),
            "sample": (raw[0] or "").strip().splitlines()[:1],
        }
        rows.append(row)
        console.print(
            f"  steps={steps:>3} block={block_size:>3} max_new={max_new:>3} -> "
            f"pass@1={row['pass@1']:.0%} syntax={row['syntax_valid_rate']:.0%} "
            f"{row['mean_ms']}ms"
        )

    out = REPORTS_DIR / f"diffusion_sweep_{args.candidate}_{args.variant}.json"
    out.write_text(json.dumps(rows, indent=2))

    table = Table(title=f"Diffusion sweep — {args.candidate} ({args.variant}, fold {args.fold})")
    for col in ("steps", "block", "max_new", "pass@1", "syntax ok", "1-line", "mean ms"):
        table.add_column(col, justify="right")
    for r in rows:
        table.add_row(
            str(r["steps"]), str(r["block_size"]), str(r["max_new_tokens"]),
            f"{r['pass@1']:.0%}", f"{r['syntax_valid_rate']:.0%}",
            f"{r['one_line_format_rate']:.0%}", str(r["mean_ms"]),
        )
    console.print(table)
    console.print(f"\nsweep -> {out}")


if __name__ == "__main__":
    main()
