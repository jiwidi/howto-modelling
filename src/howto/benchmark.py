"""Response-time benchmark for every candidate.

Latency is a first-class selection axis in the plan (section 10.3): the winner is
the highest executable pass@1 *subject to* latency and memory constraints. It is
also the axis where a parallel/diffusion decoder should beat an autoregressive
one, so the comparison has to be measured rather than assumed.

We report per-request wall-clock (mean/p50/p95), tokens per second, and peak GPU
memory, separating cold (first call, kernels not yet warm) from warm steady
state. Prompts come from the real task set so the generated lengths are realistic
for one-line shell commands.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table

from .config import REPORTS_DIR, RUNS_DIR, STUDENTS, SYSTEM_PROMPT, StudentSpec
from .evaluate import stop_token_ids
from .spec import TaskSpec

console = Console()


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(int(round((pct / 100) * (len(ordered) - 1))), len(ordered) - 1)
    return ordered[idx]


def load_prompts(limit: int) -> list[str]:
    path = REPORTS_DIR.parent / "data" / "raw" / "tasks.jsonl"
    if not path.exists():
        raise SystemExit("no tasks -- run `make generate-dataset` first")
    specs = [
        TaskSpec.model_validate_json(l)
        for l in path.read_text().splitlines()
        if l.strip()
    ]
    return [s.user_request for s in specs[:limit]]


def _build_inputs(tok, prompt: str, device):
    text = tok.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tok(text, return_tensors="pt").to(device)


@torch.inference_mode()
def measure(
    student: StudentSpec,
    adapter: Path | None,
    prompts: list[str],
    max_new_tokens: int = 96,
    warmup: int = 2,
    steps: int = 32,
    block_size: int = 32,
) -> dict:
    """Single-request latency: one prompt at a time, the way a CLI would call it."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.cuda.reset_peak_memory_stats()
    load_start = time.perf_counter()
    is_diffusion = student.decoder == "block_diffusion"

    if is_diffusion:
        from .diffusion import diffusion_generate, load_diffusion_model

        model, tok = load_diffusion_model(
            student.hf_id, str(adapter) if adapter else None
        )
    else:
        tok = AutoTokenizer.from_pretrained(
            str(adapter) if adapter else student.hf_id,
            trust_remote_code=student.remote_code,
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            student.hf_id,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            trust_remote_code=student.remote_code,
        )
        if adapter is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
        model.eval()
    load_seconds = time.perf_counter() - load_start

    device = model.device

    def run(prompt: str):
        """One request; returns (output_ids, prompt_length)."""
        enc = _build_inputs(tok, prompt, device)
        n_prompt = enc["input_ids"].shape[1]
        if is_diffusion:
            out = diffusion_generate(
                model, tok, enc["input_ids"], steps=steps,
                max_new_tokens=max_new_tokens, block_size=block_size, temperature=0.0,
            )
        else:
            out = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tok.pad_token_id,
                # Same stop-token fix as evaluation: without it Qwen3.5 runs to
                # the token cap every time and the latency number is fiction.
                eos_token_id=stop_token_ids(tok),
            )
        return out, n_prompt

    # Cold: the very first generation, before CUDA kernels are warm.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    run(prompts[0])
    torch.cuda.synchronize()
    cold_seconds = time.perf_counter() - t0

    for prompt in prompts[:warmup]:
        run(prompt)
    torch.cuda.synchronize()

    latencies: list[float] = []
    new_token_counts: list[int] = []
    for prompt in prompts:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out, n_prompt = run(prompt)
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t0)
        new_token_counts.append(int(out.shape[1] - n_prompt))

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    total_new = sum(new_token_counts)
    total_time = sum(latencies)

    del model
    torch.cuda.empty_cache()

    return {
        "model": student.key,
        "hf_id": student.hf_id,
        "decoder": student.decoder,
        "tuned": adapter is not None,
        "n_requests": len(latencies),
        "mean_ms": round(statistics.mean(latencies) * 1000, 1),
        "p50_ms": round(_percentile(latencies, 50) * 1000, 1),
        "p95_ms": round(_percentile(latencies, 95) * 1000, 1),
        "cold_ms": round(cold_seconds * 1000, 1),
        "load_seconds": round(load_seconds, 1),
        "mean_new_tokens": round(total_new / len(latencies), 1),
        "tokens_per_second": round(total_new / total_time, 1) if total_time else 0.0,
        "peak_gpu_gb": round(peak_gb, 2),
        "max_new_tokens": max_new_tokens,
        "diffusion_steps": steps if is_diffusion else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Response-time benchmark")
    ap.add_argument("--candidates", default="qwen3-0.6b,qwen25-coder-0.5b")
    ap.add_argument("--fold", type=int, default=0, help="which adapter to load")
    ap.add_argument("--n-prompts", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--skip-base", action="store_true")
    ap.add_argument("--steps", type=int, default=32, help="diffusion denoising steps")
    ap.add_argument("--block-size", type=int, default=32, help="diffusion block size")
    args = ap.parse_args()

    prompts = load_prompts(args.n_prompts)
    rows: list[dict] = []

    for key in [k.strip() for k in args.candidates.split(",") if k.strip()]:
        if key not in STUDENTS:
            console.print(f"[yellow]unknown candidate {key}, skipping[/]")
            continue
        student = STUDENTS[key]
        adapter = RUNS_DIR / f"{student.key}-fold{args.fold}"
        variants: list[tuple[str, Path | None]] = []
        if not args.skip_base:
            variants.append(("base", None))
        if adapter.exists():
            variants.append(("sft", adapter))

        for label, path in variants:
            try:
                row = measure(
                    student, path, prompts, args.max_new_tokens,
                    steps=args.steps, block_size=args.block_size,
                )
            except Exception as exc:
                console.print(f"[red]{key}-{label} failed: {type(exc).__name__}: {exc}[/]")
                continue
            row["variant"] = label
            rows.append(row)
            console.print(
                f"  {key}-{label}: mean={row['mean_ms']}ms p95={row['p95_ms']}ms "
                f"{row['tokens_per_second']} tok/s peak={row['peak_gpu_gb']}GB"
            )

    out = REPORTS_DIR / "latency.json"
    out.write_text(json.dumps(rows, indent=2))
    render(rows)
    console.print(f"\nlatency report -> {out}")


def render(rows: list[dict]) -> None:
    table = Table(title="Response time (single request, bf16, RTX 5090)")
    for col in (
        "model", "variant", "decoder", "mean ms", "p50 ms", "p95 ms",
        "cold ms", "tok/s", "new tok", "peak GB", "load s",
    ):
        table.add_column(col, justify="right" if col != "model" else "left")
    for r in sorted(rows, key=lambda r: r["mean_ms"]):
        table.add_row(
            r["model"], r["variant"], r["decoder"],
            str(r["mean_ms"]), str(r["p50_ms"]), str(r["p95_ms"]),
            str(r["cold_ms"]), str(r["tokens_per_second"]),
            str(r["mean_new_tokens"]), str(r["peak_gpu_gb"]), str(r["load_seconds"]),
        )
    console.print(table)


if __name__ == "__main__":
    main()
