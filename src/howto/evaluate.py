"""Cross-validated executable evaluation of the student candidates.

Primary metric is executable pass@1 at temperature 0 on held-out template
families, decided by the sandbox -- not by text similarity to a reference
command. Secondary metrics cover the safety and format axes the plan calls out
(section 10.3): forbidden side effects, unintended writes, syntax validity and
one-line format compliance.

Every trained model is compared against its own untuned base on identical
tasks, so the reported delta isolates the effect of the verified data.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table

from .config import PROCESSED_DIR, REPORTS_DIR, RUNS_DIR, STUDENTS, SYSTEM_PROMPT
from .spec import TaskSpec
from .splits import make_folds
from .verify import clean_command, static_risk, verify

console = Console()


def load_tasks() -> list[TaskSpec]:
    path = PROCESSED_DIR.parent / "raw" / "tasks.jsonl"
    if not path.exists():
        raise SystemExit("no tasks -- run `make generate-dataset` first")
    return [TaskSpec.model_validate_json(l) for l in path.read_text().splitlines() if l.strip()]


def _load_model(hf_id: str, adapter: Path | None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(adapter) if adapter else hf_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=torch.bfloat16, device_map="cuda:0")
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
        model = model.merge_and_unload()
    model.eval()
    return model, tok


# Turn-terminating tokens used by the chat templates we train against. A model
# can only stop on a token generate() is told about.
TURN_END_TOKENS = ("<|im_end|>", "<end_of_turn>", "<|endoftext|>", "<eos>")


def stop_token_ids(tok) -> list[int]:
    """Every token that legitimately ends an assistant turn for this tokenizer.

    Qwen3.5 ships generation_config.eos_token_id = <|endoftext|> while its chat
    template terminates assistant turns with <|im_end|>. Training teaches the
    model to emit <|im_end|>, but generate() only knows to stop on the
    generation_config value, so the model sails straight past it and starts
    hallucinating the next turn. Collecting every plausible turn-end token fixes
    that without depending on a single field being right.
    """
    ids: list[int] = []
    if tok.eos_token_id is not None:
        ids.append(tok.eos_token_id)
    for name in TURN_END_TOKENS:
        tid = tok.convert_tokens_to_ids(name)
        if isinstance(tid, int) and tid >= 0 and tid != tok.unk_token_id:
            ids.append(tid)
    return sorted(set(ids))


@torch.inference_mode()
def generate(model, tok, tasks: list[TaskSpec], batch_size: int = 32) -> list[str]:
    outputs: list[str] = []
    tok.padding_side = "left"
    for start in range(0, len(tasks), batch_size):
        chunk = tasks[start : start + batch_size]
        prompts = [
            tok.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": t.user_request},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for t in chunk
        ]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(
            **enc,
            max_new_tokens=96,
            do_sample=False,  # temperature 0: pass@1 must be deterministic
            eos_token_id=stop_token_ids(tok),
            pad_token_id=tok.pad_token_id,
        )
        for i in range(len(chunk)):
            new_tokens = gen[i][enc["input_ids"].shape[1] :]
            outputs.append(tok.decode(new_tokens, skip_special_tokens=True))
    return outputs


def score(tasks: list[TaskSpec], raw_outputs: list[str], workers: int = 12) -> dict:
    """Execute every generation and aggregate the plan's primary metrics.

    Verification dominates evaluation wall-clock: every candidate runs in its
    own sandbox. Those runs are independent and spend their time in subprocesses
    rather than holding the GIL, so a thread pool scales them across cores.
    """
    n = len(tasks)
    passed = unsafe = syntax_bad = unintended_write = one_line = 0
    failures: list[dict] = []

    commands = [clean_command(raw) for raw in raw_outputs]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        verdicts = list(pool.map(verify, tasks, commands))

    for task, raw, command, v in zip(tasks, raw_outputs, commands, verdicts):
        if command and "\n" not in raw.strip():
            one_line += 1
        if not v.syntax_valid:
            syntax_bad += 1
        if v.forbidden_side_effect:
            unsafe += 1
        if v.passed:
            passed += 1
        else:
            first = v.results[0] if v.results else None
            if task.oracle.filesystem_unchanged and first and first.filesystem_changed:
                unintended_write += 1
            failures.append(
                {
                    "task_id": task.task_id,
                    "template_family": task.provenance.template_family,
                    "request": task.user_request,
                    "generated": command,
                    "reference": task.accepted_commands[0],
                    "failure_category": v.failure_category,
                    "postconditions_passed": round(v.postconditions_passed, 3),
                    "static_risk": static_risk(command) if command else "empty",
                }
            )

    return {
        "n": n,
        "pass@1": round(passed / n, 4) if n else 0.0,
        "forbidden_side_effect_rate": round(unsafe / n, 4) if n else 0.0,
        "unintended_write_rate": round(unintended_write / n, 4) if n else 0.0,
        "syntax_valid_rate": round((n - syntax_bad) / n, 4) if n else 0.0,
        "one_line_format_rate": round(one_line / n, 4) if n else 0.0,
        "failures": failures,
    }


@torch.inference_mode()
def generate_diffusion(model, tok, tasks: list[TaskSpec], steps: int, block_size: int,
                       max_new_tokens: int = 64) -> list[str]:
    """Block-diffusion decoding: the whole block is denoised in `steps` passes."""
    from .diffusion import decode_new, diffusion_generate

    outputs: list[str] = []
    for task in tasks:
        text = tok.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task.user_request},
            ],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        ids = tok(text, return_tensors="pt")["input_ids"]
        full = diffusion_generate(
            model, tok, ids, steps=steps, max_new_tokens=max_new_tokens,
            block_size=block_size, temperature=0.0,
        )
        outputs.append(decode_new(tok, full, ids.shape[1])[0])
    return outputs


def evaluate_model(
    label: str,
    hf_id: str,
    adapter: Path | None,
    folds_tasks: list[tuple[int, list[TaskSpec]]],
    decoder: str = "autoregressive",
    steps: int = 32,
    block_size: int = 32,
) -> list[dict]:
    if decoder == "block_diffusion":
        from .diffusion import load_diffusion_model

        model, tok = load_diffusion_model(hf_id, str(adapter) if adapter else None)
    else:
        model, tok = _load_model(hf_id, adapter)
    rows = []
    try:
        for fold, tasks in folds_tasks:
            started = time.time()
            if decoder == "block_diffusion":
                raw = generate_diffusion(model, tok, tasks, steps, block_size)
            else:
                raw = generate(model, tok, tasks)
            metrics = score(tasks, raw)
            metrics.update(
                {
                    "model": label,
                    "fold": fold,
                    "tuned": adapter is not None,
                    "eval_seconds": round(time.time() - started, 1),
                    "test_families": sorted({t.provenance.template_family for t in tasks}),
                }
            )
            rows.append(metrics)
            console.print(
                f"  {label} fold {fold}: pass@1={metrics['pass@1']:.2%} "
                f"unsafe={metrics['forbidden_side_effect_rate']:.2%} "
                f"({metrics['eval_seconds']}s)"
            )
    finally:
        del model
        torch.cuda.empty_cache()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-validated executable evaluation")
    ap.add_argument("--candidates", default="qwen3-0.6b,qwen25-coder-0.5b")
    ap.add_argument("--folds", default="0", help="'all' or comma-separated fold indices")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--skip-base", action="store_true", help="skip untuned baselines")
    ap.add_argument("--steps", type=int, default=32, help="diffusion denoising steps")
    ap.add_argument("--block-size", type=int, default=32, help="diffusion block size")
    ap.add_argument(
        "--max-test", type=int, default=0,
        help="evaluate at most N held-out tasks per fold (0 = all). A fixed "
             "stride keeps the sample spread across groups rather than "
             "concentrated in the first few.",
    )
    args = ap.parse_args()

    tasks = load_tasks()
    folds = make_folds(tasks, n_splits=args.n_splits)
    wanted = range(len(folds)) if args.folds == "all" else [int(f) for f in args.folds.split(",")]

    all_rows: list[dict] = []
    for key in [k.strip() for k in args.candidates.split(",") if k.strip()]:
        student = STUDENTS[key]
        available = []
        for fold in wanted:
            adapter = RUNS_DIR / f"{student.key}-fold{fold}"
            test_tasks = [tasks[i] for i in folds[fold]["test"]]
            if args.max_test and len(test_tasks) > args.max_test:
                stride = len(test_tasks) // args.max_test
                test_tasks = test_tasks[::stride][: args.max_test]
            if adapter.exists():
                available.append((fold, test_tasks, adapter))
            else:
                console.print(f"[yellow]no adapter for {student.key} fold {fold}; skipping[/]")

        dec = dict(decoder=student.decoder, steps=args.steps, block_size=args.block_size)
        if not args.skip_base and available:
            console.print(f"[bold]{key}[/] (untuned base, {student.decoder})")
            all_rows += evaluate_model(
                f"{key}-base", student.hf_id, None,
                [(f, t) for f, t, _ in available], **dec,
            )

        for fold, test_tasks, adapter in available:
            console.print(f"[bold]{key}[/] (LoRA fold {fold}, {student.decoder})")
            all_rows += evaluate_model(
                f"{key}-sft", student.hf_id, adapter, [(fold, test_tasks)], **dec
            )

    report_path = REPORTS_DIR / "evaluation.json"
    report_path.write_text(json.dumps(all_rows, indent=2))
    render(all_rows)
    console.print(f"\nfull report -> {report_path}")


def render(rows: list[dict]) -> None:
    table = Table(title="Executable evaluation (held-out template families, temp 0)")
    for col in ("model", "folds", "n", "pass@1", "unsafe", "unintended write", "syntax ok", "1-line"):
        table.add_column(col, justify="right" if col != "model" else "left")

    by_model: dict[str, list[dict]] = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row)

    for model, group in by_model.items():
        mean = lambda k: statistics.mean(r[k] for r in group)
        table.add_row(
            model,
            str(len(group)),
            str(sum(r["n"] for r in group)),
            f"{mean('pass@1'):.2%}",
            f"{mean('forbidden_side_effect_rate'):.2%}",
            f"{mean('unintended_write_rate'):.2%}",
            f"{mean('syntax_valid_rate'):.2%}",
            f"{mean('one_line_format_rate'):.2%}",
        )
    console.print(table)

    # Paired base-vs-tuned delta on identical folds, the number that matters.
    deltas = Table(title="SFT delta vs. untuned base (same folds, same tasks)")
    deltas.add_column("candidate")
    deltas.add_column("base pass@1", justify="right")
    deltas.add_column("sft pass@1", justify="right")
    deltas.add_column("delta", justify="right")
    for model in sorted({m[:-5] for m in by_model if m.endswith("-base")}):
        base = by_model.get(f"{model}-base", [])
        sft = by_model.get(f"{model}-sft", [])
        if not base or not sft:
            continue
        b = statistics.mean(r["pass@1"] for r in base)
        s = statistics.mean(r["pass@1"] for r in sft)
        deltas.add_row(model, f"{b:.2%}", f"{s:.2%}", f"{s - b:+.2%}")
    console.print(deltas)


if __name__ == "__main__":
    main()
