"""Build the execution-verified dataset.

Pipeline per task (research plan section 5.3):
  procedural task spec -> reference self-validation -> multi-teacher proposals
  -> static filter -> sandboxed execution with full side-effect accounting
  -> randomized replay across seeds -> verified positives + hard negatives
  -> SFT records, DPO preference pairs, judge-vs-execution agreement log.

Nothing becomes training data because a model said it was right; only the
executor promotes a candidate.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .config import PROCESSED_DIR, PROVIDERS, RAW_DIR, SYSTEM_PROMPT
from .spec import Candidate, TaskSpec
from .tasks import family_distribution, generate_tasks
from .teachers import Committee
from .verify import static_risk, syntax_ok, validate_task, verify

console = Console()


def _sft_record(task: TaskSpec, command: str) -> dict:
    return {
        "task_id": task.task_id,
        "template_family": task.provenance.template_family,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task.user_request},
            {"role": "assistant", "content": command},
        ],
    }


def process_task(task: TaskSpec, committee: Committee, judge: bool) -> dict:
    """Verify one task end to end. Returns a record bundle for the writers."""
    ok, why = validate_task(task)
    if not ok:
        return {"task": task, "rejected": why, "candidates": []}

    candidates: list[Candidate] = committee.propose_all(task)
    for cand in candidates:
        if not cand.command:
            cand.verdict = None
            continue
        # Static gate first: cheap rejection before we pay for a sandbox run.
        if not syntax_ok(cand.command):
            cand.verdict = verify(task, cand.command)  # records syntax_error
            continue
        cand.verdict = verify(task, cand.command)
        if judge and cand.verdict is not None:
            cand.judge_votes = committee.judge_all(task, cand.command, cand.verdict)

    return {"task": task, "rejected": None, "candidates": candidates}


def build(
    n_tasks: int,
    providers: tuple[str, ...],
    seed: int = 0,
    judge: bool = True,
    workers: int = 6,
    proposals_per_model: int = 2,
) -> dict:
    committee = Committee(keys=providers, proposals_per_model=proposals_per_model)
    tasks = generate_tasks(n_tasks, seed=seed)
    console.print(
        f"[bold]{len(tasks)}[/] task specs across "
        f"{len(family_distribution(tasks))} template families; "
        f"teachers={list(providers)}"
    )

    started = time.time()
    bundles: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_task, t, committee, judge): t for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            bundles.append(fut.result())
            if i % 5 == 0 or i == len(tasks):
                console.print(f"  verified {i}/{len(tasks)} tasks", style="dim")

    bundles.sort(key=lambda b: b["task"].task_id)
    return write_outputs(bundles, providers, seed, time.time() - started)


def write_outputs(bundles: list[dict], providers, seed: int, elapsed: float) -> dict:
    tasks_path = RAW_DIR / "tasks.jsonl"
    cand_path = RAW_DIR / "candidates.jsonl"
    sft_path = PROCESSED_DIR / "sft.jsonl"
    dpo_path = PROCESSED_DIR / "dpo.jsonl"
    judge_path = PROCESSED_DIR / "judge_agreement.jsonl"

    n_sft = n_dpo = n_verified_tasks = n_rejected = 0
    per_model = {p: {"proposed": 0, "passed": 0, "unsafe": 0} for p in providers}
    failure_categories: dict[str, int] = {}
    judge_rows: list[dict] = []

    with (
        tasks_path.open("w") as f_tasks,
        cand_path.open("w") as f_cand,
        sft_path.open("w") as f_sft,
        dpo_path.open("w") as f_dpo,
        judge_path.open("w") as f_judge,
    ):
        for bundle in bundles:
            task: TaskSpec = bundle["task"]
            f_tasks.write(task.model_dump_json() + "\n")
            if bundle["rejected"]:
                n_rejected += 1
                continue

            positives: list[Candidate] = []
            negatives: list[Candidate] = []
            for cand in bundle["candidates"]:
                f_cand.write(cand.model_dump_json() + "\n")
                if cand.source in per_model:
                    per_model[cand.source]["proposed"] += 1
                v = cand.verdict
                if v is None:
                    continue
                if v.passed:
                    positives.append(cand)
                    if cand.source in per_model:
                        per_model[cand.source]["passed"] += 1
                else:
                    negatives.append(cand)
                    failure_categories[v.failure_category] = (
                        failure_categories.get(v.failure_category, 0) + 1
                    )
                    if v.forbidden_side_effect and cand.source in per_model:
                        per_model[cand.source]["unsafe"] += 1

                for judge_key, vote in cand.judge_votes.items():
                    if "semantically_correct" in vote:
                        judge_rows.append(
                            {
                                "task_id": task.task_id,
                                "judge": judge_key,
                                "judge_model": PROVIDERS[judge_key].model,
                                "command": cand.command,
                                "execution_passed": v.passed,
                                "judge_correct": bool(vote["semantically_correct"]),
                                "judge_safe": bool(vote.get("safe_under_constraints", True)),
                                "confidence": vote.get("confidence"),
                                "failure_category": v.failure_category,
                            }
                        )

            if positives:
                n_verified_tasks += 1
                # Deduplicate identical commands; keep the shortest per unique text.
                seen: set[str] = set()
                for cand in sorted(positives, key=lambda c: len(c.command)):
                    if cand.command in seen:
                        continue
                    seen.add(cand.command)
                    f_sft.write(json.dumps(_sft_record(task, cand.command)) + "\n")
                    n_sft += 1

                # Hard negatives: plausible failures from the same task, ranked
                # lexicographically (safety > correctness > portability).
                best = min(positives, key=lambda c: len(c.command))
                ranked_negatives = sorted(
                    negatives,
                    key=lambda c: (
                        not c.verdict.forbidden_side_effect,
                        -c.verdict.postconditions_passed,
                    ),
                )
                for neg in ranked_negatives[:3]:
                    if not neg.command or neg.command == best.command:
                        continue
                    f_dpo.write(
                        json.dumps(
                            {
                                "task_id": task.task_id,
                                "template_family": task.provenance.template_family,
                                "prompt": task.user_request,
                                "chosen": best.command,
                                "rejected": neg.command,
                                "rejected_reason": neg.verdict.failure_category,
                                "rejected_unsafe": neg.verdict.forbidden_side_effect,
                            }
                        )
                        + "\n"
                    )
                    n_dpo += 1

        for row in judge_rows:
            f_judge.write(json.dumps(row) + "\n")

    n_tasks = len(bundles)
    summary = {
        "n_tasks_generated": n_tasks,
        "n_tasks_rejected_by_self_validation": n_rejected,
        "n_tasks_with_verified_solution": n_verified_tasks,
        "task_yield": round(n_verified_tasks / max(n_tasks - n_rejected, 1), 3),
        "n_sft_records": n_sft,
        "n_dpo_pairs": n_dpo,
        "per_teacher": per_model,
        "failure_categories": failure_categories,
        "judge_agreement": judge_agreement(judge_rows),
        "seed": seed,
        "providers": list(providers),
        "elapsed_s": round(elapsed, 1),
    }
    (PROCESSED_DIR / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    render_summary(summary)
    return summary


def judge_agreement(rows: list[dict]) -> dict:
    """Report B of the plan: how well each LLM judge tracks the executor."""
    out: dict[str, dict] = {}
    for row in rows:
        stats = out.setdefault(
            row["judge"], {"n": 0, "agree": 0, "false_positive": 0, "false_negative": 0}
        )
        stats["n"] += 1
        if row["judge_correct"] == row["execution_passed"]:
            stats["agree"] += 1
        elif row["judge_correct"] and not row["execution_passed"]:
            stats["false_positive"] += 1  # judge blessed a command that failed
        else:
            stats["false_negative"] += 1  # judge rejected a verified-correct command
    for stats in out.values():
        stats["agreement"] = round(stats["agree"] / stats["n"], 3) if stats["n"] else None
    return out


def render_summary(summary: dict) -> None:
    table = Table(title="Execution-verified dataset", show_lines=False)
    table.add_column("metric")
    table.add_column("value", justify="right")
    for key in (
        "n_tasks_generated",
        "n_tasks_rejected_by_self_validation",
        "n_tasks_with_verified_solution",
        "task_yield",
        "n_sft_records",
        "n_dpo_pairs",
        "elapsed_s",
    ):
        table.add_row(key, str(summary[key]))
    console.print(table)

    teacher = Table(title="Teacher yield (execution-verified)")
    teacher.add_column("teacher")
    teacher.add_column("proposed", justify="right")
    teacher.add_column("passed", justify="right")
    teacher.add_column("pass rate", justify="right")
    teacher.add_column("unsafe", justify="right")
    for name, stats in summary["per_teacher"].items():
        rate = stats["passed"] / stats["proposed"] if stats["proposed"] else 0
        teacher.add_row(
            f"{name} ({PROVIDERS[name].model})",
            str(stats["proposed"]),
            str(stats["passed"]),
            f"{rate:.2%}",
            str(stats["unsafe"]),
        )
    console.print(teacher)

    if summary["judge_agreement"]:
        judges = Table(title="Judge vs. executor (execution is ground truth)")
        judges.add_column("judge")
        judges.add_column("n", justify="right")
        judges.add_column("agreement", justify="right")
        judges.add_column("false pos", justify="right")
        judges.add_column("false neg", justify="right")
        for name, stats in summary["judge_agreement"].items():
            judges.add_row(
                name,
                str(stats["n"]),
                f"{stats['agreement']:.2%}" if stats["agreement"] is not None else "-",
                str(stats["false_positive"]),
                str(stats["false_negative"]),
            )
        console.print(judges)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the execution-verified dataset")
    ap.add_argument("n", nargs="?", type=int, default=50, help="number of task specs")
    ap.add_argument("--openai", action="store_true", help="use gpt-5.6-luna as teacher/judge")
    ap.add_argument("--anthropic", action="store_true", help="use claude-haiku-4.5 as teacher/judge")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-judge", action="store_true", help="skip the LLM judge panel")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--proposals-per-model", type=int, default=2)
    args = ap.parse_args()

    providers = tuple(
        k for k, on in (("openai", args.openai), ("anthropic", args.anthropic)) if on
    ) or ("openai", "anthropic")

    build(
        n_tasks=args.n,
        providers=providers,
        seed=args.seed,
        judge=not args.no_judge,
        workers=args.workers,
        proposals_per_model=args.proposals_per_model,
    )


if __name__ == "__main__":
    main()
