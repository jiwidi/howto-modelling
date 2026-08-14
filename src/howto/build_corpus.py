"""Build a large, deduplicated corpus of execution-verified tasks.

Combines the two generators:
  * procedural (tasks.py)   -- airtight oracles, 10 shapes, used as the anchor
  * LLM-authored (authoring.py) -- one API call per task, huge shape diversity,
    oracle derived by execution and cross-checked by a second solution

Everything that survives goes into the same SFT format as the small pipeline, so
train.py / evaluate.py need no changes.

    uv run python -m howto.build_corpus 5000 --author gemini
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from .authoring import author_one, recipes, signature
from .config import PROCESSED_DIR, PROVIDERS, RAW_DIR, SYSTEM_PROMPT
from .spec import TaskSpec
from .tasks import generate_tasks
from .verify import validate_task

console = Console()


def _request_key(text: str) -> str:
    """Normalised request text, so two phrasings of the same ask collapse."""
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


def make_progress(enabled: bool) -> Progress:
    """Live progress display. `disable` keeps the API identical for log runs."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TextColumn("[dim]{task.fields[stats]}"),
        TextColumn("[dim]elapsed"),
        TimeElapsedColumn(),
        TextColumn("[dim]eta"),
        TimeRemainingColumn(),
        console=console,
        disable=not enabled,
        refresh_per_second=8,
    )


def procedural_anchor(
    n: int, seed: int, progress: Progress | None = None
) -> list[TaskSpec]:
    """Procedurally generated tasks, self-validated as before."""
    if n <= 0:
        return []
    specs = generate_tasks(n, seed=seed)
    kept: list[TaskSpec] = []
    bar = (
        progress.add_task("procedural", total=len(specs), stats="")
        if progress is not None
        else None
    )
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(validate_task, s): s for s in specs}
        for fut in as_completed(futures):
            ok, _ = fut.result()
            if ok:
                kept.append(futures[fut])
            if bar is not None:
                progress.advance(bar)
                progress.update(bar, stats=f"kept {len(kept)}")
    if bar is not None:
        progress.update(bar, stats=f"kept {len(kept)}", visible=True)
    return kept


def build(
    n_total: int,
    author_keys: list[str],
    seed: int = 0,
    procedural_fraction: float = 0.1,
    workers: int = 24,
    max_attempt_multiplier: float = 2.0,
    show_progress: bool = True,
) -> dict:
    started = time.time()
    n_proc = int(n_total * procedural_fraction)
    n_llm = n_total - n_proc

    models = ", ".join(PROVIDERS[k].model for k in author_keys)
    console.print(
        f"target [bold]{n_total}[/] tasks: {n_proc} procedural + {n_llm} "
        f"LLM-authored, round-robin across {len(author_keys)} families ({models})"
    )

    kept: list[TaskSpec] = []
    seen_ids: set[str] = set()
    seen_requests: set[str] = set()
    seen_sigs: Counter = Counter()
    reasons: Counter = Counter()
    # Cap how many tasks may share one structural signature, so the corpus does
    # not silently collapse into a few favourite shapes.
    sig_cap = max(5, int(n_total / 50))

    plan = recipes(int(n_llm * max_attempt_multiplier), seed=seed)
    target = n_proc + n_llm
    done = 0
    interrupted = False
    per_author: dict[str, Counter] = {k: Counter() for k in author_keys}

    # Even shares are counted in *accepted* tasks, not attempts. Acceptance
    # rates differ a lot between families (Gemini ~67%, GPT ~45%, Claude ~27%
    # on a pilot), so a fixed round-robin over attempts would hand the corpus
    # to whichever model happens to pass verification most often. Each family
    # instead gets an equal quota of kept tasks and is dispatched work until it
    # fills that quota.
    quota = {k: n_llm // len(author_keys) for k in author_keys}
    for i in range(n_llm % len(author_keys)):
        quota[author_keys[i]] += 1
    # Generous per-family attempt ceiling so one weak model cannot spin forever.
    attempt_cap = {
        k: max(int(quota[k] * max_attempt_multiplier * 2.5), 8) for k in author_keys
    }

    if len(plan) < n_llm:
        console.print(
            f"[yellow]recipe space holds only {len(plan)} distinct combinations, "
            f"below the {n_llm} LLM tasks requested. Widen DOMAINS/SKILLS in "
            "authoring.py or the run will fall short.[/]"
        )

    with make_progress(show_progress) as progress:
        for spec in procedural_anchor(n_proc, seed, progress):
            if spec.task_id in seen_ids:
                continue
            seen_ids.add(spec.task_id)
            seen_sigs[signature(spec)] += 1
            kept.append(spec)
            reasons["ok_procedural"] += 1

        # The bar tracks verified tasks kept, not API calls issued -- that is
        # the number the run is actually trying to reach, so the ETA reflects
        # the real yield rather than raw request throughput.
        bar = progress.add_task("authoring ", total=target, stats="")
        progress.update(bar, completed=len(kept))

        # Long runs are worth hours of API spend, so every accepted task is
        # appended to a checkpoint the moment it is verified. A crash or a
        # Ctrl-C then costs the in-flight calls, not the whole corpus.
        checkpoint = RAW_DIR / "tasks.partial.jsonl"
        with (
            checkpoint.open("w") as ckpt,
            ThreadPoolExecutor(max_workers=workers) as pool,
        ):
            for spec in kept:
                ckpt.write(spec.model_dump_json() + "\n")
            ckpt.flush()

            recipe_pool = iter(plan)
            futures: dict = {}

            def next_family() -> str | None:
                """Family furthest below its quota that still has budget."""
                eligible = [
                    k
                    for k in author_keys
                    if per_author[k]["kept"] + per_author[k]["inflight"] < quota[k]
                    and per_author[k]["attempts"] + per_author[k]["inflight"]
                    < attempt_cap[k]
                ]
                if not eligible:
                    return None
                return min(
                    eligible,
                    key=lambda k: (
                        per_author[k]["kept"] / max(quota[k], 1),
                        per_author[k]["attempts"],
                    ),
                )

            def dispatch() -> bool:
                key = next_family()
                if key is None:
                    return False
                recipe = next(recipe_pool, None)
                if recipe is None:
                    return False
                per_author[key]["inflight"] += 1
                futures[pool.submit(author_one, PROVIDERS[key], recipe)] = key
                return True

            for _ in range(workers * 2):
                if not dispatch():
                    break

            try:
                while futures:
                    finished, _ = wait(
                        list(futures), return_when=FIRST_COMPLETED, timeout=60
                    )
                    if not finished:
                        continue
                    for fut in finished:
                        who = futures.pop(fut)
                        per_author[who]["inflight"] -= 1
                        done += 1
                        task, reason = fut.result()
                        reasons[reason] += 1
                        per_author[who]["attempts"] += 1
                        if task is None:
                            pass
                        elif task.task_id in seen_ids:
                            reasons["duplicate_task_id"] += 1
                        elif _request_key(task.user_request) in seen_requests:
                            reasons["duplicate_request"] += 1
                        else:
                            sig = signature(task)
                            if seen_sigs[sig] >= sig_cap:
                                reasons["signature_capped"] += 1
                            else:
                                seen_ids.add(task.task_id)
                                seen_requests.add(_request_key(task.user_request))
                                seen_sigs[sig] += 1
                                kept.append(task)
                                per_author[who]["kept"] += 1
                                ckpt.write(task.model_dump_json() + "\n")
                                if len(kept) % 25 == 0:
                                    ckpt.flush()

                        progress.update(
                            bar,
                            completed=len(kept),
                            stats=(
                                f"{done} calls · yield {len(kept) / done:.0%} · "
                                + " ".join(
                                    f"{k[:3]} {per_author[k]['kept']}"
                                    for k in author_keys
                                )
                            ),
                        )
                        if len(kept) < target:
                            dispatch()
            except KeyboardInterrupt:
                # Keep whatever is already verified rather than losing the run.
                interrupted = True
                console.print(
                    f"\n[yellow]interrupted — writing the {len(kept)} tasks "
                    "verified so far[/]"
                )
            finally:
                ckpt.flush()
                for f in futures:
                    f.cancel()

    if len(kept) < target and not interrupted:
        console.print(
            f"[yellow]stopped at {len(kept)}/{target}: exhausted "
            f"{len(plan)} authoring attempts. Raise --attempt-multiplier "
            f"(currently {max_attempt_multiplier}) to push closer to target.[/]"
        )

    return write(
        kept, reasons, seen_sigs, author_keys, per_author, seed,
        time.time() - started,
    )


def write(
    kept: list[TaskSpec],
    reasons: Counter,
    sigs: Counter,
    author_keys: list[str],
    per_author: dict[str, Counter],
    seed: int,
    elapsed: float,
) -> dict:
    kept.sort(key=lambda s: s.task_id)
    tasks_path = RAW_DIR / "tasks.jsonl"
    sft_path = PROCESSED_DIR / "sft.jsonl"

    n_sft = 0
    with tasks_path.open("w") as f_tasks, sft_path.open("w") as f_sft:
        for spec in kept:
            f_tasks.write(spec.model_dump_json() + "\n")
            # Every verified accepted command is a training target; the model
            # should learn that several different commands are all correct.
            for cmd in dict.fromkeys(spec.accepted_commands):
                f_sft.write(
                    json.dumps(
                        {
                            "task_id": spec.task_id,
                            "template_family": spec.provenance.template_family,
                            "messages": [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": spec.user_request},
                                {"role": "assistant", "content": cmd},
                            ],
                        }
                    )
                    + "\n"
                )
                n_sft += 1

    families = Counter(s.provenance.template_family for s in kept)
    summary = {
        "n_tasks": len(kept),
        "n_sft_records": n_sft,
        "n_procedural": sum(v for k, v in families.items() if k != "llm_authored"),
        "n_llm_authored": families.get("llm_authored", 0),
        "family_distribution": dict(families.most_common()),
        "distinct_signatures": len(sigs),
        "top_signatures": sigs.most_common(8),
        "rejection_reasons": dict(reasons.most_common()),
        "author_models": {k: PROVIDERS[k].model for k in author_keys},
        "per_author": {
            k: {
                "attempts": per_author[k]["attempts"],
                "kept": per_author[k]["kept"],
                "yield": round(
                    per_author[k]["kept"] / max(per_author[k]["attempts"], 1), 3
                ),
            }
            for k in author_keys
        },
        "seed": seed,
        "elapsed_s": round(elapsed, 1),
    }
    (PROCESSED_DIR / "corpus_summary.json").write_text(json.dumps(summary, indent=2))

    table = Table(title="Corpus")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for key in (
        "n_tasks", "n_sft_records", "n_llm_authored", "distinct_signatures",
        "elapsed_s",
    ):
        table.add_row(key, str(summary[key]))
    console.print(table)

    if len(author_keys) > 1:
        fam = Table(title="Authoring by model family")
        fam.add_column("provider")
        fam.add_column("model")
        fam.add_column("attempts", justify="right")
        fam.add_column("kept", justify="right")
        fam.add_column("yield", justify="right")
        fam.add_column("share", justify="right")
        total_kept = sum(per_author[k]["kept"] for k in author_keys) or 1
        for key in author_keys:
            stats = per_author[key]
            fam.add_row(
                key,
                PROVIDERS[key].model,
                str(stats["attempts"]),
                str(stats["kept"]),
                f"{stats['kept'] / max(stats['attempts'], 1):.0%}",
                f"{stats['kept'] / total_kept:.0%}",
            )
        console.print(fam)

    rej = Table(title="Outcome of each authoring attempt")
    rej.add_column("reason")
    rej.add_column("count", justify="right")
    for reason, count in reasons.most_common(14):
        rej.add_row(reason, str(count))
    console.print(rej)
    console.print(f"\ntasks -> {tasks_path}\nsft   -> {sft_path}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a large verified task corpus")
    ap.add_argument("n", nargs="?", type=int, default=5000)
    ap.add_argument(
        "--author", default="gemini,openai,anthropic",
        help=(
            "comma-separated author families, round-robined evenly "
            f"(available: {','.join(sorted(PROVIDERS))})"
        ),
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--procedural-fraction", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument(
        "--attempt-multiplier", type=float, default=2.0,
        help="authoring attempts per requested task (covers rejects and dupes)",
    )
    ap.add_argument(
        "--no-progress", action="store_true",
        help="disable the live bar (for log files and non-interactive runs)",
    )
    args = ap.parse_args()

    author_keys = [k.strip() for k in args.author.split(",") if k.strip()]
    unknown = [k for k in author_keys if k not in PROVIDERS]
    if unknown:
        raise SystemExit(f"unknown author(s) {unknown}; known: {sorted(PROVIDERS)}")

    build(
        n_total=args.n,
        author_keys=author_keys,
        seed=args.seed,
        procedural_fraction=args.procedural_fraction,
        workers=args.workers,
        max_attempt_multiplier=args.attempt_multiplier,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()
